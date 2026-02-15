"""Storage backends for writing fetched data to JSON files and SQLite."""

import asyncio
import json
import logging
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import FetchResult
from .entities import ALL_ENTITIES

logger = logging.getLogger("chargebee_mapper.storage")

# Whitelist of valid table names derived from entity definitions
# This prevents SQL injection by only allowing known, safe identifiers
VALID_TABLE_NAMES: frozenset[str] = frozenset(entity.key for entity in ALL_ENTITIES)

# Regex pattern for valid SQLite identifiers (alphanumeric + underscore, not starting with digit)
_VALID_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class JsonWriter:
    """Writes entity data to per-entity JSON files."""

    def __init__(self, output_dir: Path):
        self.json_dir = output_dir / "json"
        self.json_dir.mkdir(parents=True, exist_ok=True)

    def write_entity(self, result: FetchResult) -> Path:
        """Write records for a single entity type to a JSON file."""
        filepath = self.json_dir / f"{result.entity.key}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.records, f, indent=2, default=str, ensure_ascii=False)
        size_kb = filepath.stat().st_size / 1024
        logger.info("JSON written: %s (%d records, %.1f KB)", filepath.name, result.count, size_kb)
        return filepath

    def write_summary(self, results: dict[str, FetchResult], elapsed: float, output_dir: Path) -> Path:
        """Write a summary JSON file with counts and metadata."""
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "total_records": sum(r.count for r in results.values()),
            "total_api_calls": sum(r.api_calls for r in results.values()),
            "entities": {},
        }

        for key, result in sorted(results.items()):
            summary["entities"][key] = {
                "name": result.entity.name,
                "record_count": result.count,
                "pages_fetched": result.pages_fetched,
                "api_calls": result.api_calls,
                "elapsed_seconds": round(result.elapsed, 2),
                "success": result.success,
                "errors": result.errors if result.errors else None,
            }

        filepath = output_dir / "summary.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
        logger.info("Summary written: %s", filepath)
        return filepath


class SqliteWriter:
    """Writes entity data to a SQLite database."""

    def __init__(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = output_dir / "chargebee_data.db"
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)  # Serialise DB writes
        self._tables_created: set[str] = set()

    def open(self) -> None:
        # check_same_thread=False because we access it from executor thread
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_meta_table()
        logger.info("SQLite database opened: %s", self.db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            self._executor.shutdown(wait=True)
            logger.debug("SQLite database closed")

    def _create_meta_table(self) -> None:
        """Create a metadata table to track the export run."""
        assert self._conn is not None
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS _meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            self._conn.commit()

    def _validate_table_name(self, table_name: str) -> str:
        """Validate and return a safe table name.
        
        Uses a whitelist approach to prevent SQL injection:
        1. Check against known entity keys from the registry
        2. Validate against safe identifier pattern as fallback
        
        Raises ValueError if the table name is not valid.
        """
        # Primary check: whitelist of known entity keys
        if table_name in VALID_TABLE_NAMES:
            return table_name
        
        # Fallback: validate against safe identifier pattern
        # This allows for future extensions while maintaining security
        if _VALID_IDENTIFIER_PATTERN.match(table_name) and len(table_name) <= 64:
            return table_name
        
        raise ValueError(
            f"Invalid table name '{table_name}': must be a known entity key or "
            f"match pattern [a-zA-Z_][a-zA-Z0-9_]* with max 64 characters"
        )

    def _create_entity_table(self, table_name: str) -> None:
        """Create a table for an entity type with a generic schema."""
        assert self._conn is not None
        if table_name in self._tables_created:
            return

        # Validate table name against whitelist (prevents SQL injection)
        safe_name = self._validate_table_name(table_name)
        # Using bracket notation for identifiers is safe after whitelist validation
        with self._lock:
            if table_name in self._tables_created:
                return
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS [{safe_name}] (
                    id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    created_at INTEGER,
                    updated_at INTEGER,
                    status TEXT
                )
            """)
            self._conn.commit()
            self._tables_created.add(table_name)

    def write_entity(self, result: FetchResult) -> int:
        """Write all records for an entity type to the database. Returns rows written."""
        return self.write_records(result.entity.key, result.records)

    def write_records(self, table_name: str, records: list[dict[str, Any]]) -> int:
        """Write a batch of records to the database."""
        assert self._conn is not None
        self._create_entity_table(table_name)
        safe_name = self._validate_table_name(table_name)

        rows_written = 0
        batch: list[tuple[str, str, int | None, int | None, str | None]] = []

        for record in records:
            # Generate a pseudo-ID if none exists
            record_id = str(record.get("id", f"_no_id_{rows_written}_{id(record)}"))

            data_json = json.dumps(record, default=str, ensure_ascii=False)
            created_at = record.get("created_at")
            updated_at = record.get("updated_at")
            status = record.get("status")

            if isinstance(status, str):
                pass
            elif status is not None:
                status = str(status)

            batch.append((record_id, data_json, created_at, updated_at, status))
            rows_written += 1

            if len(batch) >= 500:
                self._insert_batch(safe_name, batch)
                batch.clear()

        if batch:
            self._insert_batch(safe_name, batch)

        return rows_written

    def _insert_batch(
        self,
        table_name: str,
        batch: list[tuple[str, str, int | None, int | None, str | None]],
    ) -> None:
        assert self._conn is not None
        with self._lock:
            self._conn.executemany(
                f"""INSERT OR REPLACE INTO [{table_name}]
                    (id, data_json, created_at, updated_at, status)
                    VALUES (?, ?, ?, ?, ?)""",
                batch,
            )
            self._conn.commit()

    async def write_records_async(self, table_name: str, records: list[dict[str, Any]]) -> int:
        """Async wrapper for write_records running in executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self.write_records,
            table_name,
            records
        )

    def get_all_records(self, table_name: str) -> list[dict[str, Any]]:
        """Retrieve all records for a table as a list of dicts."""
        assert self._conn is not None
        safe_name = self._validate_table_name(table_name)

        # Check if table exists first to avoid error
        with self._lock:
            cursor = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (safe_name,)
            )
            if not cursor.fetchone():
                return []

            cursor = self._conn.execute(f"SELECT data_json FROM [{safe_name}]")
            rows = cursor.fetchall()

        records = []
        for (data_json,) in rows:
            try:
                records.append(json.loads(data_json))
            except json.JSONDecodeError:
                logger.warning("Failed to decode JSON from DB for table %s", safe_name)
        return records

    def get_all_ids(self, table_name: str) -> list[str]:
        """Retrieve all IDs for a table."""
        assert self._conn is not None
        safe_name = self._validate_table_name(table_name)

        with self._lock:
            cursor = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (safe_name,)
            )
            if not cursor.fetchone():
                return []

            cursor = self._conn.execute(f"SELECT id FROM [{safe_name}]")
            return [row[0] for row in cursor.fetchall()]

    def write_meta(self, results: dict[str, FetchResult], elapsed: float) -> None:
        """Write export metadata."""
        assert self._conn is not None
        meta = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": str(round(elapsed, 2)),
            "total_records": str(sum(r.count for r in results.values())),
            "total_api_calls": str(sum(r.api_calls for r in results.values())),
        }
        with self._lock:
            for key, value in meta.items():
                self._conn.execute(
                    "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
                    (key, value),
                )
            self._conn.commit()
        logger.debug("SQLite metadata written")


class StorageManager:
    """Coordinates writing to both JSON and SQLite outputs."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.json_writer = JsonWriter(output_dir)
        self.sqlite_writer = SqliteWriter(output_dir)

    def open(self) -> None:
        self.sqlite_writer.open()

    def close(self) -> None:
        self.sqlite_writer.close()

    async def stream_to_sqlite(self, entity_key: str, records: list[dict[str, Any]]) -> int:
        """Stream a batch of records to SQLite asynchronously."""
        if not records:
            return 0
        try:
            return await self.sqlite_writer.write_records_async(entity_key, records)
        except Exception as e:
            logger.error("Error streaming to SQLite for %s: %s", entity_key, e)
            return 0

    def get_all_ids(self, entity_key: str) -> list[str]:
        """Get all IDs for an entity from SQLite."""
        return self.sqlite_writer.get_all_ids(entity_key)

    def export_json_from_sqlite(self, entity_key: str) -> Path:
        """Read records from SQLite and write to JSON file."""
        records = self.sqlite_writer.get_all_records(entity_key)

        filepath = self.json_writer.json_dir / f"{entity_key}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str, ensure_ascii=False)

        size_kb = filepath.stat().st_size / 1024
        logger.info("JSON exported from SQLite: %s (%d records, %.1f KB)", filepath.name, len(records), size_kb)
        return filepath

    def write_all(self, results: dict[str, FetchResult], elapsed: float) -> dict[str, Any]:
        """Write all results to both JSON and SQLite. Returns a stats dict."""
        stats: dict[str, Any] = {"json_files": [], "sqlite_tables": [], "errors": []}

        for key, result in results.items():
            # If records are present in memory, write them as before
            if result.records:
                try:
                    json_path = self.json_writer.write_entity(result)
                    stats["json_files"].append(str(json_path))
                except Exception as e:
                    stats["errors"].append(f"JSON write error for {key}: {e}")
                    logger.error("JSON write error for %s: %s", key, e)

                try:
                    self.sqlite_writer.write_entity(result)
                    stats["sqlite_tables"].append(key)
                except Exception as e:
                    stats["errors"].append(f"SQLite write error for {key}: {e}")
                    logger.error("SQLite write error for %s: %s", key, e)

            # If records are NOT in memory but count > 0, assume they are in SQLite (streaming mode)
            elif result.count > 0:
                logger.info("Exporting %s from SQLite to JSON (streaming mode)", key)
                try:
                    # Data is already in SQLite, just note it
                    stats["sqlite_tables"].append(key)

                    # Export to JSON from SQLite
                    json_path = self.export_json_from_sqlite(key)
                    stats["json_files"].append(str(json_path))
                except Exception as e:
                    stats["errors"].append(f"Export error for {key}: {e}")
                    logger.error("Export error for %s: %s", key, e)

            else:
                logger.debug("Skipping write for %s (0 records)", key)

        try:
            summary_path = self.json_writer.write_summary(results, elapsed, self.output_dir)
            stats["summary_file"] = str(summary_path)
        except Exception as e:
            stats["errors"].append(f"Summary write error: {e}")
            logger.error("Summary write error: %s", e)

        try:
            self.sqlite_writer.write_meta(results, elapsed)
        except Exception as e:
            stats["errors"].append(f"SQLite meta write error: {e}")
            logger.error("SQLite meta write error: %s", e)

        logger.info(
            "Storage complete: %d JSON files, %d SQLite tables, %d errors",
            len(stats["json_files"]), len(stats["sqlite_tables"]), len(stats["errors"]),
        )
        return stats
