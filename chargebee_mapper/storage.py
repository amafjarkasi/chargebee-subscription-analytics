"""Storage backends for writing fetched data to JSON files and SQLite."""

import json
import logging
import re
import sqlite3
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

    def open(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_meta_table()
        logger.info("SQLite database opened: %s", self.db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("SQLite database closed")

    def _create_meta_table(self) -> None:
        """Create a metadata table to track the export run."""
        assert self._conn is not None
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
            logger.warning(
                "Table name '%s' not in entity whitelist but matches safe pattern",
                table_name
            )
            return table_name
        
        raise ValueError(
            f"Invalid table name '{table_name}': must be a known entity key or "
            f"match pattern [a-zA-Z_][a-zA-Z0-9_]* with max 64 characters"
        )

    def _create_entity_table(self, table_name: str) -> None:
        """Create a table for an entity type with a generic schema."""
        assert self._conn is not None
        # Validate table name against whitelist (prevents SQL injection)
        safe_name = self._validate_table_name(table_name)
        # Using bracket notation for identifiers is safe after whitelist validation
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

    def write_entity(self, result: FetchResult) -> int:
        """Write all records for an entity type to the database. Returns rows written."""
        assert self._conn is not None
        table_name = result.entity.key
        self._create_entity_table(table_name)
        # Validate table name (already validated in _create_entity_table, but explicit for safety)
        safe_name = self._validate_table_name(table_name)

        rows_written = 0
        batch: list[tuple[str, str, int | None, int | None, str | None]] = []

        for record in result.records:
            record_id = str(record.get("id", f"_no_id_{rows_written}"))
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
                logger.debug("SQLite %s: inserted batch of %d rows", safe_name, len(batch))
                batch.clear()

        if batch:
            self._insert_batch(safe_name, batch)
            logger.debug("SQLite %s: inserted final batch of %d rows", safe_name, len(batch))

        logger.info("SQLite table written: %s (%d rows)", safe_name, rows_written)
        return rows_written

    def _insert_batch(
        self,
        table_name: str,
        batch: list[tuple[str, str, int | None, int | None, str | None]],
    ) -> None:
        assert self._conn is not None
        self._conn.executemany(
            f"""INSERT OR REPLACE INTO [{table_name}]
                (id, data_json, created_at, updated_at, status)
                VALUES (?, ?, ?, ?, ?)""",
            batch,
        )
        self._conn.commit()

    def write_meta(self, results: dict[str, FetchResult], elapsed: float) -> None:
        """Write export metadata."""
        assert self._conn is not None
        meta = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": str(round(elapsed, 2)),
            "total_records": str(sum(r.count for r in results.values())),
            "total_api_calls": str(sum(r.api_calls for r in results.values())),
        }
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

    def write_all(self, results: dict[str, FetchResult], elapsed: float) -> dict[str, Any]:
        """Write all results to both JSON and SQLite. Returns a stats dict."""
        stats: dict[str, Any] = {"json_files": [], "sqlite_tables": [], "errors": []}

        for key, result in results.items():
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
