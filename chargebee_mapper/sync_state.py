"""Incremental sync state tracking for efficient delta fetches.

This module tracks the last sync timestamp for each entity type,
enabling incremental fetches using Chargebee's `updated_at[after]` filter.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("chargebee_mapper.sync_state")


@dataclass
class EntitySyncState:
    """State for a single entity type."""
    
    entity_key: str
    last_sync_at: datetime | None = None
    last_updated_at: int | None = None  # Unix timestamp of most recent record
    last_record_count: int = 0
    total_synced: int = 0


@dataclass
class SyncState:
    """Overall sync state for all entities."""
    
    entities: dict[str, EntitySyncState] = field(default_factory=dict)
    last_full_sync: datetime | None = None
    sync_mode: str = "full"  # "full" or "incremental"
    
    def get_entity_state(self, entity_key: str) -> EntitySyncState:
        """Get or create state for an entity."""
        if entity_key not in self.entities:
            self.entities[entity_key] = EntitySyncState(entity_key=entity_key)
        return self.entities[entity_key]
    
    def update_entity_state(
        self,
        entity_key: str,
        record_count: int,
        max_updated_at: int | None = None,
    ) -> None:
        """Update sync state for an entity after fetch."""
        state = self.get_entity_state(entity_key)
        state.last_sync_at = datetime.now(timezone.utc)
        state.last_record_count = record_count
        state.total_synced += record_count
        
        if max_updated_at is not None:
            # Keep the most recent updated_at timestamp
            if state.last_updated_at is None or max_updated_at > state.last_updated_at:
                state.last_updated_at = max_updated_at


class SyncStateManager:
    """Manages persistent sync state using SQLite."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.db_path = output_dir / "sync_state.db"
        self._conn: sqlite3.Connection | None = None
        self._state: SyncState | None = None
    
    def open(self) -> None:
        """Open the sync state database."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        self._state = self._load_state()
        logger.debug("Sync state opened: %s", self.db_path)
    
    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._state = None
    
    @property
    def state(self) -> SyncState:
        """Get the current sync state."""
        if self._state is None:
            raise RuntimeError("SyncStateManager not opened")
        return self._state
    
    def _create_tables(self) -> None:
        """Create sync state tables if they don't exist."""
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sync_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            
            CREATE TABLE IF NOT EXISTS entity_sync_state (
                entity_key TEXT PRIMARY KEY,
                last_sync_at TEXT,
                last_updated_at INTEGER,
                last_record_count INTEGER DEFAULT 0,
                total_synced INTEGER DEFAULT 0
            );
        """)
        self._conn.commit()
    
    def _load_state(self) -> SyncState:
        """Load sync state from database."""
        assert self._conn is not None
        state = SyncState()
        
        # Load meta info
        cursor = self._conn.execute(
            "SELECT key, value FROM sync_meta WHERE key IN ('last_full_sync', 'sync_mode')"
        )
        for row in cursor:
            if row["key"] == "last_full_sync" and row["value"]:
                state.last_full_sync = datetime.fromisoformat(row["value"])
            elif row["key"] == "sync_mode":
                state.sync_mode = row["value"]
        
        # Load entity states
        cursor = self._conn.execute(
            "SELECT entity_key, last_sync_at, last_updated_at, last_record_count, total_synced "
            "FROM entity_sync_state"
        )
        for row in cursor:
            entity_state = EntitySyncState(
                entity_key=row["entity_key"],
                last_sync_at=datetime.fromisoformat(row["last_sync_at"]) if row["last_sync_at"] else None,
                last_updated_at=row["last_updated_at"],
                last_record_count=row["last_record_count"] or 0,
                total_synced=row["total_synced"] or 0,
            )
            state.entities[row["entity_key"]] = entity_state
        
        logger.info(
            "Loaded sync state: %d entities, last full sync: %s",
            len(state.entities),
            state.last_full_sync.isoformat() if state.last_full_sync else "never",
        )
        return state
    
    def save(self) -> None:
        """Save current sync state to database."""
        assert self._conn is not None
        assert self._state is not None
        
        # Save meta info
        self._conn.execute(
            "INSERT OR REPLACE INTO sync_meta (key, value) VALUES (?, ?)",
            ("last_full_sync", self._state.last_full_sync.isoformat() if self._state.last_full_sync else None),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO sync_meta (key, value) VALUES (?, ?)",
            ("sync_mode", self._state.sync_mode),
        )
        
        # Save entity states
        for entity_key, entity_state in self._state.entities.items():
            self._conn.execute(
                """
                INSERT OR REPLACE INTO entity_sync_state 
                (entity_key, last_sync_at, last_updated_at, last_record_count, total_synced)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    entity_key,
                    entity_state.last_sync_at.isoformat() if entity_state.last_sync_at else None,
                    entity_state.last_updated_at,
                    entity_state.last_record_count,
                    entity_state.total_synced,
                ),
            )
        
        self._conn.commit()
        logger.debug("Sync state saved")
    
    def get_incremental_filter(self, entity_key: str) -> dict[str, Any] | None:
        """Get the filter parameters for an incremental fetch.
        
        Returns None if a full fetch is needed (no previous sync state).
        Returns a dict with 'updated_at[after]' filter if incremental is possible.
        """
        if self._state is None:
            return None
        
        entity_state = self._state.entities.get(entity_key)
        if not entity_state or not entity_state.last_updated_at:
            logger.debug("%s: No previous sync state, full fetch required", entity_key)
            return None
        
        # Return filter for incremental fetch
        # Subtract 60 seconds as safety margin for clock skew
        after_timestamp = max(0, entity_state.last_updated_at - 60)
        logger.info(
            "%s: Incremental fetch, updated_at > %d (last sync: %s)",
            entity_key,
            after_timestamp,
            entity_state.last_sync_at.isoformat() if entity_state.last_sync_at else "never",
        )
        return {"updated_at[after]": after_timestamp}
    
    def mark_full_sync_complete(self) -> None:
        """Mark that a full sync has completed."""
        if self._state:
            self._state.last_full_sync = datetime.now(timezone.utc)
            self._state.sync_mode = "incremental"
            self.save()
    
    def get_sync_summary(self) -> dict[str, Any]:
        """Get a summary of the current sync state."""
        if not self._state:
            return {"status": "not_initialized"}
        
        return {
            "last_full_sync": self._state.last_full_sync.isoformat() if self._state.last_full_sync else None,
            "sync_mode": self._state.sync_mode,
            "entities_tracked": len(self._state.entities),
            "total_records_synced": sum(
                e.total_synced for e in self._state.entities.values()
            ),
            "entities": {
                k: {
                    "last_sync": v.last_sync_at.isoformat() if v.last_sync_at else None,
                    "last_updated_at": v.last_updated_at,
                    "last_count": v.last_record_count,
                    "total": v.total_synced,
                }
                for k, v in self._state.entities.items()
            },
        }


def extract_max_updated_at(records: list[dict]) -> int | None:
    """Extract the maximum updated_at timestamp from a list of records.
    
    Chargebee uses Unix timestamps for updated_at fields.
    """
    if not records:
        return None
    
    max_ts: int | None = None
    for record in records:
        # Try common updated_at field names
        for field in ("updated_at", "modified_at", "changed_at"):
            ts = record.get(field)
            if ts is not None and isinstance(ts, int):
                if max_ts is None or ts > max_ts:
                    max_ts = ts
                break
    
    return max_ts
