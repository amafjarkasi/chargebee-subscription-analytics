"""Unit tests for chargebee_mapper.sync_state module."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from chargebee_mapper.sync_state import (
    EntitySyncState,
    SyncState,
    SyncStateManager,
    extract_max_updated_at,
)


class TestEntitySyncState:
    """Tests for EntitySyncState dataclass."""

    def test_init(self):
        """Test basic initialization."""
        state = EntitySyncState(entity_key="customers")
        
        assert state.entity_key == "customers"
        assert state.last_sync_at is None
        assert state.last_updated_at is None
        assert state.last_record_count == 0
        assert state.total_synced == 0

    def test_with_values(self):
        """Test initialization with values."""
        now = datetime.now(timezone.utc)
        state = EntitySyncState(
            entity_key="invoices",
            last_sync_at=now,
            last_updated_at=1707750000,
            last_record_count=100,
            total_synced=500,
        )
        
        assert state.entity_key == "invoices"
        assert state.last_sync_at == now
        assert state.last_updated_at == 1707750000
        assert state.last_record_count == 100
        assert state.total_synced == 500


class TestSyncState:
    """Tests for SyncState dataclass."""

    def test_init(self):
        """Test basic initialization."""
        state = SyncState()
        
        assert state.entities == {}
        assert state.last_full_sync is None
        assert state.sync_mode == "full"

    def test_get_entity_state_creates_new(self):
        """Test that get_entity_state creates new state if not exists."""
        state = SyncState()
        
        entity_state = state.get_entity_state("customers")
        
        assert entity_state.entity_key == "customers"
        assert "customers" in state.entities

    def test_get_entity_state_returns_existing(self):
        """Test that get_entity_state returns existing state."""
        state = SyncState()
        state.entities["customers"] = EntitySyncState(
            entity_key="customers",
            total_synced=100,
        )
        
        entity_state = state.get_entity_state("customers")
        
        assert entity_state.total_synced == 100

    def test_update_entity_state(self):
        """Test updating entity state."""
        state = SyncState()
        
        state.update_entity_state("customers", record_count=50, max_updated_at=1707750000)
        
        entity_state = state.entities["customers"]
        assert entity_state.last_record_count == 50
        assert entity_state.total_synced == 50
        assert entity_state.last_updated_at == 1707750000
        assert entity_state.last_sync_at is not None

    def test_update_entity_state_keeps_max_timestamp(self):
        """Test that update keeps the maximum timestamp."""
        state = SyncState()
        
        # First update with high timestamp
        state.update_entity_state("customers", record_count=50, max_updated_at=1707750000)
        
        # Second update with lower timestamp should not overwrite
        state.update_entity_state("customers", record_count=10, max_updated_at=1707740000)
        
        entity_state = state.entities["customers"]
        assert entity_state.last_updated_at == 1707750000
        assert entity_state.total_synced == 60  # Should accumulate


class TestSyncStateManager:
    """Tests for SyncStateManager class."""

    def test_open_creates_database(self, temp_dir):
        """Test that open creates the database file."""
        manager = SyncStateManager(temp_dir)
        
        manager.open()
        
        assert manager.db_path.exists()
        assert manager._conn is not None
        
        manager.close()

    def test_save_and_load(self, temp_dir):
        """Test saving and loading state."""
        manager = SyncStateManager(temp_dir)
        manager.open()
        
        # Modify state
        manager.state.update_entity_state("customers", 100, 1707750000)
        manager.state.sync_mode = "incremental"
        manager.save()
        manager.close()
        
        # Reopen and verify
        manager2 = SyncStateManager(temp_dir)
        manager2.open()
        
        assert manager2.state.sync_mode == "incremental"
        assert "customers" in manager2.state.entities
        assert manager2.state.entities["customers"].last_updated_at == 1707750000
        
        manager2.close()

    def test_get_incremental_filter_no_state(self, temp_dir):
        """Test incremental filter returns None when no state."""
        manager = SyncStateManager(temp_dir)
        manager.open()
        
        filter_params = manager.get_incremental_filter("customers")
        
        assert filter_params is None
        
        manager.close()

    def test_get_incremental_filter_with_state(self, temp_dir):
        """Test incremental filter returns correct params."""
        manager = SyncStateManager(temp_dir)
        manager.open()
        
        manager.state.update_entity_state("customers", 100, 1707750000)
        
        filter_params = manager.get_incremental_filter("customers")
        
        # Should have safety margin of 60 seconds
        assert filter_params is not None
        assert "updated_at[after]" in filter_params
        assert filter_params["updated_at[after]"] == 1707750000 - 60
        
        manager.close()

    def test_mark_full_sync_complete(self, temp_dir):
        """Test marking full sync as complete."""
        manager = SyncStateManager(temp_dir)
        manager.open()
        
        manager.mark_full_sync_complete()
        
        assert manager.state.last_full_sync is not None
        assert manager.state.sync_mode == "incremental"
        
        manager.close()


class TestExtractMaxUpdatedAt:
    """Tests for extract_max_updated_at function."""

    def test_empty_records(self):
        """Test with empty records list."""
        assert extract_max_updated_at([]) is None

    def test_no_updated_at_field(self):
        """Test records without updated_at field."""
        records = [{"id": "1"}, {"id": "2"}]
        assert extract_max_updated_at(records) is None

    def test_single_record(self):
        """Test with single record."""
        records = [{"id": "1", "updated_at": 1707750000}]
        assert extract_max_updated_at(records) == 1707750000

    def test_multiple_records(self):
        """Test with multiple records, finds max."""
        records = [
            {"id": "1", "updated_at": 1707750000},
            {"id": "2", "updated_at": 1707760000},
            {"id": "3", "updated_at": 1707740000},
        ]
        assert extract_max_updated_at(records) == 1707760000

    def test_mixed_fields(self):
        """Test with mixed timestamp field names."""
        records = [
            {"id": "1", "updated_at": 1707750000},
            {"id": "2", "modified_at": 1707760000},  # Different field name
        ]
        # Should find max across different field names
        assert extract_max_updated_at(records) == 1707760000
