"""Unit tests for chargebee_mapper.storage module."""

import json
import sqlite3

import pytest

from chargebee_mapper.client import FetchResult
from chargebee_mapper.entities import EntityDef
from chargebee_mapper.storage import (
    JsonWriter,
    SqliteWriter,
    StorageManager,
    VALID_TABLE_NAMES,
    _VALID_IDENTIFIER_PATTERN,
)


class TestValidTableNames:
    """Tests for table name validation."""

    def test_valid_table_names_contains_known_entities(self):
        """Test that VALID_TABLE_NAMES contains expected entity keys."""
        expected_keys = [
            "customers",
            "subscriptions",
            "invoices",
            "transactions",
            "items",
            "item_prices",
            "coupons",
            "events",
        ]
        for key in expected_keys:
            assert key in VALID_TABLE_NAMES, f"Expected '{key}' in VALID_TABLE_NAMES"

    def test_valid_table_names_count(self):
        """Test that VALID_TABLE_NAMES has expected number of entries."""
        # Based on entities.py, should have 36 entities
        assert len(VALID_TABLE_NAMES) == 36

    def test_identifier_pattern_valid_names(self):
        """Test that valid identifier pattern matches safe names."""
        valid_names = [
            "customers",
            "item_prices",
            "_internal",
            "A1B2C3",
            "CamelCase",
        ]
        for name in valid_names:
            assert _VALID_IDENTIFIER_PATTERN.match(name), f"Pattern should match '{name}'"

    def test_identifier_pattern_invalid_names(self):
        """Test that valid identifier pattern rejects unsafe names."""
        invalid_names = [
            "123abc",  # starts with digit
            "table-name",  # contains hyphen
            "table name",  # contains space
            "table;drop",  # contains semicolon
            "",  # empty
        ]
        for name in invalid_names:
            assert not _VALID_IDENTIFIER_PATTERN.match(name), f"Pattern should not match '{name}'"


class TestSqliteWriter:
    """Tests for SqliteWriter class."""

    def test_validate_table_name_whitelist(self, temp_dir):
        """Test that whitelisted table names are accepted."""
        writer = SqliteWriter(temp_dir)
        
        # Should not raise for valid entity keys
        assert writer._validate_table_name("customers") == "customers"
        assert writer._validate_table_name("subscriptions") == "subscriptions"
        assert writer._validate_table_name("item_prices") == "item_prices"

    def test_validate_table_name_safe_pattern(self, temp_dir):
        """Test that safe pattern names are accepted with warning."""
        writer = SqliteWriter(temp_dir)
        
        # Should accept safe patterns not in whitelist
        assert writer._validate_table_name("custom_table") == "custom_table"

    def test_validate_table_name_rejects_unsafe(self, temp_dir):
        """Test that unsafe table names are rejected."""
        writer = SqliteWriter(temp_dir)
        
        with pytest.raises(ValueError) as exc_info:
            writer._validate_table_name("table;DROP TABLE customers")
        assert "Invalid table name" in str(exc_info.value)

    def test_validate_table_name_rejects_injection(self, temp_dir):
        """Test that SQL injection attempts are rejected."""
        writer = SqliteWriter(temp_dir)
        
        injection_attempts = [
            "customers; DROP TABLE",
            "customers--",
            "customers' OR '1'='1",
            "customers] DROP TABLE [",
            "../../../etc/passwd",
        ]
        
        for attempt in injection_attempts:
            with pytest.raises(ValueError):
                writer._validate_table_name(attempt)

    def test_validate_table_name_length_limit(self, temp_dir):
        """Test that very long table names are rejected."""
        writer = SqliteWriter(temp_dir)
        
        # 65 character name should be rejected
        long_name = "a" * 65
        with pytest.raises(ValueError):
            writer._validate_table_name(long_name)

    def test_open_and_close(self, temp_dir):
        """Test database open and close operations."""
        writer = SqliteWriter(temp_dir)
        
        writer.open()
        assert writer._conn is not None
        assert writer.db_path.exists()
        
        writer.close()
        assert writer._conn is None

    def test_meta_table_created(self, temp_dir):
        """Test that _meta table is created on open."""
        writer = SqliteWriter(temp_dir)
        writer.open()
        
        # Check _meta table exists
        cursor = writer._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_meta'"
        )
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == "_meta"
        
        writer.close()

    def test_create_entity_table(self, temp_dir):
        """Test entity table creation."""
        writer = SqliteWriter(temp_dir)
        writer.open()
        
        writer._create_entity_table("customers")
        
        # Check table was created with expected schema
        cursor = writer._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='customers'"
        )
        result = cursor.fetchone()
        assert result is not None
        assert "id TEXT PRIMARY KEY" in result[0]
        assert "data_json TEXT NOT NULL" in result[0]
        
        writer.close()

    def test_write_entity(self, temp_dir, sample_entity):
        """Test writing entity records to database."""
        writer = SqliteWriter(temp_dir)
        writer.open()
        
        result = FetchResult(sample_entity)
        result.records = [
            {"id": "cust_1", "name": "Alice", "status": "active"},
            {"id": "cust_2", "name": "Bob", "status": "inactive"},
        ]
        
        rows_written = writer.write_entity(result)
        
        assert rows_written == 2
        
        # Verify data was written
        cursor = writer._conn.execute("SELECT COUNT(*) FROM customers")
        count = cursor.fetchone()[0]
        assert count == 2
        
        # Verify data content
        cursor = writer._conn.execute("SELECT id, status FROM customers ORDER BY id")
        rows = cursor.fetchall()
        assert rows[0] == ("cust_1", "active")
        assert rows[1] == ("cust_2", "inactive")
        
        writer.close()


class TestJsonWriter:
    """Tests for JsonWriter class."""

    def test_init_creates_directory(self, temp_dir):
        """Test that JsonWriter creates the json subdirectory."""
        writer = JsonWriter(temp_dir)
        assert writer.json_dir.exists()
        assert writer.json_dir.name == "json"

    def test_write_entity(self, temp_dir, sample_entity):
        """Test writing entity records to JSON file."""
        writer = JsonWriter(temp_dir)
        
        result = FetchResult(sample_entity)
        result.records = [
            {"id": "cust_1", "name": "Alice"},
            {"id": "cust_2", "name": "Bob"},
        ]
        
        filepath = writer.write_entity(result)
        
        assert filepath.exists()
        assert filepath.name == "customers.json"
        
        with open(filepath) as f:
            data = json.load(f)
        
        assert len(data) == 2
        assert data[0]["id"] == "cust_1"
        assert data[1]["name"] == "Bob"


class TestStorageManager:
    """Tests for StorageManager class."""

    def test_init(self, temp_dir):
        """Test StorageManager initialization."""
        manager = StorageManager(temp_dir)
        
        assert manager.output_dir == temp_dir
        assert isinstance(manager.json_writer, JsonWriter)
        assert isinstance(manager.sqlite_writer, SqliteWriter)

    def test_open_and_close(self, temp_dir):
        """Test StorageManager open and close."""
        manager = StorageManager(temp_dir)
        
        manager.open()
        assert manager.sqlite_writer._conn is not None
        
        manager.close()
        assert manager.sqlite_writer._conn is None
