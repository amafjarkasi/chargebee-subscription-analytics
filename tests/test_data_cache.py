"""Unit tests for chargebee_mapper.data_cache module."""

import json
import tempfile
from pathlib import Path

import pytest

from chargebee_mapper.data_cache import DataCache, AnalysisDataLoader


class TestDataCache:
    """Tests for DataCache class."""

    @pytest.fixture
    def cache_with_data(self, temp_dir):
        """Create a cache with sample data files."""
        json_dir = temp_dir / "json"
        json_dir.mkdir()
        
        # Create sample data files
        customers = [
            {"id": "cust_1", "name": "Alice"},
            {"id": "cust_2", "name": "Bob"},
        ]
        invoices = [
            {"id": "inv_1", "customer_id": "cust_1", "amount": 100},
            {"id": "inv_2", "customer_id": "cust_2", "amount": 200},
        ]
        
        with open(json_dir / "customers.json", "w") as f:
            json.dump(customers, f)
        with open(json_dir / "invoices.json", "w") as f:
            json.dump(invoices, f)
        
        return DataCache(json_dir)

    def test_init(self, temp_dir):
        """Test cache initialization."""
        cache = DataCache(temp_dir)
        
        assert cache.json_dir == temp_dir
        assert len(cache) == 0

    def test_get_loads_data(self, cache_with_data):
        """Test that get loads data from disk."""
        customers = cache_with_data.get("customers")
        
        assert len(customers) == 2
        assert customers[0]["id"] == "cust_1"

    def test_get_caches_data(self, cache_with_data):
        """Test that get caches loaded data."""
        # First load
        customers1 = cache_with_data.get("customers")
        # Second load should return same object
        customers2 = cache_with_data.get("customers")
        
        assert customers1 is customers2
        assert "customers" in cache_with_data

    def test_get_missing_file(self, temp_dir):
        """Test get returns empty list for missing file."""
        cache = DataCache(temp_dir)
        
        result = cache.get("nonexistent")
        
        assert result == []

    def test_preload(self, cache_with_data):
        """Test preloading multiple entities."""
        cache_with_data.preload(["customers", "invoices"])
        
        assert "customers" in cache_with_data
        assert "invoices" in cache_with_data

    def test_invalidate_single(self, cache_with_data):
        """Test invalidating a single entity."""
        cache_with_data.get("customers")
        assert "customers" in cache_with_data
        
        cache_with_data.invalidate("customers")
        
        assert "customers" not in cache_with_data

    def test_invalidate_all(self, cache_with_data):
        """Test invalidating all entities."""
        cache_with_data.get("customers")
        cache_with_data.get("invoices")
        
        cache_with_data.invalidate()
        
        assert len(cache_with_data) == 0

    def test_stats(self, cache_with_data):
        """Test cache statistics."""
        cache_with_data.get("customers")
        cache_with_data.get("customers")  # Second access
        cache_with_data.get("invoices")
        
        stats = cache_with_data.stats()
        
        assert stats["entities_cached"] == 2
        assert stats["total_records"] == 4
        assert stats["total_accesses"] == 3
        assert stats["cache_hits"] == 1  # One cache hit for customers
        assert stats["cache_hit_rate"] == 1 / 3

    def test_contains(self, cache_with_data):
        """Test __contains__ method."""
        assert "customers" not in cache_with_data
        
        cache_with_data.get("customers")
        
        assert "customers" in cache_with_data

    def test_len(self, cache_with_data):
        """Test __len__ method."""
        assert len(cache_with_data) == 0
        
        cache_with_data.get("customers")
        assert len(cache_with_data) == 1
        
        cache_with_data.get("invoices")
        assert len(cache_with_data) == 2


class TestAnalysisDataLoader:
    """Tests for AnalysisDataLoader class."""

    @pytest.fixture
    def loader_with_data(self, temp_dir):
        """Create a loader with sample data files."""
        json_dir = temp_dir / "json"
        json_dir.mkdir()
        
        # Create sample data files
        data = {
            "customers": [{"id": "cust_1"}, {"id": "cust_2"}],
            "subscriptions": [
                {"id": "sub_1", "customer_id": "cust_1"},
                {"id": "sub_2", "customer_id": "cust_2"},
            ],
            "invoices": [
                {"id": "inv_1", "customer_id": "cust_1"},
                {"id": "inv_2", "customer_id": "cust_1"},
                {"id": "inv_3", "customer_id": "cust_2"},
            ],
            "transactions": [
                {"id": "txn_1", "customer_id": "cust_1"},
            ],
        }
        
        for name, records in data.items():
            with open(json_dir / f"{name}.json", "w") as f:
                json.dump(records, f)
        
        return AnalysisDataLoader(temp_dir)

    def test_property_access(self, loader_with_data):
        """Test property-based data access."""
        customers = loader_with_data.customers
        subscriptions = loader_with_data.subscriptions
        
        assert len(customers) == 2
        assert len(subscriptions) == 2

    def test_get_method(self, loader_with_data):
        """Test get method for arbitrary entities."""
        customers = loader_with_data.get("customers")
        
        assert len(customers) == 2

    def test_get_customer_by_id(self, loader_with_data):
        """Test looking up customer by ID."""
        customer = loader_with_data.get_customer_by_id("cust_1")
        
        assert customer is not None
        assert customer["id"] == "cust_1"

    def test_get_customer_by_id_not_found(self, loader_with_data):
        """Test looking up non-existent customer."""
        customer = loader_with_data.get_customer_by_id("nonexistent")
        
        assert customer is None

    def test_get_subscriptions_for_customer(self, loader_with_data):
        """Test getting subscriptions for a customer."""
        subs = loader_with_data.get_subscriptions_for_customer("cust_1")
        
        assert len(subs) == 1
        assert subs[0]["id"] == "sub_1"

    def test_get_invoices_for_customer(self, loader_with_data):
        """Test getting invoices for a customer."""
        invoices = loader_with_data.get_invoices_for_customer("cust_1")
        
        assert len(invoices) == 2

    def test_get_transactions_for_customer(self, loader_with_data):
        """Test getting transactions for a customer."""
        txns = loader_with_data.get_transactions_for_customer("cust_1")
        
        assert len(txns) == 1
        assert txns[0]["id"] == "txn_1"
