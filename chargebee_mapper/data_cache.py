"""Data cache for efficient analysis operations.

This module provides a caching layer to avoid repeatedly loading the same
JSON files when running multiple analyses. This is particularly useful
during "analyze all" which runs 11 different analyzers.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("chargebee_mapper.data_cache")


class DataCache:
    """Cache for loaded JSON data files.
    
    Provides a single point of loading for entity data, avoiding redundant
    file I/O operations when multiple analyzers need the same data.
    
    Usage:
        cache = DataCache(data_dir / "json")
        customers = cache.get("customers")
        invoices = cache.get("invoices")
        
        # Check what's loaded
        print(cache.stats())
    """
    
    def __init__(self, json_dir: Path):
        """Initialize the cache with the JSON data directory.
        
        Args:
            json_dir: Path to directory containing entity JSON files
        """
        self.json_dir = json_dir
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self._load_count: dict[str, int] = {}  # Track access counts
        self._file_sizes: dict[str, int] = {}  # Track file sizes
    
    def get(self, entity: str) -> list[dict[str, Any]]:
        """Get data for an entity, loading from disk if not cached.
        
        Args:
            entity: Entity key (e.g., "customers", "invoices")
            
        Returns:
            List of entity records, or empty list if file doesn't exist
        """
        if entity in self._cache:
            self._load_count[entity] = self._load_count.get(entity, 0) + 1
            logger.debug("Cache hit for %s (access #%d)", entity, self._load_count[entity])
            return self._cache[entity]
        
        # Load from disk
        file_path = self.json_dir / f"{entity}.json"
        if not file_path.exists():
            logger.warning("Data file not found: %s", file_path)
            self._cache[entity] = []
            self._load_count[entity] = 1
            return []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.warning("%s: Expected list, got %s", entity, type(data).__name__)
                data = []
            
            self._cache[entity] = data
            self._load_count[entity] = 1
            self._file_sizes[entity] = file_path.stat().st_size
            
            logger.info(
                "Loaded %s: %d records (%.1f KB)",
                entity,
                len(data),
                self._file_sizes[entity] / 1024,
            )
            return data
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse %s: %s", file_path, e)
            self._cache[entity] = []
            return []
        except Exception as e:
            logger.error("Failed to load %s: %s", file_path, e)
            self._cache[entity] = []
            return []
    
    def preload(self, entities: list[str]) -> None:
        """Preload multiple entities into cache.
        
        Args:
            entities: List of entity keys to preload
        """
        logger.info("Preloading %d entities into cache", len(entities))
        for entity in entities:
            self.get(entity)
    
    def preload_common(self) -> None:
        """Preload commonly used entities for analysis."""
        common_entities = [
            "customers",
            "subscriptions", 
            "invoices",
            "transactions",
            "credit_notes",
            "events",
            "coupons",
            "item_prices",
            "payment_sources",
        ]
        self.preload(common_entities)
    
    def invalidate(self, entity: str | None = None) -> None:
        """Invalidate cached data.
        
        Args:
            entity: Specific entity to invalidate, or None to clear all
        """
        if entity is None:
            count = len(self._cache)
            self._cache.clear()
            self._load_count.clear()
            self._file_sizes.clear()
            logger.info("Cache cleared: %d entities invalidated", count)
        elif entity in self._cache:
            del self._cache[entity]
            self._load_count.pop(entity, None)
            self._file_sizes.pop(entity, None)
            logger.debug("Cache invalidated: %s", entity)
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_records = sum(len(data) for data in self._cache.values())
        total_size = sum(self._file_sizes.values())
        total_accesses = sum(self._load_count.values())
        cache_hits = total_accesses - len(self._cache)
        
        return {
            "entities_cached": len(self._cache),
            "total_records": total_records,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_accesses": total_accesses,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / total_accesses if total_accesses > 0 else 0,
            "entities": {
                entity: {
                    "records": len(data),
                    "size_kb": self._file_sizes.get(entity, 0) / 1024,
                    "accesses": self._load_count.get(entity, 0),
                }
                for entity, data in self._cache.items()
            },
        }
    
    def __contains__(self, entity: str) -> bool:
        """Check if an entity is in the cache."""
        return entity in self._cache
    
    def __len__(self) -> int:
        """Return number of entities in cache."""
        return len(self._cache)


class AnalysisDataLoader:
    """High-level data loader for analysis operations.
    
    Wraps DataCache with analysis-specific convenience methods.
    """
    
    def __init__(self, data_dir: Path):
        """Initialize the data loader.
        
        Args:
            data_dir: Root data directory (containing json/ subdirectory)
        """
        self.data_dir = data_dir
        self.json_dir = data_dir / "json"
        self.cache = DataCache(self.json_dir)
    
    @property
    def customers(self) -> list[dict[str, Any]]:
        """Get customer records."""
        return self.cache.get("customers")
    
    @property
    def subscriptions(self) -> list[dict[str, Any]]:
        """Get subscription records."""
        return self.cache.get("subscriptions")
    
    @property
    def invoices(self) -> list[dict[str, Any]]:
        """Get invoice records."""
        return self.cache.get("invoices")
    
    @property
    def transactions(self) -> list[dict[str, Any]]:
        """Get transaction records."""
        return self.cache.get("transactions")
    
    @property
    def credit_notes(self) -> list[dict[str, Any]]:
        """Get credit note records."""
        return self.cache.get("credit_notes")
    
    @property
    def events(self) -> list[dict[str, Any]]:
        """Get event records."""
        return self.cache.get("events")
    
    @property
    def coupons(self) -> list[dict[str, Any]]:
        """Get coupon records."""
        return self.cache.get("coupons")
    
    @property
    def item_prices(self) -> list[dict[str, Any]]:
        """Get item price records."""
        return self.cache.get("item_prices")
    
    @property
    def payment_sources(self) -> list[dict[str, Any]]:
        """Get payment source records."""
        return self.cache.get("payment_sources")
    
    def get(self, entity: str) -> list[dict[str, Any]]:
        """Get any entity by name."""
        return self.cache.get(entity)
    
    def preload_for_analysis(self) -> None:
        """Preload all commonly needed data for analysis."""
        self.cache.preload_common()
    
    def get_customer_by_id(self, customer_id: str) -> dict[str, Any] | None:
        """Look up a customer by ID."""
        for customer in self.customers:
            if customer.get("id") == customer_id:
                return customer
        return None
    
    def get_subscriptions_for_customer(self, customer_id: str) -> list[dict[str, Any]]:
        """Get all subscriptions for a customer."""
        return [
            sub for sub in self.subscriptions
            if sub.get("customer_id") == customer_id
        ]
    
    def get_invoices_for_customer(self, customer_id: str) -> list[dict[str, Any]]:
        """Get all invoices for a customer."""
        return [
            inv for inv in self.invoices
            if inv.get("customer_id") == customer_id
        ]
    
    def get_transactions_for_customer(self, customer_id: str) -> list[dict[str, Any]]:
        """Get all transactions for a customer."""
        return [
            txn for txn in self.transactions
            if txn.get("customer_id") == customer_id
        ]
