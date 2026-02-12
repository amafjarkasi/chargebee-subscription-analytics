"""Unit tests for chargebee_mapper.client module."""

import asyncio
import time

import pytest

from chargebee_mapper.client import FetchResult, RateLimiter
from chargebee_mapper.entities import EntityDef


class TestFetchResult:
    """Tests for FetchResult class."""

    def test_init(self, sample_entity):
        """Test FetchResult initialization."""
        result = FetchResult(sample_entity)
        
        assert result.entity == sample_entity
        assert result.records == []
        assert result.errors == []
        assert result.pages_fetched == 0
        assert result.api_calls == 0
        assert result.elapsed == 0.0

    def test_count_property(self, sample_entity):
        """Test count property returns correct record count."""
        result = FetchResult(sample_entity)
        assert result.count == 0
        
        result.records = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        assert result.count == 3

    def test_success_property_no_errors(self, sample_entity):
        """Test success is True when no errors."""
        result = FetchResult(sample_entity)
        assert result.success is True

    def test_success_property_with_errors(self, sample_entity):
        """Test success is False when errors exist."""
        result = FetchResult(sample_entity)
        result.errors.append("Something went wrong")
        assert result.success is False


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(rpm=100, max_concurrent=10)
        
        assert limiter._rpm == 100
        assert limiter._semaphore._value == 10
        assert len(limiter._timestamps) == 0

    @pytest.mark.asyncio
    async def test_acquire_release(self):
        """Test basic acquire and release."""
        limiter = RateLimiter(rpm=100, max_concurrent=5)
        
        # Acquire should succeed
        await limiter.acquire()
        assert limiter._semaphore._value == 4
        assert len(limiter._timestamps) == 1
        
        # Release should restore semaphore
        limiter.release()
        assert limiter._semaphore._value == 5

    @pytest.mark.asyncio
    async def test_concurrency_limiting(self):
        """Test that concurrency is properly limited."""
        limiter = RateLimiter(rpm=1000, max_concurrent=2)
        
        acquired_count = []
        
        async def acquire_and_hold():
            await limiter.acquire()
            acquired_count.append(1)
            await asyncio.sleep(0.2)
            limiter.release()
        
        # Start 3 tasks with only 2 concurrent slots
        task1 = asyncio.create_task(acquire_and_hold())
        task2 = asyncio.create_task(acquire_and_hold())
        task3 = asyncio.create_task(acquire_and_hold())
        
        # Give first two tasks time to acquire
        await asyncio.sleep(0.05)
        assert len(acquired_count) == 2  # Only 2 should have acquired
        
        # Wait for all to complete
        await asyncio.gather(task1, task2, task3)
        assert len(acquired_count) == 3

    @pytest.mark.asyncio
    async def test_timestamps_recorded(self):
        """Test that timestamps are recorded on acquire."""
        limiter = RateLimiter(rpm=100, max_concurrent=10)
        
        start_time = time.monotonic()
        await limiter.acquire()
        
        assert len(limiter._timestamps) == 1
        assert limiter._timestamps[0] >= start_time
        
        limiter.release()

    @pytest.mark.asyncio
    async def test_deque_maxlen_prevents_unbounded_growth(self):
        """Test that timestamps deque has maxlen to prevent memory leaks."""
        rpm = 10
        limiter = RateLimiter(rpm=rpm, max_concurrent=100)
        
        # Maxlen should be set to 2x RPM or 1000, whichever is larger
        expected_maxlen = max(rpm * 2, 1000)
        assert limiter._timestamps.maxlen == expected_maxlen

    @pytest.mark.asyncio
    async def test_multiple_acquires(self):
        """Test multiple sequential acquires work correctly."""
        limiter = RateLimiter(rpm=100, max_concurrent=5)
        
        for i in range(5):
            await limiter.acquire()
            assert len(limiter._timestamps) == i + 1
        
        # All 5 slots used
        assert limiter._semaphore._value == 0
        
        # Release all
        for _ in range(5):
            limiter.release()
        
        assert limiter._semaphore._value == 5

    @pytest.mark.asyncio
    async def test_old_timestamps_purged(self):
        """Test that timestamps older than 60 seconds are purged."""
        limiter = RateLimiter(rpm=100, max_concurrent=10)
        
        # Manually add old timestamps (older than 60 seconds)
        old_time = time.monotonic() - 65  # 65 seconds ago
        limiter._timestamps.append(old_time)
        limiter._timestamps.append(old_time + 1)
        
        assert len(limiter._timestamps) == 2
        
        # Acquire should purge old timestamps
        await limiter.acquire()
        
        # Old timestamps should be purged, only new one remains
        assert len(limiter._timestamps) == 1
        assert limiter._timestamps[0] > old_time + 60
        
        limiter.release()
