"""Async Chargebee API client with rate limiting, retries, and logging."""

import asyncio
import logging
import random
import time
from collections import deque
from typing import Any

from chargebee import Chargebee
from chargebee.retry_config import RetryConfig

from .config import Config
from .entities import EntityDef

logger = logging.getLogger("chargebee_mapper.client")


class AuthenticationError(Exception):
    """Raised when API key is invalid."""


class FetchResult:
    """Result of fetching a single entity type."""

    def __init__(self, entity: EntityDef):
        self.entity = entity
        self.records: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.pages_fetched: int = 0
        self.api_calls: int = 0
        self.elapsed: float = 0.0

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class RateLimiter:
    """Token-bucket rate limiter with concurrency semaphore.

    Enforces a per-minute request quota (RPM) using a sliding window and limits
    the number of simultaneous in-flight requests via an asyncio.Semaphore.

    Chargebee API rate limits (requests per minute):
      Starter / Test:      150
      Performance:        1,000
      Enterprise:         3,500
      Enterprise custom:  unlimited
    GET concurrency limit: 50 simultaneous requests.
    """

    def __init__(self, rpm: int, max_concurrent: int):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rpm = rpm
        # Use maxlen to prevent unbounded memory growth (2x RPM as safety margin)
        self._timestamps: deque[float] = deque(maxlen=max(rpm * 2, 1000))
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until both a concurrency slot and an RPM slot are available.
        
        Order of operations:
        1. Acquire semaphore (concurrency slot) - ensures we can actually make request
        2. Check/enforce RPM quota - timestamps are accurate since we hold the slot
        """
        # First acquire concurrency slot
        await self._semaphore.acquire()
        
        # Then enforce RPM quota (sliding 60-second window)
        try:
            async with self._lock:
                now = time.monotonic()
                # Purge timestamps older than 60 seconds
                while self._timestamps and now - self._timestamps[0] > 60.0:
                    self._timestamps.popleft()
                # If at capacity, wait for the oldest timestamp to expire
                if len(self._timestamps) >= self._rpm:
                    wait = 60.0 - (now - self._timestamps[0]) + 0.05
                    if wait > 0:
                        logger.debug(
                            "Rate limiter: at %d/%d RPM capacity, waiting %.1fs",
                            len(self._timestamps), self._rpm, wait
                        )
                        # Release lock while sleeping to allow other waiters to check
                        # but keep semaphore to maintain our concurrency slot
                        self._lock.release()
                        try:
                            await asyncio.sleep(wait)
                        finally:
                            await self._lock.acquire()
                        # Re-purge after sleep
                        now = time.monotonic()
                        while self._timestamps and now - self._timestamps[0] > 60.0:
                            self._timestamps.popleft()
                # Record timestamp for this request
                self._timestamps.append(time.monotonic())
        except Exception:
            # If RPM check fails, release the semaphore we acquired
            self._semaphore.release()
            raise

    def release(self) -> None:
        """Release the concurrency slot."""
        self._semaphore.release()


class ChargebeeClient:
    """Async Chargebee client with rate limiting, retries, and structured logging."""

    def __init__(self, config: Config):
        self.config = config
        self._rate_limiter = RateLimiter(
            rpm=config.rate_limit_rpm,
            max_concurrent=config.max_concurrency,
        )
        self._api_calls = 0
        self._lock = asyncio.Lock()

        # SDK-level retry as a safety net (our own backoff handles 429 explicitly)
        retry_config = RetryConfig(
            enabled=True,
            max_retries=config.max_retries,
            delay_ms=1000,
            retry_on=[500, 502, 503, 504],
        )

        self._client = Chargebee(
            api_key=config.api_key,
            site=config.site,
            use_async_client=True,
        )
        self._client.update_retry_config(retry_config)

        logger.info(
            "Chargebee client initialized: site=%s, rpm=%d, concurrency=%d, max_retries=%d",
            config.site, config.rate_limit_rpm, config.max_concurrency, config.max_retries,
        )

    async def _increment_calls(self) -> None:
        async with self._lock:
            self._api_calls += 1

    @property
    def total_api_calls(self) -> int:
        return self._api_calls

    def _get_resource(self, sdk_resource: str) -> Any:
        """Get an SDK resource class by name (e.g., 'Customer' -> cb_client.Customer)."""
        return getattr(self._client, sdk_resource)

    def _extract_record(self, entry: Any, response_key: str) -> dict[str, Any]:
        """Extract a record dict from a list response entry.

        The SDK v3 response entries have entity attributes (e.g., entry.customer)
        whose objects carry a ``raw_data`` dict with the full API JSON.
        """
        record_obj = getattr(entry, response_key, None)
        if record_obj is None:
            return {}
        # Prefer raw_data (complete JSON dict from the API)
        if hasattr(record_obj, "raw_data") and isinstance(record_obj.raw_data, dict):
            return record_obj.raw_data
        # Fallback: build dict from public attributes
        if hasattr(record_obj, "__dict__"):
            return {
                k: v for k, v in record_obj.__dict__.items()
                if not k.startswith("_")
            }
        return {}

    def _compute_backoff(self, attempt: int) -> float:
        """Compute exponential backoff delay with jitter per Chargebee docs.

        Formula: min(base * 2^attempt + jitter, max_delay)
        """
        base = 1.0
        max_delay = 60.0
        jitter = random.uniform(0, 0.5)
        delay = min(base * (2 ** attempt) + jitter, max_delay)
        return delay

    async def _call_list(
        self,
        entity: EntityDef,
        offset: str | None = None,
        parent_id: str | None = None,
    ) -> Any:
        """Execute a single SDK list() call, handling rate limits with backoff.

        Retries up to ``config.max_retries`` times on 429 responses, respecting
        the ``Retry-After`` header when present.
        """
        resource = self._get_resource(entity.sdk_resource)

        for attempt in range(self.config.max_retries + 1):
            await self._rate_limiter.acquire()
            try:
                await self._increment_calls()

                if entity.no_list_params:
                    logger.debug(
                        "%s: API call (no params), attempt=%d",
                        entity.name, attempt,
                    )
                    return await resource.list()
                else:
                    params_class = resource.ListParams
                    kwargs: dict[str, Any] = {"limit": self.config.page_size}
                    if offset:
                        kwargs["offset"] = offset
                    params = params_class(**kwargs)

                    logger.debug(
                        "%s: API call, offset=%s, limit=%d, attempt=%d",
                        entity.name, offset, self.config.page_size, attempt,
                    )

                    if entity.requires_parent and parent_id:
                        return await resource.list(parent_id, params)
                    else:
                        return await resource.list(params)

            except Exception as e:
                error_msg = str(e)

                # Authentication errors are non-retryable
                if "authentication" in error_msg.lower() or "api_authentication_failed" in error_msg.lower():
                    logger.error("%s: Authentication failed for site '%s'", entity.name, self.config.site)
                    raise AuthenticationError(f"Authentication failed for site '{self.config.site}'") from e

                # Rate limit (429) - retry with backoff
                if "api_request_limit_exceeded" in error_msg.lower() or "429" in error_msg:
                    # Check for Retry-After header in the exception
                    retry_after = None
                    if hasattr(e, "http_body") and hasattr(e, "http_status_code"):
                        # Some SDK exceptions carry response metadata
                        retry_after_hdr = getattr(e, "retry_after", None)
                        if retry_after_hdr:
                            try:
                                retry_after = float(retry_after_hdr)
                            except (ValueError, TypeError):
                                pass

                    if retry_after and retry_after > 0:
                        delay = retry_after
                        logger.warning(
                            "%s: Rate limited (429), Retry-After=%.1fs, attempt %d/%d",
                            entity.name, delay, attempt + 1, self.config.max_retries,
                        )
                    else:
                        delay = self._compute_backoff(attempt)
                        logger.warning(
                            "%s: Rate limited (429), backoff=%.1fs, attempt %d/%d",
                            entity.name, delay, attempt + 1, self.config.max_retries,
                        )

                    if attempt < self.config.max_retries:
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(
                            "%s: Rate limit retries exhausted after %d attempts",
                            entity.name, self.config.max_retries,
                        )
                        raise

                # Other errors - don't retry here (SDK RetryConfig handles 5xx)
                raise
            finally:
                self._rate_limiter.release()

        # Should not reach here, but just in case
        raise RuntimeError(f"{entity.name}: Unexpected exit from retry loop")

    async def fetch_page(
        self,
        entity: EntityDef,
        offset: str | None = None,
        parent_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch a single page of records for an entity.

        Returns (records, next_offset). next_offset is None when no more pages.
        """
        response = await self._call_list(entity, offset=offset, parent_id=parent_id)

        records = []
        # SDK v3: entries are in response.list, not directly iterable
        entries = getattr(response, "list", None) or []
        for entry in entries:
            record = self._extract_record(entry, entity.response_key)
            if record:
                records.append(record)

        next_offset = getattr(response, "next_offset", None)

        logger.debug(
            "%s: page fetched, %d records, next_offset=%s",
            entity.name, len(records), next_offset,
        )
        return records, next_offset

    async def fetch_entity(
        self,
        entity: EntityDef,
        parent_id: str | None = None,
        on_page: Any = None,
    ) -> FetchResult:
        """Fetch all records for an entity type, handling pagination.

        on_page: optional async callback(entity, page_records, page_num) called after each page.
        """
        result = FetchResult(entity)
        start = time.monotonic()
        offset = None

        logger.info("%s: starting fetch%s", entity.name, f" (parent={parent_id})" if parent_id else "")

        try:
            while True:
                page_records, next_offset = await self.fetch_page(
                    entity, offset=offset, parent_id=parent_id
                )
                result.pages_fetched += 1
                result.api_calls += 1
                result.records.extend(page_records)

                if on_page:
                    await on_page(entity, page_records, result.pages_fetched)

                if not next_offset or not page_records:
                    break
                offset = next_offset

        except AuthenticationError:
            raise
        except Exception as e:
            error_msg = f"{entity.name}: {type(e).__name__}: {e}"
            result.errors.append(error_msg)
            logger.error("%s: fetch failed: %s", entity.name, e)

        result.elapsed = time.monotonic() - start

        if result.errors:
            logger.error("%s: completed with errors, %d records in %d pages (%.1fs)", entity.name, result.count, result.pages_fetched, result.elapsed)
        elif result.count == 0:
            logger.warning("%s: 0 records returned (%.1fs)", entity.name, result.elapsed)
        else:
            logger.info("%s: fetch complete, %d records in %d pages (%.1fs)", entity.name, result.count, result.pages_fetched, result.elapsed)

        return result
