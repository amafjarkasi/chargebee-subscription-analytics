"""Parallel fetch orchestration for all Chargebee entities."""

import asyncio
import logging
import time
from typing import Any, Callable, Awaitable

from .client import ChargebeeClient, FetchResult, AuthenticationError
from .config import Config
from .entities import INDEPENDENT_ENTITIES, DEPENDENT_ENTITIES, EntityDef
from .storage import StorageManager

logger = logging.getLogger("chargebee_mapper.fetcher")


class FetchOrchestrator:
    """Orchestrates parallel fetching of all Chargebee entity types."""

    def __init__(
        self,
        client: ChargebeeClient,
        config: Config,
        on_entity_start: Callable[[EntityDef], Awaitable[None]] | None = None,
        on_entity_page: Callable[[EntityDef, list[dict], int], Awaitable[None]] | None = None,
        on_entity_done: Callable[[FetchResult], Awaitable[None]] | None = None,
        on_entity_error: Callable[[EntityDef, str], Awaitable[None]] | None = None,
        storage_manager: StorageManager | None = None,
    ):
        self.client = client
        self.config = config
        self._on_entity_start = on_entity_start
        self._on_entity_page = on_entity_page
        self._on_entity_done = on_entity_done
        self._on_entity_error = on_entity_error
        self.storage_manager = storage_manager
        self.results: dict[str, FetchResult] = {}

    async def _fetch_single(self, entity: EntityDef, parent_id: str | None = None) -> FetchResult:
        """Fetch a single entity type with callbacks."""
        logger.debug("Scheduling fetch for %s", entity.name)

        if self._on_entity_start:
            await self._on_entity_start(entity)

        async def on_page_wrapper(ent: EntityDef, records: list[dict], page_num: int) -> None:
            if self.storage_manager:
                await self.storage_manager.stream_to_sqlite(ent.key, records)
            if self._on_entity_page:
                await self._on_entity_page(ent, records, page_num)

        result = await self.client.fetch_entity(
            entity,
            parent_id=parent_id,
            on_page=on_page_wrapper,
            streaming=bool(self.storage_manager),
        )

        if result.errors and self._on_entity_error:
            for error in result.errors:
                await self._on_entity_error(entity, error)

        if self._on_entity_done:
            await self._on_entity_done(result)

        return result

    async def fetch_independent(self) -> dict[str, FetchResult]:
        """Fetch all independent entities in parallel."""
        entity_count = len(INDEPENDENT_ENTITIES)
        logger.info("Phase 1: Fetching %d independent entities in parallel", entity_count)
        start = time.monotonic()

        tasks = [self._fetch_single(entity) for entity in INDEPENDENT_ENTITIES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for entity, result in zip(INDEPENDENT_ENTITIES, results):
            if isinstance(result, AuthenticationError):
                logger.error("Authentication error during independent fetch, aborting")
                raise result
            if isinstance(result, Exception):
                fetch_result = FetchResult(entity)
                fetch_result.errors.append(f"{type(result).__name__}: {result}")
                self.results[entity.key] = fetch_result
                logger.error("Unhandled exception for %s: %s", entity.name, result)
                if self._on_entity_error:
                    await self._on_entity_error(entity, str(result))
                if self._on_entity_done:
                    await self._on_entity_done(fetch_result)
            else:
                self.results[entity.key] = result

        elapsed = time.monotonic() - start
        total_records = sum(r.count for r in self.results.values())
        error_count = sum(1 for r in self.results.values() if not r.success)
        logger.info(
            "Phase 1 complete: %d records from %d entities in %.1fs (%d errors)",
            total_records, entity_count, elapsed, error_count,
        )

        return self.results

    async def fetch_dependent(self) -> dict[str, FetchResult]:
        """Fetch dependent entities using parent IDs from previously fetched data."""
        if not DEPENDENT_ENTITIES:
            logger.info("Phase 2: No dependent entities to fetch")
            return self.results

        logger.info("Phase 2: Fetching %d dependent entity types", len(DEPENDENT_ENTITIES))

        for entity in DEPENDENT_ENTITIES:
            parent_result = self.results.get(entity.parent_entity)

            # Determine parent IDs (either from memory or storage)
            parent_ids: list[str] = []
            if parent_result and parent_result.records:
                parent_ids = [str(r["id"]) for r in parent_result.records if r.get("id")]
            elif parent_result and parent_result.count > 0 and self.storage_manager:
                logger.info("Fetching parent IDs from storage for %s", entity.parent_entity)
                parent_ids = self.storage_manager.get_all_ids(entity.parent_entity)

            if not parent_ids:
                logger.warning(
                    "%s: Skipping, no parent data available from '%s'",
                    entity.name, entity.parent_entity,
                )
                result = FetchResult(entity, streaming=bool(self.storage_manager))
                if self._on_entity_start:
                    await self._on_entity_start(entity)
                if self._on_entity_done:
                    await self._on_entity_done(result)
                self.results[entity.key] = result
                continue

            logger.info("%s: Fetching for %d parent IDs from '%s'", entity.name, len(parent_ids), entity.parent_entity)

            if self._on_entity_start:
                await self._on_entity_start(entity)

            combined_result = FetchResult(entity, streaming=bool(self.storage_manager))
            start = time.monotonic()

            # Limit concurrency to avoid creating too many tasks
            semaphore = asyncio.Semaphore(20)

            async def on_page_wrapper(ent: EntityDef, records: list[dict], page_num: int) -> None:
                if self.storage_manager:
                    await self.storage_manager.stream_to_sqlite(ent.key, records)
                if self._on_entity_page:
                    await self._on_entity_page(ent, records, page_num)

            async def fetch_for_parent(pid: str) -> FetchResult:
                async with semaphore:
                    return await self.client.fetch_entity(
                        entity,
                        parent_id=pid,
                        on_page=on_page_wrapper,
                        streaming=bool(self.storage_manager)
                    )

            sub_results = await asyncio.gather(
                *[fetch_for_parent(pid) for pid in parent_ids],
                return_exceptions=True,
            )

            for sub in sub_results:
                if isinstance(sub, Exception):
                    combined_result.errors.append(f"{type(sub).__name__}: {sub}")
                    logger.error("%s: Sub-fetch failed: %s", entity.name, sub)
                else:
                    if combined_result.streaming:
                        combined_result._streaming_count += sub.count
                    else:
                        combined_result.records.extend(sub.records)

                    combined_result.pages_fetched += sub.pages_fetched
                    combined_result.api_calls += sub.api_calls
                    combined_result.errors.extend(sub.errors)

            combined_result.elapsed = time.monotonic() - start

            logger.info(
                "%s: dependent fetch complete, %d records from %d parents (%.1fs)",
                entity.name, combined_result.count, len(parent_ids), combined_result.elapsed,
            )

            if combined_result.errors and self._on_entity_error:
                for error in combined_result.errors:
                    await self._on_entity_error(entity, error)
            if self._on_entity_done:
                await self._on_entity_done(combined_result)

            self.results[entity.key] = combined_result

        logger.info("Phase 2 complete")
        return self.results

    async def fetch_all(self) -> dict[str, FetchResult]:
        """Fetch all entities: independent first, then dependent."""
        await self.fetch_independent()
        await self.fetch_dependent()
        return self.results
