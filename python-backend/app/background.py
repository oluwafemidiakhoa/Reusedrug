from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.services import ranking

logger = logging.getLogger(__name__)


class Prefetcher:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._seen: set[str] = set()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._worker(), name="prefetch-worker")

    async def stop(self) -> None:
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        self._seen.clear()

    def enqueue(self, query: str) -> None:
        key = query.strip().lower()
        if not key or key in self._seen:
            return
        self._seen.add(key)
        self._queue.put_nowait(key)

    async def _worker(self) -> None:
        logger.info("Prefetcher worker started")
        while True:
            query = await self._queue.get()
            try:
                await ranking.compute_rank(query, force_refresh=True, background=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Background prefetch failed for %s: %s", query, exc)
            finally:
                self._seen.discard(query)
                self._queue.task_done()


prefetcher = Prefetcher()

