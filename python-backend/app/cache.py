from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any, Callable, Hashable, Optional


class TTLCache:
    """Simple async-aware TTL cache with optional max size."""

    def __init__(self, ttl_seconds: float, maxsize: Optional[int] = None) -> None:
        self.ttl = ttl_seconds
        self.maxsize = maxsize
        self._store: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(
        self, key: Hashable, default: Optional[Any] = None
    ) -> Any:
        async with self._lock:
            now = time.monotonic()
            if key in self._store:
                expires_at, value = self._store[key]
                if expires_at > now:
                    self._store.move_to_end(key, last=True)
                    return value
                del self._store[key]
            return default

    async def set(self, key: Hashable, value: Any) -> None:
        async with self._lock:
            now = time.monotonic()
            self._store[key] = (now + self.ttl, value)
            self._evict_if_needed()

    async def get_or_set(
        self, key: Hashable, factory: Callable[[], Any]
    ) -> Any:
        """Return cached value or create a new one from factory."""
        async with self._lock:
            now = time.monotonic()
            if key in self._store:
                expires_at, value = self._store[key]
                if expires_at > now:
                    self._store.move_to_end(key, last=True)
                    return value
                del self._store[key]

            value = factory()
            self._store[key] = (now + self.ttl, value)
            self._evict_if_needed()
            return value

    def _evict_if_needed(self) -> None:
        if self.maxsize is None:
            return
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
