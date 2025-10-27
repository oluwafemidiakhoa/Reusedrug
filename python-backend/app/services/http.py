from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import httpx
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential

from app.cache import TTLCache
_CLIENT_CACHE = TTLCache(ttl_seconds=300, maxsize=4)
_CLIENT_KEY = "httpx-client"


def _build_client() -> httpx.AsyncClient:
    timeout = httpx.Timeout(
        connect=float(os.getenv("HTTP_CLIENT_CONNECT_TIMEOUT", "5")),
        read=float(os.getenv("HTTP_CLIENT_READ_TIMEOUT", "10")),
        write=float(os.getenv("HTTP_CLIENT_WRITE_TIMEOUT", "10")),
        pool=float(os.getenv("HTTP_CLIENT_POOL_TIMEOUT", "5")),
    )
    limits = httpx.Limits(
        max_connections=int(os.getenv("HTTP_CLIENT_MAX_CONNECTIONS", "20")),
        max_keepalive_connections=int(os.getenv("HTTP_CLIENT_KEEPALIVE", "10")),
    )
    return httpx.AsyncClient(timeout=timeout, limits=limits)


async def get_client() -> httpx.AsyncClient:
    return await _CLIENT_CACHE.get_or_set(_CLIENT_KEY, _build_client)


@asynccontextmanager
async def get_retrying_client() -> AsyncIterator[httpx.AsyncClient]:
    client = await _CLIENT_CACHE.get_or_set(_CLIENT_KEY, _build_client)
    try:
        yield client
    finally:
        # client kept open for reuse
        await asyncio.sleep(0)


async def request_with_retry(
    method: str,
    url: str,
    *,
    params: Optional[dict] = None,
    json: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> httpx.Response:
    client = await get_client()
    attempt_limit = int(os.getenv("HTTP_RETRY_ATTEMPTS", "3"))
    base = float(os.getenv("HTTP_RETRY_BASE", "0.5"))
    max_wait = float(os.getenv("HTTP_RETRY_MAX", "4"))

    async for attempt in AsyncRetrying(
        wait=wait_exponential(multiplier=base, max=max_wait),
        stop=stop_after_attempt(attempt_limit),
        retry_error_cls=RetryError,
    ):
        with attempt:
            response = await client.request(
                method, url, params=params, json=json, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            return response

    raise RuntimeError("request_with_retry exhausted retries")
