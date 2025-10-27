import os

import pytest_asyncio
from httpx import AsyncClient

os.environ.setdefault("MONGODB_URI", "mongomock://localhost")
os.environ.setdefault("MONGODB_DB", "test_repurposing")
os.environ.setdefault("RESULT_CACHE_TTL_SECONDS", "0")

from app.main import app  # noqa: E402
from app.db import clear_database, init_db  # noqa: E402


@pytest_asyncio.fixture
async def test_client():
    await clear_database()
    await init_db()
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
