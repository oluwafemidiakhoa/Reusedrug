from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

from pymongo import MongoClient, ReturnDocument, uri_parser
from pymongo.collection import Collection
from pymongo.database import Database

import mongomock

_client: MongoClient | mongomock.MongoClient | None = None
_database: Database | None = None
_client_lock = asyncio.Lock()


def _build_client(uri: str) -> tuple[MongoClient | mongomock.MongoClient, Database]:
    db_name = os.getenv("MONGODB_DB")
    if uri.startswith("mongomock://"):
        client = mongomock.MongoClient()
        database = client[db_name or "drug_repurposing"]
        _ensure_indexes(database)
        return client, database

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    # Verify connectivity eagerly so failures surface during startup
    client.admin.command("ping")

    if not db_name:
        try:
            parsed = uri_parser.parse_uri(uri)
            db_name = parsed.get("database") or "drug_repurposing"
        except Exception:  # pragma: no cover - fallback safety
            db_name = "drug_repurposing"

    database = client[db_name]
    _ensure_indexes(database)
    return client, database


def _ensure_indexes(db: Database) -> None:
    db["rank_cache"].create_index("query_key", unique=True)
    db["rank_cache"].create_index("updated_at")
    db["saved_queries"].create_index([("user_id", 1), ("created_at", -1)])


async def _get_database() -> Database:
    global _client, _database
    if _database is not None:
        return _database

    async with _client_lock:
        if _database is not None:
            return _database
        uri = os.getenv("MONGODB_URI", "mongomock://localhost")

        def init_connection() -> tuple[MongoClient | mongomock.MongoClient, Database]:
            return _build_client(uri)

        client, database = await asyncio.to_thread(init_connection)
        _client = client
        _database = database
        return _database


async def _collection(name: str) -> Collection:
    db = await _get_database()
    return db[name]


async def init_db() -> None:
    await _get_database()


async def get_cached_rank(query: str, ttl_seconds: float) -> Optional[dict[str, Any]]:
    if ttl_seconds <= 0:
        return None
    collection = await _collection("rank_cache")

    def fetch() -> Optional[dict[str, Any]]:
        return collection.find_one({"query_key": query.lower()})

    record = await asyncio.to_thread(fetch)
    if not record:
        return None
    updated_at = record.get("updated_at", 0.0)
    if (time.time() - updated_at) > ttl_seconds:
        return None
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    return response


async def store_rank(
    query: str,
    normalized: Optional[str],
    response: dict[str, Any],
) -> None:
    collection = await _collection("rank_cache")
    payload = {
        "query_key": query.lower(),
        "original_query": query,
        "normalized": normalized,
        "response": response,
        "updated_at": time.time(),
    }

    def upsert() -> None:
        collection.update_one(
            {"query_key": payload["query_key"]},
            {"$set": payload},
            upsert=True,
        )

    await asyncio.to_thread(upsert)


async def ensure_user(user_id: str) -> None:
    collection = await _collection("users")
    now = time.time()

    def upsert() -> None:
        collection.update_one(
            {"_id": user_id},
            {"$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    await asyncio.to_thread(upsert)


def _next_saved_query_id(collection: Collection) -> int:
    counters = collection.database["counters"]
    document = counters.find_one_and_update(
        {"_id": "saved_queries"},
        {"$inc": {"value": 1}, "$setOnInsert": {"value": 0}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return int(document["value"])


async def save_query(
    user_id: str,
    disease: str,
    response: dict[str, Any],
    note: Optional[str] = None,
) -> None:
    await ensure_user(user_id)
    collection = await _collection("saved_queries")
    created_at = time.time()

    def insert() -> None:
        record_id = _next_saved_query_id(collection)
        collection.insert_one(
            {
                "_id": record_id,
                "id": record_id,
                "user_id": user_id,
                "disease": disease,
                "response": response,
                "created_at": created_at,
                "note": note,
            }
        )

    await asyncio.to_thread(insert)


async def list_saved_queries(user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    collection = await _collection("saved_queries")

    def fetch() -> list[dict[str, Any]]:
        cursor = (
            collection.find({"user_id": user_id})
            .sort("created_at", -1)
            .limit(limit)
        )
        records: list[dict[str, Any]] = []
        for doc in cursor:
            records.append(
                {
                    "id": int(doc.get("id", doc.get("_id"))),
                    "disease": doc.get("disease"),
                    "created_at": float(doc.get("created_at", 0.0)),
                    "response": doc.get("response") or {},
                    "note": doc.get("note"),
                }
            )
        return records

    return await asyncio.to_thread(fetch)


async def update_query_note(user_id: str, query_id: int, note: Optional[str]) -> Optional[dict[str, Any]]:
    collection = await _collection("saved_queries")

    def update() -> Optional[dict[str, Any]]:
        document = collection.find_one_and_update(
            {"id": query_id, "user_id": user_id},
            {"$set": {"note": note}},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            return None
        return document

    doc = await asyncio.to_thread(update)
    if not doc:
        return None
    return {
        "id": int(doc.get("id", doc.get("_id"))),
        "disease": doc.get("disease"),
        "created_at": float(doc.get("created_at", 0.0)),
        "response": doc.get("response") or {},
        "note": doc.get("note"),
    }


async def delete_saved_query(user_id: str, query_id: int) -> bool:
    collection = await _collection("saved_queries")

    def delete() -> bool:
        result = collection.delete_one({"id": query_id, "user_id": user_id})
        return result.deleted_count > 0

    return await asyncio.to_thread(delete)


async def clear_database() -> None:
    """Utility for tests to reset collections."""
    db = await _get_database()

    def wipe() -> None:
        db["rank_cache"].delete_many({})
        db["users"].delete_many({})
        db["saved_queries"].delete_many({})
        db["counters"].delete_many({})

    await asyncio.to_thread(wipe)

