from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.db import delete_saved_query, list_saved_queries, save_query, update_query_note
from app.models import RankResponse, SaveQueryRequest, SavedQuery, UpdateQueryRequest
from app.security import get_user_id

router = APIRouter(prefix="/v1/workspaces", tags=["workspace"])


@router.get("/queries", response_model=list[SavedQuery])
async def list_queries(user_id: str = Depends(get_user_id)) -> list[SavedQuery]:
    records = await list_saved_queries(user_id)
    return [
        SavedQuery(
            id=record["id"],
            disease=record["disease"],
            created_at=record["created_at"],
            response=RankResponse(**record["response"]),
            note=record.get("note"),
        )
        for record in records
    ]


@router.post(
    "/queries",
    status_code=status.HTTP_201_CREATED,
    response_model=SavedQuery,
)
async def create_query(payload: SaveQueryRequest, user_id: str = Depends(get_user_id)) -> SavedQuery:
    await save_query(
        user_id,
        payload.disease,
        payload.response.model_dump(),
        note=payload.note,
    )
    records = await list_saved_queries(user_id, limit=1)
    record = records[0]
    return SavedQuery(
        id=record["id"],
        disease=record["disease"],
        created_at=record["created_at"],
        response=RankResponse(**record["response"]),
        note=record.get("note"),
    )

@router.patch(
    "/queries/{query_id}",
    response_model=SavedQuery,
)
async def update_query(
    query_id: int,
    payload: UpdateQueryRequest,
    user_id: str = Depends(get_user_id),
) -> SavedQuery:
    record = await update_query_note(user_id, query_id, payload.note)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved query not found")
    return SavedQuery(
        id=record["id"],
        disease=record["disease"],
        created_at=record["created_at"],
        response=RankResponse(**record["response"]),
        note=record.get("note"),
    )


@router.delete(
    "/queries/{query_id}",
    response_class=Response,
)
async def delete_query(query_id: int, user_id: str = Depends(get_user_id)) -> None:
    removed = await delete_saved_query(user_id, query_id)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved query not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


