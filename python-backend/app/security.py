from __future__ import annotations

import os

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=False)


async def require_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    expected_key = os.getenv("WORKSPACE_API_KEY", "")
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Workspace API key not configured",
        )
    if not api_key or api_key != expected_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return api_key


async def get_user_id(api_key: str = Depends(require_api_key)) -> str:
    # For now, API key maps directly to a single user/tenant
    return os.getenv("WORKSPACE_USER_ID", "default")
