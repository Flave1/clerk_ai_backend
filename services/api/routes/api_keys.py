"""
API Keys API routes.
"""
import logging
import secrets
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from shared.schemas import ApiKey, ApiKeyStatus

from ..auth import get_current_user, hash_password
from ..dao import MongoDBDAO, get_dao

logger = logging.getLogger(__name__)

router = APIRouter()


class ApiKeyCreateRequest(BaseModel):
    """API Key creation request."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Name for the API key")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Number of days until expiration (1-365)")
    scopes: List[str] = Field(default_factory=list, description="List of permission scopes")


class ApiKeyResponse(BaseModel):
    """API Key response model."""
    
    id: str
    user_id: str
    name: str
    key_prefix: str
    status: str
    last_used_at: Optional[str]
    expires_at: Optional[str]
    scopes: List[str]
    created_at: str
    updated_at: str
    plain_key: Optional[str] = None  # Only present on creation


class ApiKeyUpdateRequest(BaseModel):
    """API Key update request."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    status: Optional[str] = None  # "active" or "revoked"


def generate_api_key() -> str:
    """Generate a secure API key."""
    # Format: sk_live_<random_secret>
    random_part = secrets.token_urlsafe(32)  # 43-44 characters
    return f"sk_live_{random_part}"


@router.post("/", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: ApiKeyCreateRequest,
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """Create a new API key for the authenticated user."""
    try:
        user_id = UUID(current_user["user_id"])
        
        # Generate API key
        plain_key = generate_api_key()
        
        # Hash the key (using same method as passwords)
        key_hash = hash_password(plain_key)
        
        # Extract prefix (first 12 characters)
        key_prefix = plain_key[:12]
        
        # Calculate expiration
        expires_at = None
        if request.expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)
        
        # Create API key object
        api_key = ApiKey(
            id=uuid4(),
            user_id=user_id,
            name=request.name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            status=ApiKeyStatus.ACTIVE,
            expires_at=expires_at,
            scopes=request.scopes or [],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        
        # Save to database
        await dao.create_api_key(api_key)
        
        # Return response with plain key (only time it's shown)
        return ApiKeyResponse(
            id=str(api_key.id),
            user_id=str(api_key.user_id),
            name=api_key.name,
            key_prefix=api_key.key_prefix,
            status=api_key.status.value,
            last_used_at=api_key.last_used_at.isoformat() if api_key.last_used_at else None,
            expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
            scopes=api_key.scopes,
            created_at=api_key.created_at.isoformat(),
            updated_at=api_key.updated_at.isoformat(),
            plain_key=plain_key,  # Include plain key only on creation
        )
    
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[ApiKeyResponse])
async def list_api_keys(
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """Get all API keys for the authenticated user."""
    try:
        user_id = current_user["user_id"]
        api_keys = await dao.get_api_keys_by_user(user_id)
        
        return [
            ApiKeyResponse(
                id=str(key.id),
                user_id=str(key.user_id),
                name=key.name,
                key_prefix=key.key_prefix,
                status=key.status.value,
                last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
                expires_at=key.expires_at.isoformat() if key.expires_at else None,
                scopes=key.scopes,
                created_at=key.created_at.isoformat(),
                updated_at=key.updated_at.isoformat(),
            )
            for key in api_keys
        ]
    
    except Exception as e:
        logger.error(f"Failed to list API keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{api_key_id}", response_model=ApiKeyResponse)
async def get_api_key(
    api_key_id: str,
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """Get a specific API key. Only accessible by the owner."""
    try:
        user_id = current_user["user_id"]
        api_key = await dao.get_api_key_by_id(api_key_id)
        
        if not api_key:
            raise HTTPException(status_code=404, detail="API key not found")
        
        if str(api_key.user_id) != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return ApiKeyResponse(
            id=str(api_key.id),
            user_id=str(api_key.user_id),
            name=api_key.name,
            key_prefix=api_key.key_prefix,
            status=api_key.status.value,
            last_used_at=api_key.last_used_at.isoformat() if api_key.last_used_at else None,
            expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
            scopes=api_key.scopes,
            created_at=api_key.created_at.isoformat(),
            updated_at=api_key.updated_at.isoformat(),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{api_key_id}", response_model=ApiKeyResponse)
async def update_api_key(
    api_key_id: str,
    request: ApiKeyUpdateRequest,
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """Update an API key. Only accessible by the owner."""
    try:
        user_id = current_user["user_id"]
        api_key = await dao.get_api_key_by_id(api_key_id)
        
        if not api_key:
            raise HTTPException(status_code=404, detail="API key not found")
        
        if str(api_key.user_id) != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update fields
        if request.name is not None:
            api_key.name = request.name
        
        if request.status is not None:
            api_key.status = ApiKeyStatus(request.status)
        
        api_key.updated_at = datetime.utcnow()
        
        # Save changes
        updated_key = await dao.update_api_key(api_key)
        
        return ApiKeyResponse(
            id=str(updated_key.id),
            user_id=str(updated_key.user_id),
            name=updated_key.name,
            key_prefix=updated_key.key_prefix,
            status=updated_key.status.value,
            last_used_at=updated_key.last_used_at.isoformat() if updated_key.last_used_at else None,
            expires_at=updated_key.expires_at.isoformat() if updated_key.expires_at else None,
            scopes=updated_key.scopes,
            created_at=updated_key.created_at.isoformat(),
            updated_at=updated_key.updated_at.isoformat(),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    api_key_id: str,
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """Delete (revoke) an API key. Only accessible by the owner."""
    try:
        user_id = current_user["user_id"]
        deleted = await dao.delete_api_key(api_key_id, user_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="API key not found or access denied")
        
        return None
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

