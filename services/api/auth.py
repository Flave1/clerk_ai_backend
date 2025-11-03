"""
Authentication utilities for JWT tokens and password hashing.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# HTTP Bearer token scheme
security = HTTPBearer()


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.
    
    Bcrypt has a 72-byte limit, so we truncate longer passwords.
    Uses bcrypt directly to avoid passlib initialization issues.
    """
    if not password:
        raise ValueError("Password cannot be empty")
    
    # Convert to bytes for bcrypt
    password_bytes = password.encode('utf-8')
    
    # CRITICAL: Bcrypt limit is exactly 72 bytes - truncate if needed
    if len(password_bytes) > 72:
        logger.warning(f"Password exceeds 72 bytes ({len(password_bytes)} bytes), truncating to 72 bytes")
        # Truncate to exactly 72 bytes
        password_bytes = password_bytes[:72]
    
    # Hash using bcrypt directly (returns bytes, convert to string)
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash.
    
    Bcrypt has a 72-byte limit, so we truncate longer passwords before verification.
    Uses bcrypt directly to avoid passlib initialization issues.
    """
    if not plain_password:
        return False
    
    # Convert to bytes and ensure it's <= 72 bytes
    password_bytes = plain_password.encode('utf-8')
    
    # Bcrypt limit is 72 bytes - truncate if needed
    if len(password_bytes) > 72:
        logger.warning(f"Password for verification exceeds 72 bytes ({len(password_bytes)} bytes), truncating")
        password_bytes = password_bytes[:72]
    
    # Convert hashed_password to bytes if it's a string
    if isinstance(hashed_password, str):
        hashed_bytes = hashed_password.encode('utf-8')
    else:
        hashed_bytes = hashed_password
    
    # Verify using bcrypt directly
    try:
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary containing the data to encode (e.g., {"sub": user_id, "email": email})
        expires_delta: Optional expiration time delta. Defaults to settings.access_token_expire_minutes
        
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm
    )
    
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode and verify a JWT access token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Dependency to get the current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        User payload from token (contains user_id and email)
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    token = credentials.credentials
    payload = decode_access_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {
        "user_id": user_id,
        "email": payload.get("email"),
    }

