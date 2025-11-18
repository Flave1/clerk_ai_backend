"""
Authentication API routes.
"""
import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, field_validator

from shared.schemas import User

from ..auth import create_access_token, get_current_user, hash_password, verify_password
from ..dao import MongoDBDAO, get_dao

logger = logging.getLogger(__name__)

router = APIRouter()


class RegisterRequest(BaseModel):
    """User registration request."""
    
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=72, description="Password must be 6-72 characters (alphanumeric + special characters)")
    name: str = Field(..., min_length=1)
    phone: str | None = None
    timezone: str = "UTC"
    
    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength - alphanumeric + special characters."""
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters long")
        
        # Check byte length (bcrypt limit is 72 bytes, not characters)
        if len(v.encode('utf-8')) > 72:
            raise ValueError("Password must be no longer than 72 bytes (approximately 72 characters)")
        
        # Check if password contains at least one alphanumeric character
        # Special characters are optional
        has_alnum = any(c.isalnum() for c in v)
        if not has_alnum:
            raise ValueError("Password must contain at least one letter or number")
        
        return v


class SignInRequest(BaseModel):
    """User sign-in request."""
    
    email: EmailStr
    password: str


class GoogleOAuthRequest(BaseModel):
    """Google OAuth login request."""
    
    email: EmailStr
    first_name: str = Field(..., min_length=1, description="User's first name from Google")
    last_name: str = Field(..., min_length=1, description="User's last name from Google")
    phone: str | None = None
    timezone: str = "UTC"


class AuthResponse(BaseModel):
    """Authentication response."""
    
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    name: str


class UserResponse(BaseModel):
    """User response (without password hash)."""
    
    id: str
    email: str
    name: str
    phone: str | None
    timezone: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    dao: MongoDBDAO = Depends(get_dao),
):
    """
    Register a new user.
    
    Creates a new user account with hashed password and returns an access token.
    """
    try:
        # Check if user already exists
        existing_user = await dao.get_user_by_email(request.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists",
            )
        
        # Hash password - strip whitespace and ensure it's clean
        clean_password = request.password.strip()
        if not clean_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password cannot be empty or whitespace only",
            )
        password_hash = hash_password(clean_password)
        
        # Create user
        user = User(
            id=uuid4(),
            email=request.email,
            name=request.name,
            phone=request.phone,
            password_hash=password_hash,
            auth_provider="password",
            timezone=request.timezone,
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        
        # Save to database
        created_user = await dao.create_user(user)
        
        # Generate access token
        access_token = create_access_token(
            data={
                "sub": str(created_user.id),
                "email": created_user.email,
            }
        )
        
        logger.info(f"User registered: {created_user.email} ({created_user.id})")
        
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=str(created_user.id),
            email=created_user.email,
            name=created_user.name,
        )
        
    except ValueError as e:
        # Re-raise as HTTPException for validation errors from DAO
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user",
        )


@router.post("/signin", response_model=AuthResponse)
async def signin(
    request: SignInRequest,
    dao: MongoDBDAO = Depends(get_dao),
):
    """
    Sign in an existing user.
    
    Authenticates user credentials and returns an access token.
    """
    try:
        # Get user by email
        user = await dao.get_user_by_email(request.email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive",
            )
        
        # Check if user has a password (not OAuth-only user)
        if not user.password_hash:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This account uses OAuth login. Please use Google login instead.",
            )
        
        # Verify password
        if not verify_password(request.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        
        # Generate access token
        access_token = create_access_token(
            data={
                "sub": str(user.id),
                "email": user.email,
            }
        )
        
        logger.info(f"User signed in: {user.email} ({user.id})")
        
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=str(user.id),
            email=user.email,
            name=user.name,
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Sign-in error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to sign in",
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """
    Get current authenticated user information.
    
    Returns user details for the authenticated user.
    """
    try:
        user_id = current_user["user_id"]
        user = await dao.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        return UserResponse(
            id=str(user.id),
            email=user.email,
            name=user.name,
            phone=user.phone,
            timezone=user.timezone,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user info error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user information",
        )


@router.post("/google", response_model=AuthResponse)
async def google_oauth(
    request: GoogleOAuthRequest,
    dao: MongoDBDAO = Depends(get_dao),
):
    """
    Google OAuth login/registration.
    
    Creates a new user if they don't exist, or logs in existing user.
    No password required - authentication is handled by Google.
    """
    try:
        # Combine first and last name
        full_name = f"{request.first_name} {request.last_name}".strip()
        
        # Check if user already exists
        existing_user = await dao.get_user_by_email(request.email)
        
        if existing_user:
            # User exists - log them in
            if not existing_user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is inactive",
                )
            
            # Update user info from Google (profile might have updated)
            needs_update = False
            if existing_user.name != full_name:
                existing_user.name = full_name
                needs_update = True
            if request.phone and not existing_user.phone:
                existing_user.phone = request.phone
                needs_update = True
            if existing_user.auth_provider != "google":
                existing_user.auth_provider = "google"
                needs_update = True
            
            if needs_update:
                existing_user.updated_at = datetime.utcnow()
                await dao.update_user(existing_user)
                logger.info(f"Updated user profile from Google: {existing_user.email}")
            
            # Generate access token
            access_token = create_access_token(
                data={
                    "sub": str(existing_user.id),
                    "email": existing_user.email,
                }
            )
            
            logger.info(f"User signed in via Google: {existing_user.email} ({existing_user.id})")
            
            return AuthResponse(
                access_token=access_token,
                token_type="bearer",
                user_id=str(existing_user.id),
                email=existing_user.email,
                name=existing_user.name,
            )
        else:
            # User doesn't exist - create new user
            new_user = User(
                id=uuid4(),
                email=request.email,
                name=full_name,
                phone=request.phone,
                password_hash=None,  # No password for OAuth users
                auth_provider="google",
                timezone=request.timezone,
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            
            # Save to database
            created_user = await dao.create_user(new_user)
            
            # Generate access token
            access_token = create_access_token(
                data={
                    "sub": str(created_user.id),
                    "email": created_user.email,
                }
            )
            
            logger.info(f"User registered via Google: {created_user.email} ({created_user.id})")
            
            return AuthResponse(
                access_token=access_token,
                token_type="bearer",
                user_id=str(created_user.id),
                email=created_user.email,
                name=created_user.name,
            )
        
    except ValueError as e:
        # Re-raise as HTTPException for validation errors from DAO
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Google OAuth error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process Google OAuth login",
        )

