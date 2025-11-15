"""
Newsletter/Waiting List API routes backed by DynamoDB.
"""
import logging
from datetime import timezone, datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from botocore.exceptions import ClientError

from services.api.dao import DynamoDBDAO, get_dao
from shared.schemas import NewsletterSubscription

logger = logging.getLogger(__name__)

router = APIRouter()


class NewsletterSignupRequest(BaseModel):
    """Newsletter signup request."""
    
    name: str = Field(..., min_length=1, max_length=200, description="Full name")
    email: EmailStr = Field(..., description="Email address")
    country: str = Field(..., min_length=1, max_length=100, description="Country")


class NewsletterSignupResponse(BaseModel):
    """Newsletter signup response."""
    
    success: bool
    message: str
    timestamp: str


@router.post("/", response_model=NewsletterSignupResponse, status_code=status.HTTP_201_CREATED)
async def signup_newsletter(
    request: NewsletterSignupRequest,
    dao: DynamoDBDAO = Depends(get_dao),
):
    """
    Add a user to the waiting list/newsletter.
    
    This endpoint is public and does not require authentication.
    """
    try:
        email = request.email.lower()

        existing = await dao.get_newsletter_signup(email)
        if existing:
            return NewsletterSignupResponse(
                success=True,
                message="You're already on the waiting list!",
                timestamp=existing.signed_up_at.isoformat(),
            )

        signup = NewsletterSubscription(
            email=email,
            name=request.name,
            country=request.country,
        )

        try:
            await dao.add_newsletter_signup(signup)
        except ValueError:
            return NewsletterSignupResponse(
                success=True,
                message="You're already on the waiting list!",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        
        logger.info("New newsletter signup stored for %s", email)
        
        return NewsletterSignupResponse(
            success=True,
            message="Successfully added to waiting list!",
            timestamp=signup.signed_up_at.isoformat(),
        )
    
    except HTTPException:
        raise
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "ResourceNotFoundException":
            logger.error(f"Newsletter table does not exist: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Newsletter service is not configured. Please create the newsletter table in DynamoDB.",
            )
        logger.error(f"Failed to add to waiting list: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add to waiting list: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Failed to add to waiting list: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add to waiting list: {str(e)}",
        )


@router.get("/", status_code=status.HTTP_200_OK)
async def get_waiting_list(dao: DynamoDBDAO = Depends(get_dao)):
    """
    Get all waiting list entries.
    
    Note: This endpoint should be protected in production.
    For now, it's public for testing purposes.
    """
    try:
        waiting_list = await dao.list_newsletter_signups()
        return {
            "success": True,
            "count": len(waiting_list),
            "entries": [
                {
                    "email": signup.email,
                    "name": signup.name,
                    "country": signup.country,
                    "signed_up_at": signup.signed_up_at.isoformat(),
                }
                for signup in waiting_list
            ],
        }
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "ResourceNotFoundException":
            logger.error(f"Newsletter table does not exist: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Newsletter service is not configured. Please create the newsletter table in DynamoDB.",
            )
        logger.error(f"Failed to get waiting list: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get waiting list: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Failed to get waiting list: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get waiting list: {str(e)}",
        )
