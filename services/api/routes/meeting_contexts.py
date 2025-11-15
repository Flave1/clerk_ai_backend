"""
Meeting Contexts API routes.
"""
import logging
from datetime import datetime
from typing import List
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from shared.schemas import MeetingContext, MeetingRole, TonePersonality
from shared.config import get_settings

from ..auth import get_current_user
from ..dao import DynamoDBDAO, get_dao
from services.api.service_meeting_context import ServiceMeetingContext

logger = logging.getLogger(__name__)

router = APIRouter()


def get_meeting_context_service(dao: DynamoDBDAO) -> ServiceMeetingContext:
    return ServiceMeetingContext(dao)


class MeetingContextCreate(BaseModel):
    """Request model for creating a meeting context."""
    
    name: str
    voice_id: str
    context_description: str
    tools_integrations: List[str] = []
    meeting_role: MeetingRole = MeetingRole.PARTICIPANT
    tone_personality: TonePersonality = TonePersonality.FRIENDLY
    custom_tone: str = None
    is_default: bool = False


class MeetingContextUpdate(BaseModel):
    """Request model for updating a meeting context."""
    
    name: str = None
    voice_id: str = None
    context_description: str = None
    tools_integrations: List[str] = None
    meeting_role: MeetingRole = None
    tone_personality: TonePersonality = None
    custom_tone: str = None
    is_default: bool = None


class MeetingContextResponse(BaseModel):
    """Response model for meeting context."""
    
    id: str
    user_id: str
    name: str
    voice_id: str
    context_description: str
    tools_integrations: List[str]
    meeting_role: str
    tone_personality: str
    custom_tone: str = None
    created_at: str
    updated_at: str
    is_default: bool
    
    @classmethod
    def from_meeting_context(cls, context: MeetingContext):
        """Create response from MeetingContext model."""
        return cls(
            id=str(context.id),
            user_id=str(context.user_id),
            name=context.name,
            voice_id=context.voice_id,
            context_description=context.context_description,
            tools_integrations=context.tools_integrations,
            meeting_role=context.meeting_role.value,
            tone_personality=context.tone_personality.value,
            custom_tone=context.custom_tone,
            created_at=context.created_at.isoformat(),
            updated_at=context.updated_at.isoformat(),
            is_default=context.is_default,
        )


@router.post("/", response_model=MeetingContextResponse, status_code=status.HTTP_201_CREATED)
async def create_meeting_context(
    context_data: MeetingContextCreate,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao),
):
    """Create a new meeting context."""
    try:
        user_id = UUID(current_user["user_id"])
        user_id_str = str(user_id)
        
        context_service = get_meeting_context_service(dao)

        if context_data.is_default:
            await context_service.clear_default_contexts(user_id_str)
        
        meeting_context = MeetingContext(
            id=uuid4(),
            user_id=user_id,
            name=context_data.name,
            voice_id=context_data.voice_id,
            context_description=context_data.context_description,
            tools_integrations=context_data.tools_integrations,
            meeting_role=context_data.meeting_role,
            tone_personality=context_data.tone_personality,
            custom_tone=context_data.custom_tone,
            is_default=context_data.is_default,
        )
        
        created_context = await context_service.create_context(meeting_context)
        return MeetingContextResponse.from_meeting_context(created_context)
        
    except Exception as e:
        logger.error(f"Failed to create meeting context: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create meeting context"
        )


@router.get("/", response_model=List[MeetingContextResponse])
async def get_meeting_contexts(
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao),
):
    """Get all meeting contexts for the current user."""
    try:
        user_id = current_user["user_id"]
        context_service = get_meeting_context_service(dao)
        contexts = await context_service.get_contexts_by_user(user_id)
        return [MeetingContextResponse.from_meeting_context(ctx) for ctx in contexts]
        
    except Exception as e:
        logger.error(f"Failed to get meeting contexts: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve meeting contexts"
        )


@router.get("/{context_id}", response_model=MeetingContextResponse)
async def get_meeting_context(
    context_id: str,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao),
):
    """Get a specific meeting context by ID."""
    try:
        user_id = current_user["user_id"]
        context_service = get_meeting_context_service(dao)
        context = await context_service.get_context(context_id, user_id)
        
        if not context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Meeting context not found"
            )
        
        return MeetingContextResponse.from_meeting_context(context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get meeting context: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve meeting context"
        )


@router.put("/{context_id}", response_model=MeetingContextResponse)
async def update_meeting_context(
    context_id: str,
    context_data: MeetingContextUpdate,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao),
):
    """Update a meeting context."""
    try:
        user_id = current_user["user_id"]
        
        # Get existing context
        context_service = get_meeting_context_service(dao)
        existing_context = await context_service.get_context(context_id, user_id)
        if not existing_context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Meeting context not found"
            )
        
        # Update fields
        if context_data.name is not None:
            existing_context.name = context_data.name
        if context_data.voice_id is not None:
            existing_context.voice_id = context_data.voice_id
        if context_data.context_description is not None:
            existing_context.context_description = context_data.context_description
        if context_data.tools_integrations is not None:
            existing_context.tools_integrations = context_data.tools_integrations
        if context_data.meeting_role is not None:
            existing_context.meeting_role = context_data.meeting_role
        if context_data.tone_personality is not None:
            existing_context.tone_personality = context_data.tone_personality
        if context_data.custom_tone is not None:
            existing_context.custom_tone = context_data.custom_tone
        if context_data.is_default is not None:
            if context_data.is_default:
                await context_service.clear_default_contexts(user_id, exclude_context_id=context_id)
            existing_context.is_default = context_data.is_default
        
        updated_context = await context_service.update_context(existing_context)
        return MeetingContextResponse.from_meeting_context(updated_context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update meeting context: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update meeting context"
        )


@router.delete("/{context_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_meeting_context(
    context_id: str,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao),
):
    """Delete a meeting context."""
    try:
        user_id = current_user["user_id"]
        context_service = get_meeting_context_service(dao)
        deleted = await context_service.delete_context(context_id, user_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Meeting context not found"
            )
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete meeting context: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete meeting context"
        )

