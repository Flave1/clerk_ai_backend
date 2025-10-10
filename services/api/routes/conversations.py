"""
Conversation API routes.
"""
import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from shared.schemas import Conversation, ConversationStatus, Turn

from ..dao import DynamoDBDAO, get_dao

logger = logging.getLogger(__name__)

router = APIRouter()


class ConversationCreate(BaseModel):
    """Conversation creation request."""

    user_id: str
    room_id: str
    metadata: Optional[dict] = {}


class ConversationResponse(BaseModel):
    """Conversation response model."""

    id: str
    user_id: str
    room_id: str
    status: str
    started_at: datetime
    ended_at: Optional[datetime]
    summary: Optional[str]
    metadata: dict
    turn_count: int


class TurnResponse(BaseModel):
    """Turn response model."""

    id: str
    conversation_id: str
    turn_number: int
    turn_type: str
    content: str
    confidence_score: Optional[float]
    timestamp: datetime


@router.get("/", response_model=List[ConversationResponse])
async def get_conversations(
    user_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(10, le=100),
    offset: int = Query(0, ge=0),
    dao: DynamoDBDAO = Depends(get_dao),
):
    """Get list of conversations."""
    try:
        conversations = await dao.get_conversations(
            user_id=user_id, status=status, limit=limit, offset=offset
        )

        # Convert to response format
        response = []
        for conv in conversations:
            turn_count = await dao.get_turn_count(str(conv.id))
            response.append(
                ConversationResponse(
                    id=str(conv.id),
                    user_id=str(conv.user_id),
                    room_id=conv.room_id,
                    status=conv.status.value,
                    started_at=conv.started_at,
                    ended_at=conv.ended_at,
                    summary=conv.summary,
                    metadata=conv.metadata,
                    turn_count=turn_count,
                )
            )

        return response

    except Exception as e:
        logger.error(f"Failed to get conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, dao: DynamoDBDAO = Depends(get_dao)):
    """Get a specific conversation."""
    try:
        conversation = await dao.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        turn_count = await dao.get_turn_count(conversation_id)

        return ConversationResponse(
            id=str(conversation.id),
            user_id=str(conversation.user_id),
            room_id=conversation.room_id,
            status=conversation.status.value,
            started_at=conversation.started_at,
            ended_at=conversation.ended_at,
            summary=conversation.summary,
            metadata=conversation.metadata,
            turn_count=turn_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationCreate, dao: DynamoDBDAO = Depends(get_dao)
):
    """Create a new conversation."""
    try:
        conversation = Conversation(
            user_id=UUID(request.user_id),
            room_id=request.room_id,
            metadata=request.metadata,
        )

        created_conversation = await dao.create_conversation(conversation)

        return ConversationResponse(
            id=str(created_conversation.id),
            user_id=str(created_conversation.user_id),
            room_id=created_conversation.room_id,
            status=created_conversation.status.value,
            started_at=created_conversation.started_at,
            ended_at=created_conversation.ended_at,
            summary=created_conversation.summary,
            metadata=created_conversation.metadata,
            turn_count=0,
        )

    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{conversation_id}/status")
async def update_conversation_status(
    conversation_id: str,
    status: str,
    summary: Optional[str] = None,
    dao: DynamoDBDAO = Depends(get_dao),
):
    """Update conversation status."""
    try:
        conversation = await dao.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Update status
        conversation.status = ConversationStatus(status)
        if status == "completed":
            conversation.ended_at = datetime.utcnow()
        if summary:
            conversation.summary = summary

        updated_conversation = await dao.update_conversation(conversation)

        return {
            "id": str(updated_conversation.id),
            "status": updated_conversation.status.value,
            "updated_at": datetime.utcnow(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update conversation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}/turns", response_model=List[TurnResponse])
async def get_conversation_turns(
    conversation_id: str,
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    dao: DynamoDBDAO = Depends(get_dao),
):
    """Get turns for a conversation."""
    try:
        # Verify conversation exists
        conversation = await dao.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        turns = await dao.get_conversation_turns(
            conversation_id=conversation_id, limit=limit, offset=offset
        )

        response = []
        for turn in turns:
            response.append(
                TurnResponse(
                    id=str(turn.id),
                    conversation_id=str(turn.conversation_id),
                    turn_number=turn.turn_number,
                    turn_type=turn.turn_type.value,
                    content=turn.content,
                    confidence_score=turn.confidence_score,
                    timestamp=turn.timestamp,
                )
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation turns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str, dao: DynamoDBDAO = Depends(get_dao)
):
    """Delete a conversation."""
    try:
        conversation = await dao.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Delete conversation and associated turns/actions
        await dao.delete_conversation(conversation_id)

        return {"message": "Conversation deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
