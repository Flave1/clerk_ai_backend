"""
Room management API routes.
"""
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from shared.schemas import RoomInfo

from ..auth import get_current_user
from ..dao import DynamoDBDAO, get_dao

logger = logging.getLogger(__name__)

router = APIRouter()


class RoomResponse(BaseModel):
    """Room response model."""

    room_id: str
    name: str
    participant_count: int
    is_active: bool
    created_at: str


class RoomCreate(BaseModel):
    """Room creation request."""

    room_id: str
    name: str


@router.get("/", response_model=List[RoomResponse])
async def get_rooms(
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao),
):
    """Get list of active rooms for the authenticated user."""
    try:
        rooms = await dao.get_active_rooms()

        response = []
        for room in rooms:
            response.append(
                RoomResponse(
                    room_id=room.room_id,
                    name=room.name,
                    participant_count=room.participant_count,
                    is_active=room.is_active,
                    created_at=room.created_at.isoformat(),
                )
            )

        return response

    except Exception as e:
        logger.error(f"Failed to get rooms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{room_id}", response_model=RoomResponse)
async def get_room(
    room_id: str,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao),
):
    """Get a specific room."""
    try:
        room = await dao.get_room(room_id)
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")

        return RoomResponse(
            room_id=room.room_id,
            name=room.name,
            participant_count=room.participant_count,
            is_active=room.is_active,
            created_at=room.created_at.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get room {room_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=RoomResponse)
async def create_room(
    request: RoomCreate,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao),
):
    """Create a new room."""
    try:
        room = RoomInfo(
            room_id=request.room_id,
            name=request.name,
            participant_count=0,
            is_active=True,
        )

        created_room = await dao.create_room(room)

        return RoomResponse(
            room_id=created_room.room_id,
            name=created_room.name,
            participant_count=created_room.participant_count,
            is_active=created_room.is_active,
            created_at=created_room.created_at.isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to create room: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{room_id}/participants")
async def update_room_participants(
    room_id: str,
    participant_count: int,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao),
):
    """Update room participant count."""
    try:
        room = await dao.get_room(room_id)
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")

        room.participant_count = participant_count
        room.is_active = participant_count > 0

        updated_room = await dao.update_room(room)

        return RoomResponse(
            room_id=updated_room.room_id,
            name=updated_room.name,
            participant_count=updated_room.participant_count,
            is_active=updated_room.is_active,
            created_at=updated_room.created_at.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update room participants: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{room_id}")
async def delete_room(
    room_id: str,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao),
):
    """Delete a room."""
    try:
        room = await dao.get_room(room_id)
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")

        await dao.delete_room(room_id)

        return {"message": "Room deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete room: {e}")
        raise HTTPException(status_code=500, detail=str(e))
