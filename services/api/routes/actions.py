"""
Actions API routes.
"""
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from shared.schemas import Action, ActionStatus, ActionType

from ..auth import get_current_user
from ..dao import MongoDBDAO, get_dao

logger = logging.getLogger(__name__)

router = APIRouter()


class ActionResponse(BaseModel):
    """Action response model."""

    id: str
    turn_id: Optional[str]
    action_type: str
    status: str
    parameters: dict
    result: Optional[dict]
    error_message: Optional[str]
    created_at: str
    completed_at: Optional[str]


class ActionUpdate(BaseModel):
    """Action update request."""

    status: str
    result: Optional[dict] = None
    error_message: Optional[str] = None


@router.get("/", response_model=List[ActionResponse])
async def get_actions(
    action_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """Get list of actions for the authenticated user."""
    try:
        actions = await dao.get_actions(
            action_type=action_type,
            status=status,
            limit=limit,
            offset=offset,
        )

        response = []
        for action in actions:
            response.append(
                ActionResponse(
                    id=str(action.id),
                    turn_id=str(action.turn_id) if action.turn_id else None,
                    action_type=action.action_type.value,
                    status=action.status.value,
                    parameters=action.parameters,
                    result=action.result,
                    error_message=action.error_message,
                    created_at=action.created_at.isoformat(),
                    completed_at=action.completed_at.isoformat()
                    if action.completed_at
                    else None,
                )
            )

        return response

    except Exception as e:
        logger.error(f"Failed to get actions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{action_id}", response_model=ActionResponse)
async def get_action(
    action_id: str,
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """Get a specific action."""
    try:
        action = await dao.get_action(action_id)
        if not action:
            raise HTTPException(status_code=404, detail="Action not found")

        return ActionResponse(
            id=str(action.id),
            turn_id=str(action.turn_id) if action.turn_id else None,
            action_type=action.action_type.value,
            status=action.status.value,
            parameters=action.parameters,
            result=action.result,
            error_message=action.error_message,
            created_at=action.created_at.isoformat(),
            completed_at=action.completed_at.isoformat()
            if action.completed_at
            else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get action {action_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{action_id}", response_model=ActionResponse)
async def update_action(
    action_id: str,
    update: ActionUpdate,
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """Update an action."""
    try:
        action = await dao.get_action(action_id)
        if not action:
            raise HTTPException(status_code=404, detail="Action not found")

        # Update action
        action.status = ActionStatus(update.status)
        if update.result is not None:
            action.result = update.result
        if update.error_message is not None:
            action.error_message = update.error_message

        if action.status in [
            ActionStatus.COMPLETED,
            ActionStatus.FAILED,
            ActionStatus.CANCELLED,
        ]:
            from datetime import datetime

            action.completed_at = datetime.utcnow()

        updated_action = await dao.update_action(action)

        return ActionResponse(
            id=str(updated_action.id),
            turn_id=str(updated_action.turn_id) if updated_action.turn_id else None,
            action_type=updated_action.action_type.value,
            status=updated_action.status.value,
            parameters=updated_action.parameters,
            result=updated_action.result,
            error_message=updated_action.error_message,
            created_at=updated_action.created_at.isoformat(),
            completed_at=updated_action.completed_at.isoformat()
            if updated_action.completed_at
            else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update action {action_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
