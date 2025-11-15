"""
Conversation management routes for RT Gateway.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from shared.schemas import MeetingStatus, JoinMeetingRequest

from ..services import (
    active_conversations,
    broadcast_to_conversation,
)
from .. import services as rt_services
from services.api.dao import DynamoDBDAO, get_dao
from services.api.auth import get_current_user
from services.api.service_meeting import ServiceMeeting
from services.api.service_meeting_context import ServiceMeetingContext
from shared.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)
router = APIRouter()


class ConversationResponse(BaseModel):
    conversation_id: str
    meeting_id: Optional[str] = None
    meeting_url: Optional[str] = None
    meeting_ui_url: Optional[str] = None
    metadata: Dict[str, Any]


class ExternalMeetingStartRequest(JoinMeetingRequest):
    """Request payload for starting an external meeting."""


@router.post("/start")
async def start_conversation(
    request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
    dao: Optional[DynamoDBDAO] = Depends(get_dao),
) -> Dict[str, Any]:
    """Start a new conversation with rollback on failure."""
    meeting_id: Optional[str] = None
    conversation_id: Optional[str] = None
    meeting_service: Optional[ServiceMeeting] = None
    context_service: Optional[ServiceMeetingContext] = None
    
    try:
        room_id = request.get("room_id")
        request_user_id = request.get("user_id")
        # Prefer authenticated user id; fall back to request payload for backward compatibility
        user_id = current_user.get("user_id") if current_user else request_user_id
        context_id: Optional[str] = request.get("context_id")
        meeting_platform: Optional[str] = request.get("meeting_platform", "aurray")
        
        if not room_id or not user_id:
            raise HTTPException(status_code=400, detail="room_id and user_id are required")
        
        metadata: Dict[str, Any] = {}
        meeting_ui_url: Optional[str] = None
        context_id_str: Optional[str] = None

        # Use injected DAO when available; fallback to new instance
        dao_instance = dao or DynamoDBDAO()
        if not getattr(dao_instance, "initialized", False):
            await dao_instance.initialize()
        context_service = ServiceMeetingContext(dao_instance)

        if context_id:
            context = await context_service.get_context_by_id(context_id)
            if not context:
                raise HTTPException(status_code=404, detail="Meeting context not found")
            context_id_str = str(context.id)
        else:
            # Fetch the default context for the user using a context_service function
            context = await context_service.fetch_default_context(str(user_id))
            if not context:
                raise HTTPException(status_code=404, detail="Default meeting context not found! Please create one")
            context_id_str = str(context.id)

            # Normalize meeting platform: "clerk" -> "aurray", accept "aurray" directly
            if meeting_platform:
                platform_lower = meeting_platform.lower()
                if platform_lower == "clerk":
                    normalized_platform = "aurray"
                elif platform_lower == "aurray":
                    normalized_platform = "aurray"
                else:
                    normalized_platform = meeting_platform
            else:
                normalized_platform = "aurray"

            meeting_service = ServiceMeeting(dao_instance)
            meeting = await meeting_service.create_meeting_record(
                user_id=context.user_id,
                meeting_type=normalized_platform,
                context_id=context_id_str,
                voice_id=context.voice_id,
                bot_name=context.name,
                status=MeetingStatus.ACTIVE,
                description=f"Meeting created with {context.name}",
            )
            meeting_id = str(meeting.id)
            metadata.update(
                {
                    "meeting_id": meeting_id,
                    "context_id": context_id_str,
                    "meeting_url": meeting.meeting_url,
                }
            )

            if context_id_str is None:
                raise HTTPException(status_code=500, detail="Failed to resolve context identifier")

            payload = await context_service.fetch_context_payload(
                context_id_str, str(context.user_id)
            )
            if payload:
                try:
                    await context_service.cache_payload(meeting_id, payload)
                except Exception as cache_error:
                    logger.warning(
                        "Failed to cache meeting context payload in in-process cache: %s. Continuing without cache.",
                        cache_error,
                    )

        if rt_services.turn_manager is None:
            raise HTTPException(
                status_code=503,
                detail="Turn manager is not initialized. Please check server logs."
            )

        conversation = await rt_services.turn_manager.start_conversation(
            room_id, user_id, metadata=metadata
        )
        conversation_id = str(conversation.id)
        active_conversations[conversation_id] = conversation

        if meeting_id:
            base_frontend = settings.frontend_base_url or "http://localhost:3000"
            meeting_ui_url = (
                f"{base_frontend.rstrip('/')}/standalone-call"
                f"?meetingId={meeting_id}&conversationId={conversation_id}"
            )

        return {
            "conversation_id": conversation_id,
            "room_id": room_id,
            "status": "started",
            "meeting_id": meeting_id,
            "meeting_url": metadata.get("meeting_url"),
            "meeting_ui_url": meeting_ui_url,
        }
    except HTTPException:
        # HTTPExceptions are expected errors, don't rollback
        raise
    except Exception as e:
        logger.error(f"Failed to start conversation: {e}", exc_info=True)
        
        # Rollback: Clean up any resources that were created
        rollback_errors = []
        
        # Clean up conversation if it was created
        if conversation_id:
            try:
                if conversation_id in active_conversations:
                    del active_conversations[conversation_id]
                if rt_services.turn_manager is not None:
                    await rt_services.turn_manager.end_conversation(conversation_id)
                    logger.info(f"Rolled back conversation {conversation_id}")
                else:
                    logger.warning(f"Turn manager not initialized, skipped conversation cleanup for {conversation_id}")
            except Exception as cleanup_error:
                rollback_errors.append(f"Failed to cleanup conversation: {cleanup_error}")
                logger.error(f"Error during conversation rollback: {cleanup_error}")
        
        # Clean up meeting if it was created
        if meeting_id and meeting_service:
            try:
                await meeting_service.delete_meeting(meeting_id)
                logger.info(f"Rolled back meeting {meeting_id}")
            except Exception as cleanup_error:
                rollback_errors.append(f"Failed to cleanup meeting: {cleanup_error}")
                logger.error(f"Error during meeting rollback: {cleanup_error}")
        
        # Clean up cache if it was created
        if meeting_id and context_service:
            try:
                await context_service.clear_cached_payload(meeting_id)
                logger.info(f"Cleared cached payload for meeting {meeting_id}")
            except Exception as cleanup_error:
                # Cache cleanup failure is non-critical, just log
                logger.warning(f"Failed to clear cache during rollback: {cleanup_error}")
        
        if rollback_errors:
            logger.warning(f"Some rollback operations failed: {rollback_errors}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start conversation: {str(e)}"
        )


@router.post("/{conversation_id}/join")
async def join_conversation(conversation_id: str, request: Dict[str, str]) -> Dict[str, Any]:
    """Join an existing conversation."""
    try:
        user_id = request.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        conversation = active_conversations.get(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        if rt_services.turn_manager is None:
            raise HTTPException(
                status_code=503,
                detail="Turn manager is not initialized. Please check server logs."
            )
        
        await rt_services.turn_manager.add_participant_to_conversation(conversation_id, user_id)

        await broadcast_to_conversation(conversation_id, {
            "type": "participant_joined",
            "data": {
                "participant": {
                    "id": user_id,
                    "name": f"Participant {user_id[:8]}",
                    "joined_at": datetime.now(timezone.utc).isoformat()
                }
            }
        })

        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "status": "joined",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to join conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _resolve_base_api_url() -> str:
    """Resolve the base URL for calling internal API endpoints."""
    if settings.api_base_url:
        return settings.api_base_url.rstrip("/")

    host = settings.api_host or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = settings.api_port or 8000
    return f"http://{host}:{port}"


@router.post("/start-external")
async def start_external_meeting(
    payload: ExternalMeetingStartRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Start an external meeting by invoking the webhook join endpoint."""

    if not settings.aurray_api_key:
        raise HTTPException(status_code=500, detail="Aurray API key is not configured")

    webhook_url = f"{_resolve_base_api_url()}/v1/api.aurray.net/join_meeting"
    headers = {
        "Authorization": f"Bearer {settings.aurray_api_key}",
        "Content-Type": "application/json",
    }

    request_body = payload.model_dump(exclude_none=True)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(webhook_url, json=request_body, headers=headers)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = None
        try:
            detail = exc.response.json()
        except json.JSONDecodeError:
            detail = exc.response.text
        logger.error(
            "join_meeting webhook returned error %s: %s",
            exc.response.status_code,
            detail,
        )
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except httpx.RequestError as exc:
        logger.error("Failed to reach join_meeting webhook: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to contact meeting webhook") from exc
    try:
        data = response.json()
    except json.JSONDecodeError:
        logger.warning("join_meeting webhook returned non-JSON payload: %s", response.text)
        return {
            "success": True,
            "message": response.text,
            "platform": payload.type,
        }

    normalized = {
        "success": data.get("success", True),
        "message": data.get("message"),
        "status": data.get("status"),
        "timestamp": data.get("timestamp"),
        "meeting_id": data.get("meeting_id"),
        "meeting_url": data.get("meeting_url"),
        "meeting_ui_url": data.get("meeting_ui_url"),
        "platform": data.get("platform", payload.type),
        "voice_id": data.get("voice_id"),
        "capabilities": data.get("capabilities"),
    }

    logger.info("External meeting started successfully via webhook", extra={"response": normalized})
    return normalized


@router.post("/{conversation_id}/end")
async def end_conversation(conversation_id: str) -> Dict[str, Any]:
    """End a conversation."""
    try:
        if rt_services.turn_manager is None:
            logger.warning("Turn manager is not initialized, cleaning up active_conversations only")
            if conversation_id in active_conversations:
                del active_conversations[conversation_id]
            return {"status": "ended", "note": "Turn manager not initialized"}
        
        conversation = active_conversations.get(conversation_id)
        if not conversation:
            logger.warning(f"Conversation {conversation_id} not found in active_conversations")
            try:
                await rt_services.turn_manager.end_conversation(conversation_id)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup turn_manager state: {cleanup_error}")
            
            return {"status": "ended", "note": "Conversation was not active"}

        await rt_services.turn_manager.end_conversation(conversation_id)
        del active_conversations[conversation_id]

        logger.info(f"Successfully ended conversation {conversation_id}")
        return {"status": "ended"}
        
    except Exception as e:
        logger.error(f"Failed to end conversation {conversation_id}: {e}")
        try:
            if conversation_id in active_conversations:
                del active_conversations[conversation_id]
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str) -> Dict[str, Any]:
    """Delete a conversation and all related data."""
    try:
        conversation = active_conversations.get(conversation_id)
        
        if conversation:
            logger.info(f"Ending active conversation {conversation_id} before deletion")
            if rt_services.turn_manager is not None:
                await rt_services.turn_manager.end_conversation(conversation_id)
            del active_conversations[conversation_id]
        
        try:
            # Use dependency-injected DAO or get from global instance
            from services.api.dao import get_dao
            try:
                dao_instance = get_dao()
            except RuntimeError:
                # Fallback if DAO not initialized (shouldn't happen in production)
                from services.api.dao import DynamoDBDAO
                dao_instance = DynamoDBDAO()
                await dao_instance.initialize()
            
            await dao_instance.delete_conversation(conversation_id)
            logger.info(f"Deleted conversation {conversation_id} from database")
        except Exception as db_error:
            logger.error(f"Failed to delete conversation from database: {db_error}")
            raise HTTPException(status_code=500, detail=f"Database deletion failed: {str(db_error)}")
        
        return {"status": "deleted", "conversation_id": conversation_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk-delete")
async def bulk_delete_conversations(request: Dict[str, Any]) -> Dict[str, Any]:
    """Delete multiple conversations and all related data."""
    try:
        conversation_ids = request.get("conversation_ids", [])
        
        if not conversation_ids:
            raise HTTPException(status_code=400, detail="No conversation IDs provided")
        
        if not isinstance(conversation_ids, list):
            raise HTTPException(status_code=400, detail="conversation_ids must be a list")
        
        deleted_count = 0
        failed_deletions = []
        
        for conversation_id in conversation_ids:
            try:
                conversation = active_conversations.get(conversation_id)
                
                if conversation:
                    logger.info(f"Ending active conversation {conversation_id} before deletion")
                    if rt_services.turn_manager is not None:
                        await rt_services.turn_manager.end_conversation(conversation_id)
                    del active_conversations[conversation_id]
                
                try:
                    # Use dependency-injected DAO or get from global instance
                    from services.api.dao import get_dao
                    try:
                        dao_instance = get_dao()
                    except RuntimeError:
                        # Fallback if DAO not initialized (shouldn't happen in production)
                        from services.api.dao import DynamoDBDAO
                        dao_instance = DynamoDBDAO()
                        await dao_instance.initialize()
                    
                    await dao_instance.delete_conversation(conversation_id)
                    logger.info(f"Deleted conversation {conversation_id} from database")
                    deleted_count += 1
                except Exception as db_error:
                    logger.error(f"Failed to delete conversation {conversation_id} from database: {db_error}")
                    failed_deletions.append({
                        "conversation_id": conversation_id,
                        "error": f"Database deletion failed: {str(db_error)}"
                    })
                    
            except Exception as e:
                logger.error(f"Failed to delete conversation {conversation_id}: {e}")
                failed_deletions.append({
                    "conversation_id": conversation_id,
                    "error": str(e)
                })
        
        return {
            "status": "completed",
            "deleted_count": deleted_count,
            "total_requested": len(conversation_ids),
            "failed_deletions": failed_deletions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to bulk delete conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

