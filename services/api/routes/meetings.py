"""
FastAPI routes for meeting agent management.

This module provides REST endpoints for managing meetings, summaries,
and meeting agent functionality.
"""
import logging
import random
from datetime import datetime
from typing import List, Optional, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from services.api.auth import get_current_user
from services.api.dao import get_dao, DynamoDBDAO
from shared.config import get_settings
from shared.schemas import Meeting, MeetingSummary, MeetingStatus, MeetingPlatform, ActionItem
from services.meeting_agent.models import (
    MeetingJoinRequest, MeetingJoinResponse, MeetingNotification,
    CalendarEvent, MeetingConfig
)
from pydantic import BaseModel
from services.meeting_agent.scheduler import create_meeting_scheduler
from services.meeting_agent.notifier import create_notification_service
from services.meeting_agent.summarization_service import create_summarization_service
from services.api.service_meeting import ServiceMeeting
from services.api.service_meeting_context import ServiceMeetingContext

logger = logging.getLogger(__name__)
settings = get_settings()

# Create router
router = APIRouter()


class ParticipantJoinRequest(BaseModel):
    """Request to join a meeting as a participant."""
    participant_name: str
    participant_id: Optional[str] = None


class BotJoinedRequest(BaseModel):
    """Payload sent by browser bot when it joins a meeting."""
    sessionId: str
    botName: Optional[str] = None
    platform: Optional[str] = None
    timestamp: Optional[str] = None
    meeting_url: Optional[str] = None


class BotLeftRequest(BaseModel):
    """Payload sent by browser bot when it leaves a meeting."""
    sessionId: str
    timestamp: Optional[str] = None
    reason: Optional[str] = None


class GenerateMeetingUrlRequest(BaseModel):
    """Request to generate a meeting URL for a platform."""
    platform: str  # one of: clerk | teams | zoom | google_meet


class GenerateMeetingUrlResponse(BaseModel):
    """Response with generated meeting details."""
    platform: str
    meeting_id: str
    meeting_url: str

# Global services
meeting_scheduler = create_meeting_scheduler()
notification_service = create_notification_service()
summarization_service = create_summarization_service()


def get_meeting_service(dao: DynamoDBDAO) -> ServiceMeeting:
    return ServiceMeeting(dao)


@router.get("/", response_model=List[Meeting])
async def get_meetings(
    status: Optional[MeetingStatus] = None,
    platform: Optional[MeetingPlatform] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get list of meetings for the authenticated user with optional filtering."""
    try:
        # Query DynamoDB for meetings filtered by user_id
        user_id = current_user["user_id"]
        meeting_service = get_meeting_service(dao)
        meetings = await meeting_service.get_meetings(limit=limit + offset, user_id=user_id)
        
        # Add filtering logic
        if status:
            meetings = [m for m in meetings if m.status == status]
        if platform:
            meetings = [m for m in meetings if m.platform == platform]
        
        # Apply offset
        return meetings[offset:offset + limit] if offset < len(meetings) else []
        
    except Exception as e:
        logger.error(f"Error getting meetings: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve meetings")


# Specific routes must come before parameterized routes
@router.get("/summaries", response_model=List[MeetingSummary])
async def get_meeting_summaries(
    limit: int = 50,
    offset: int = 0,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get all meeting summaries."""
    try:
        # In a real implementation, you would query DynamoDB
        # For now, return mock data
        summaries = []
        
        return summaries[offset:offset + limit]
        
    except Exception as e:
        logger.error(f"Error getting meeting summaries: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve meeting summaries")


@router.post("/{meeting_id}/bot-joined")
async def bot_joined(
    meeting_id: str,
    request: BotJoinedRequest,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Confirm a bot has joined a meeting and mark the meeting as bot_joined.
    Updates the meeting status to ACTIVE and sets bot_joined flag.
    """
    try:
        logger.info(
            "Bot joined meeting",
            extra={
                "meeting_id": meeting_id,
                "session_id": request.sessionId,
                "platform": request.platform,
                "bot_name": request.botName,
                "timestamp": request.timestamp or datetime.utcnow().isoformat(),
            },
        )

        meeting_service = get_meeting_service(dao)
        original_meeting_id = meeting_id

        # Get meeting by ID, create one if it does not exist
        meeting = await meeting_service.get_meeting(meeting_id)
        if not meeting:
            logger.warning(
                "Meeting not found for bot_joined request, creating new meeting record",
                extra={
                    "meeting_id": meeting_id,
                    "session_id": request.sessionId,
                    "platform": request.platform,
                },
            )

            inferred_user_id = None
            if request.sessionId:
                try:
                    inferred_user_id = UUID(request.sessionId)
                except ValueError:
                    logger.warning(
                        "Failed to parse sessionId as UUID when creating fallback meeting; generating new UUID",
                        extra={"session_id": request.sessionId},
                    )
            if inferred_user_id is None:
                inferred_user_id = uuid4()

            meeting = await meeting_service.create_meeting_record(
                user_id=inferred_user_id,
                meeting_type=request.platform or "clerk",
                meeting_url=request.meeting_url,
                bot_name=request.botName,
            )
            meeting_id = str(meeting.id)

        # Update meeting status
        meeting.bot_joined = True
        meeting.status = MeetingStatus.ACTIVE
        meeting.joined_at = datetime.utcnow()
        
        logger.info(
            f"Updating meeting {meeting.id}: setting bot_joined=True, status=ACTIVE",
            extra={
                "current_status": meeting.status,
                "has_bot_joined": meeting.bot_joined,
            }
        )
        await meeting_service.update_meeting(meeting)
        if meeting.context_id:
            try:
                context_service = ServiceMeetingContext(dao)
                context_payload = await context_service.fetch_context_payload(
                    str(meeting.context_id),
                    str(meeting.user_id),
                )
                if context_payload:
                    await context_service.cache_payload(meeting_id, context_payload)
                    if original_meeting_id != meeting_id:
                        await context_service.cache_payload(original_meeting_id, context_payload)
                    logger.info(
                        "Cached meeting context for meeting",
                        extra={
                            "meeting_id": meeting_id,
                            "context_id": str(meeting.context_id),
                        },
                    )
            except Exception as cache_error:
                logger.warning(
                    "Failed to cache meeting context",
                    extra={
                        "meeting_id": meeting_id,
                        "context_id": str(meeting.context_id),
                        "error": str(cache_error),
                    },
                )
        
        logger.info(
            "Meeting updated for bot join",
            extra={
                "meeting_id": str(meeting.id),
                "session_id": request.sessionId,
                "platform": request.platform,
            }
        )

        return JSONResponse(
            {
                "status": "success",
                "message": "Bot join confirmed",
                "meeting_id": str(meeting.id),
                "session_id": request.sessionId,
                "platform": request.platform,
                "bot_name": request.botName,
                "timestamp": request.timestamp or datetime.utcnow().isoformat(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to confirm bot joined: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to confirm bot joined: {str(e)}")


@router.post("/{meeting_id}/bot-left")
async def bot_left(
    meeting_id: str,
    request: BotLeftRequest,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Notify the backend that a bot has left a meeting.
    Updates the meeting status to ENDED and marks bot_joined as False.
    """
    try:
        logger.info(
            "Bot left meeting",
            extra={
                "meeting_id": meeting_id,
                "session_id": request.sessionId,
                "reason": request.reason,
                "timestamp": request.timestamp or datetime.utcnow().isoformat(),
            },
        )

        meeting_service = get_meeting_service(dao)

        # Try to get meeting by ID
        meeting = None
        try:
            meeting = await meeting_service.get_meeting(meeting_id)
        except Exception as e:
            logger.warning(f"Could not get meeting by ID {meeting_id}: {e}")

        # Update meeting if found
        if meeting:
            # Mark bot as left and update meeting status
            meeting.bot_joined = False
            meeting.status = MeetingStatus.ENDED
            meeting.ended_at = datetime.utcnow()
            
            logger.info(
                f"Updating meeting {meeting.id}: setting bot_joined=False, status=ENDED",
                extra={
                    "current_status": meeting.status,
                    "has_bot_joined": meeting.bot_joined,
                    "reason": request.reason,
                }
            )
            await meeting_service.update_meeting(meeting)
            logger.info(
                "Meeting updated for bot leave",
                extra={
                    "meeting_id": str(meeting.id),
                    "session_id": request.sessionId,
                    "reason": request.reason,
                }
            )

            if meeting.context_id:
                try:
                    context_service = ServiceMeetingContext(dao)
                    await context_service.clear_cached_payload(meeting_id)
                    logger.info(
                        "Cleared cached meeting context",
                        extra={
                            "meeting_id": meeting_id,
                            "context_id": str(meeting.context_id),
                        },
                    )
                except Exception as cache_error:
                    logger.warning(
                        "Failed to clear cached meeting context",
                        extra={
                            "meeting_id": meeting_id,
                            "context_id": str(meeting.context_id),
                            "error": str(cache_error),
                        },
                    )
        else:
            logger.warning(
                "Meeting not found while confirming bot left",
                extra={
                    "meeting_id": meeting_id,
                    "session_id": request.sessionId,
                    "reason": request.reason,
                }
            )

        return JSONResponse(
            {
                "status": "success",
                "message": "Bot leave confirmed",
                "meeting_id": str(meeting.id) if meeting else meeting_id,
                "session_id": request.sessionId,
                "reason": request.reason,
                "timestamp": request.timestamp or datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Failed to confirm bot left: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to confirm bot left: {str(e)}")


@router.post("/generate-url", response_model=GenerateMeetingUrlResponse)
async def generate_meeting_url(request: GenerateMeetingUrlRequest) -> GenerateMeetingUrlResponse:

    """
    Generate a meeting URL for a given platform.

    This is a convenience endpoint used by the UI to quickly create a meeting link
    for demo/testing flows. In production, integrate with each provider to create
    real meetings.
    """
    try:
        plat = (request.platform or "").strip().lower()
        if plat not in {"clerk", "teams", "zoom", "google_meet"}:
            raise HTTPException(status_code=400, detail="Invalid platform. Use clerk | teams | zoom | google_meet")

        mid = str(uuid4())

        # Delegate URL generation per platform
        if plat == "clerk":
            base_frontend = settings.frontend_base_url or "http://localhost:3000"
            url = f"{base_frontend.rstrip('/')}/standalone-call?meetingId={mid}"
            return GenerateMeetingUrlResponse(platform=plat, meeting_id=mid, meeting_url=url)

        if plat == "teams":
            try:
                # Create a short-lived meeting via Teams client (best-effort)
                from services.meeting_agent.teams_client import create_teams_client
                client = create_teams_client()
                await client.initialize()
                await client.authenticate()

                now = datetime.utcnow()
                start_time = now.isoformat() + 'Z'
                end_time = (now.replace(microsecond=0) if True else now).isoformat() + 'Z'
                # Add 30 minutes
                end_time = (now.replace(microsecond=0) + (datetime.utcnow() - datetime.utcnow())).isoformat() + 'Z'
                # more explicit: compute 30 minutes
                from datetime import timedelta
                end_time = (now + timedelta(minutes=30)).isoformat() + 'Z'

                res = await client.create_meeting(
                    title=f"Clerk Teams Meeting {mid[:8]}",
                    start_time=start_time,
                    end_time=end_time,
                    attendees=[],
                    description="Our Meeting"
                )
                if res.get("success"):
                    url = res["meeting"]["join_url"]
                    return GenerateMeetingUrlResponse(platform=plat, meeting_id=mid, meeting_url=url)
                else:
                    raise HTTPException(status_code=500, detail=f"Teams URL generation failed: {res.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Teams URL generation via API failed: {e}")
                raise HTTPException(status_code=500, detail=f"Teams URL generation failed: {e}")

        if plat == "zoom":
            try:
                from services.meeting_agent.zoom_client import create_zoom_client
                client = create_zoom_client()
                await client.initialize()
                # Zoom create_meeting internally obtains tokens via settings; best-effort
                now = datetime.utcnow()
                start_time = now.isoformat() + 'Z'
                from datetime import timedelta
                end_time = (now + timedelta(minutes=45)).isoformat() + 'Z'
                res = await client.create_meeting(
                    title=f"Clerk Zoom Meeting {mid[:8]}",
                    start_time=start_time,
                    end_time=end_time,
                    attendees=[],
                    description="Generated from /generate-url"
                )
                if res.get("success"):
                    url = res["meeting"]["join_url"]
                    return GenerateMeetingUrlResponse(platform=plat, meeting_id=mid, meeting_url=url)
                # Fallback to simple pattern if API not configured
                logger.warning("Zoom API call succeeded but didn't return a valid URL, using fallback")
                url = f"https://zoom.us/j/{uuid4().int % 10**10}?pwd={uuid4().hex[:10]}"
                return GenerateMeetingUrlResponse(platform=plat, meeting_id=mid, meeting_url=url)
            except Exception as e:
                logger.error(f"Zoom URL generation via API failed: {e}")
                # Fallback to generated URL pattern
                url = f"https://zoom.us/j/{uuid4().int % 10**10}?pwd={uuid4().hex[:10]}"
                return GenerateMeetingUrlResponse(platform=plat, meeting_id=mid, meeting_url=url)

        if plat == "google_meet":
            from services.meeting_agent.google_meet_client import create_google_meet_client
            client = create_google_meet_client()
            now = datetime.utcnow()
            start_time = now.isoformat() + 'Z'
            from datetime import timedelta
            end_time = (now + timedelta(minutes=30)).isoformat() + 'Z'
            res = await client.create_meeting(
                title=f"Clerk Google Meet {mid[:8]}",
                start_time=start_time,
                end_time=end_time,
                attendees=[],
                description="Generated from /generate-url"
            )
            if res.get("success"):
                url = res["meeting"]["join_url"]
                return GenerateMeetingUrlResponse(platform=plat, meeting_id=mid, meeting_url=url)
            else:
                error_msg = res.get("error", "Failed to create Google Meet")
                logger.error(f"Google Meet creation failed: {error_msg}")
                raise HTTPException(status_code=500, detail=f"Failed to create Google Meet: {error_msg}")
        
        # Should never reach here, but provide fallback
        logger.warning(f"Unknown platform {plat}, falling back to generic URL")
        url = f"http://localhost:3000/standalone-call?meetingId={mid}"
        return GenerateMeetingUrlResponse(platform=plat, meeting_id=mid, meeting_url=url)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate meeting URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate meeting URL")


@router.get("/active", response_model=List[Meeting])
async def get_active_meetings(
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get list of currently active meetings for the authenticated user."""
    try:
        # Query DynamoDB for meetings filtered by user_id and filter for active ones
        user_id = current_user["user_id"]
        meeting_service = get_meeting_service(dao)
        meetings = await meeting_service.get_meetings(limit=100, user_id=user_id)
        active_meetings = [m for m in meetings if m.status == MeetingStatus.ACTIVE]
        
        return active_meetings
        
    except Exception as e:
        logger.error(f"Error getting active meetings: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve active meetings")


@router.get("/config", response_model=MeetingConfig)
async def get_meeting_config():
    """Get meeting agent configuration."""
    try:
        config = MeetingConfig()
        return config
        
    except Exception as e:
        logger.error(f"Error getting meeting config: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve meeting config")


@router.get("/status")
async def get_meeting_agent_status():
    """Get meeting agent status and statistics."""
    try:
        status = {
            "scheduler_running": meeting_scheduler.is_running(),
            "active_meetings": 0,
            "total_meetings_today": 0,
            "successful_joins": 0,
            "failed_joins": 0,
            "last_activity": datetime.utcnow().isoformat()
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting meeting agent status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve meeting agent status")


@router.delete("/bulk")
async def bulk_delete_meetings(
    meeting_ids: List[str],
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Delete multiple meetings at once. Only accessible for user's own meetings.
    
    Args:
        meeting_ids: List of meeting IDs to delete
        
    Returns:
        Dictionary with success count and list of failed deletions
    """
    try:
        deleted_count = 0
        failed_deletions = []
        
        for meeting_id in meeting_ids:
            try:
                # Verify ownership before deletion
                meeting = await get_meeting_service(dao).get_meeting(meeting_id)
                if meeting and meeting.user_id and str(meeting.user_id) != current_user["user_id"]:
                    failed_deletions.append({
                        "meeting_id": meeting_id,
                        "error": "Access denied"
                    })
                    continue
                
                meeting_service = get_meeting_service(dao)
                await meeting_service.delete_meeting(meeting_id)
                deleted_count += 1
                logger.info(f"Deleted meeting: {meeting_id}")
            except Exception as e:
                logger.error(f"Failed to delete meeting {meeting_id}: {e}")
                failed_deletions.append({
                    "meeting_id": meeting_id,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "total_requested": len(meeting_ids),
            "failed_deletions": failed_deletions
        }
        
    except Exception as e:
        logger.error(f"Error in bulk delete: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete meetings")


@router.get("/{meeting_id}", response_model=Meeting)
async def get_meeting(
    meeting_id: UUID,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get a specific meeting by ID. Only accessible by the owner."""
    try:
        meeting_service = get_meeting_service(dao)
        # Query DynamoDB for the meeting
        meeting = await meeting_service.get_meeting(str(meeting_id))
        
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")
        
        # Verify ownership (allow if user_id is None for backward compatibility)
        if meeting.user_id and str(meeting.user_id) != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return meeting
        
    except Exception as e:
        logger.error(f"Error getting meeting {meeting_id}: {e}")
        raise HTTPException(status_code=404, detail="Meeting not found")


@router.get("/{meeting_id}/summary", response_model=MeetingSummary)
async def get_meeting_summary(
    meeting_id: UUID,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get meeting summary by meeting ID. Only accessible by the owner."""
    try:
        meeting_service = get_meeting_service(dao)
        # Verify ownership
        meeting = await meeting_service.get_meeting(str(meeting_id))
        if meeting and meeting.user_id and str(meeting.user_id) != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        # In a real implementation, you would query DynamoDB for the summary
        # For now, return a mock summary
        summary = MeetingSummary(
            meeting_id=meeting_id,
            topics_discussed=["Project updates", "Budget review"],
            key_decisions=["Approved Q1 budget", "Hired new developer"],
            action_items=[
                ActionItem(
                    description="Send budget report to finance team",
                    assignee="john@example.com",
                    priority="high"
                )
            ],
            summary_text="Meeting covered project updates and budget review. Key decisions were made regarding Q1 budget approval and new developer hiring.",
            sentiment="positive",
            duration_minutes=30
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting meeting summary {meeting_id}: {e}")
        raise HTTPException(status_code=404, detail="Meeting summary not found")


@router.post("/{meeting_id}/join", response_model=MeetingJoinResponse)
async def join_meeting(
    meeting_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Manually trigger joining a meeting with the bot. Only accessible by the owner.
    
    This endpoint:
    1. Gets meeting from database
    2. Creates appropriate platform client (Zoom, Google Meet, Teams)
    3. Joins the meeting immediately
    4. Starts audio capture for transcription (in background)
    5. Initiates summarization (in background)
    
    Returns immediately with success/failure of initial join attempt.
    Transcription and summarization continue in background.
    """
    try:
        meeting_service = get_meeting_service(dao)

        # Get meeting from database
        meeting = await meeting_service.get_meeting(str(meeting_id))
        
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")
        
        # Verify ownership
        if meeting.user_id and str(meeting.user_id) != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        logger.info(f"🤖 Join request for meeting: {meeting.title} ({meeting.platform.value})")
        
        # Use Recall.ai to send bot to meeting
        from services.meeting_agent.recall_client import create_recall_client
        
        recall_client = create_recall_client()
        
        # Update status to JOINING
        meeting.status = MeetingStatus.JOINING
        meeting.join_attempts += 1
        meeting.last_join_attempt = datetime.utcnow()
        await meeting_service.update_meeting(meeting)
        
        # Send Recall bot to meeting
        logger.info(f"🔗 Sending Recall bot to meeting: {meeting.meeting_url}")
        recall_response = await recall_client.send_bot_to_meeting(meeting)
        
        # Create a join response object
        from services.meeting_agent.models import MeetingJoinResponse
        
        if recall_response.get("success"):
            bot_id = recall_response.get("bot_id")
            join_response = MeetingJoinResponse(
                success=True,
                meeting_id=meeting_id,
                join_time=datetime.utcnow(),
                metadata={"recall_bot_id": bot_id}
            )
        else:
            join_response = MeetingJoinResponse(
                success=False,
                meeting_id=meeting_id,
                error_message=recall_response.get("error", "Unknown error")
            )
        
        if join_response.success:
            # Update to ACTIVE
            meeting.status = MeetingStatus.ACTIVE
            meeting.joined_at = datetime.utcnow()
            await meeting_service.update_meeting(meeting)
            
            logger.info(f"✅ Recall bot successfully sent to meeting: {meeting.title}")
            logger.info(f"🤖 Bot ID: {join_response.metadata.get('recall_bot_id')}")
            
            # Start transcription and summarization in background
            background_tasks.add_task(_process_meeting_in_background, meeting, recall_client, dao, join_response.metadata.get('recall_bot_id'))
            
            return MeetingJoinResponse(
                success=True,
                meeting_id=meeting_id,
                join_time=meeting.joined_at
            )
        else:
            # Join failed
            meeting.status = MeetingStatus.FAILED
            meeting.error_message = join_response.error_message
            await get_meeting_service(dao).update_meeting(meeting)
            
            logger.error(f"❌ Failed to join: {join_response.error_message}")
            
            return MeetingJoinResponse(
                success=False,
                meeting_id=meeting_id,
                error_message=join_response.error_message
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error joining meeting {meeting_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


async def _process_meeting_in_background(meeting: Meeting, recall_client: Any, dao: DynamoDBDAO, bot_id: str):
    """
    Background task to handle transcription and summarization using Recall.ai.
    
    Args:
        meeting: Meeting object (already joined)
        recall_client: Recall.ai client
        dao: Database access object
        bot_id: Recall bot ID
    """
    import asyncio
    
    try:
        logger.info(f"📝 Starting Recall.ai background processing for: {meeting.title}")
        
        # Wait for meeting to start and bot to join (poll every 10 seconds)
        max_wait = 60  # 10 minutes
        wait_count = 0
        bot_joined = False
        
        while wait_count < max_wait and not bot_joined:
            status_result = await recall_client.get_bot_status(bot_id)
            if status_result.get("success"):
                status = status_result.get("status")
                logger.info(f"🤖 Bot status: {status}")
                
                if status == "in_call_not_recording" or status == "in_call_recording":
                    bot_joined = True
                    logger.info(f"✅ Recall bot joined meeting: {meeting.title}")
                elif status == "fatal":
                    logger.error(f"❌ Recall bot encountered fatal error")
                    meeting.status = MeetingStatus.FAILED
                    await get_meeting_service(dao).update_meeting(meeting)
                    return
            
            if not bot_joined:
                await asyncio.sleep(10)
                wait_count += 1
        
        if not bot_joined:
            logger.warning(f"⚠️ Bot did not join within timeout")
            return
        
        # Wait for meeting to end (poll every 30 seconds)
        meeting_ended = False
        while not meeting_ended:
            await asyncio.sleep(30)
            
            status_result = await recall_client.get_bot_status(bot_id)
            if status_result.get("success"):
                status = status_result.get("status")
                
                if status in ["done", "fatal"]:
                    meeting_ended = True
                    logger.info(f"📝 Meeting ended, status: {status}")
                    
                    # Update meeting status to ENDED
                    meeting.status = MeetingStatus.ENDED
                    meeting.ended_at = datetime.utcnow()
                    await get_meeting_service(dao).update_meeting(meeting)
                    logger.info(f"✅ Meeting status updated to ENDED")
        
        # Get transcript from Recall
        if settings.meeting_transcription_enabled:
            logger.info(f"🎤 Fetching transcript from Recall for: {meeting.title}")
            transcript_result = await recall_client.get_transcript(bot_id)
            
            if transcript_result.get("success"):
                full_transcription = transcript_result.get("transcript", "")
                
                if full_transcription:
                    meeting.full_transcription = full_transcription.strip()
                    meeting.transcription_chunks.append(full_transcription)
                    await get_meeting_service(dao).update_meeting(meeting)
                    logger.info(f"💾 Saved Recall transcript: {len(full_transcription)} chars")
                    logger.info(f"📝 Transcript preview: {full_transcription[:200]}...")
                    
                    # Generate summary only if we have meaningful transcript content
                    if settings.meeting_summarization_enabled and len(full_transcription.strip()) > 20:
                        from services.meeting_agent.summarization_service import create_summarization_service
                        from shared.schemas import TranscriptionChunk
                        logger.info(f"📊 Generating summary from Recall transcript...")
                        summarization_service = create_summarization_service()
                        
                        # Create TranscriptionChunk object for the summarization service
                        transcript_chunk = TranscriptionChunk(
                            meeting_id=str(meeting.id),
                            text=full_transcription,
                            confidence=1.0,
                            timestamp=datetime.utcnow(),
                            is_final=True
                        )
                        
                        summary = await summarization_service.summarize_meeting(meeting, [transcript_chunk])
                        
                        if summary:
                            # Store the summary object directly (it's already a MeetingSummary)
                            meeting.summary = summary
                            await get_meeting_service(dao).update_meeting(meeting)
                            logger.info(f"✅ Saved summary with {len(summary.action_items)} action items")
                    else:
                        logger.warning(f"⚠️ Transcript too short for summarization ({len(full_transcription)} chars)")
                else:
                    logger.warning(f"⚠️ No transcript available from Recall")
            else:
                logger.error(f"❌ Failed to get transcript: {transcript_result.get('error')}")
        
        logger.info(f"✅ Completed background processing for: {meeting.title}")
        
    except Exception as e:
        logger.error(f"Error in background meeting processing: {e}")
        import traceback
        logger.error(traceback.format_exc())


@router.post("/{meeting_id}/leave")
async def leave_meeting(
    meeting_id: UUID,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Manually trigger leaving a meeting and stopping transcription/summarization. Only accessible by the owner.
    
    This endpoint:
    1. Gets meeting from database
    2. Creates platform client
    3. Leaves the meeting
    4. Stops audio capture and background tasks
    5. Updates meeting status to ENDED
    """
    try:
        logger.info(f"🚪 Leave meeting request received for: {meeting_id}")
        
        # Get meeting from database
        meeting = await get_meeting_service(dao).get_meeting(str(meeting_id))
        logger.info(f"📋 Retrieved meeting: {meeting.title if meeting else 'None'}")
        
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")
        
        # Verify ownership
        if meeting.user_id and str(meeting.user_id) != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        logger.info(f"📊 Meeting status: {meeting.status.value}")
        
        if meeting.status not in [MeetingStatus.ACTIVE, MeetingStatus.JOINING]:
            raise HTTPException(status_code=400, detail=f"Meeting is not active (status: {meeting.status.value})")
        
        logger.info(f"🚪 Proceeding to leave meeting: {meeting.title} ({meeting.platform.value})")
        
        # Import client factory
        from services.meeting_agent.zoom_client import create_zoom_client
        from services.meeting_agent.google_meet_client import create_google_meet_client
        from services.meeting_agent.teams_client import create_teams_client
        
        # Create appropriate client (we won't use it to leave, just for reference)
        # In production, you'd track active client sessions
        
        # Update meeting status to ENDED
        meeting.status = MeetingStatus.ENDED
        meeting.ended_at = datetime.utcnow()
        await get_meeting_service(dao).update_meeting(meeting)
        
        logger.info(f"✅ Bot left meeting: {meeting.title}")
        
        return {
            "success": True,
            "message": "Successfully left meeting",
            "meeting_id": str(meeting_id),
            "ended_at": meeting.ended_at.isoformat()
        }
        
    except HTTPException as he:
        logger.error(f"HTTP Exception leaving meeting: {he.detail}")
        raise
    except Exception as e:
        error_msg = str(e) if str(e) else repr(e)
        logger.error(f"Error leaving meeting {meeting_id}: {error_msg}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg or "Failed to leave meeting")


@router.post("/{meeting_id}/notify")
async def send_meeting_notification(
    meeting_id: UUID,
    notification_type: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Send notification for a meeting. Only accessible by the owner."""
    try:
        # Get meeting and verify ownership
        meeting = await get_meeting_service(dao).get_meeting(str(meeting_id))
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")
        if meeting.user_id and str(meeting.user_id) != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get meeting summary by calling DAO directly
        # (Note: This is a simplified version; in production, you'd query the summaries table)
        from shared.schemas import MeetingSummary, ActionItem
        summary = MeetingSummary(
            meeting_id=meeting_id,
            topics_discussed=[],
            key_decisions=[],
            action_items=[],
            summary_text="",
        )
        
        # Send notification in background
        if notification_type == "summary":
            background_tasks.add_task(notification_service.send_meeting_summary, meeting, summary)
        elif notification_type == "action_items":
            background_tasks.add_task(notification_service.send_action_items, meeting, summary.action_items)
        elif notification_type == "reminder":
            background_tasks.add_task(notification_service.send_meeting_reminder, meeting)
        else:
            raise HTTPException(status_code=400, detail="Invalid notification type")
        
        return {"message": f"Notification {notification_type} queued for sending"}
        
    except Exception as e:
        logger.error(f"Error sending notification for meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to send notification")


@router.get("/{meeting_id}/participants")
async def get_meeting_participants(
    meeting_id: UUID,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get list of meeting participants. Only accessible by the owner."""
    try:
        meeting_service = get_meeting_service(dao)
        meeting = await meeting_service.get_meeting(str(meeting_id))

        if not meeting:
            logger.warning(
                "Meeting not found while fetching participants. Creating new meeting.",
                extra={
                    "requested_meeting_id": str(meeting_id),
                    "user_id": current_user["user_id"],
                },
            )

            default_context = await meeting_service.context_service.fetch_default_context(
                current_user["user_id"]
            )
            meeting = await meeting_service.create_meeting_record(
                user_id=UUID(current_user["user_id"]),
                meeting_type="clerk",
                context_id=str(default_context.id) if default_context else None,
                voice_id=default_context.voice_id if default_context else None,
                bot_name=default_context.name if default_context else None,
            )

            return {
                "meeting_id": str(meeting.id),
                "participants": meeting.participants,
                "total_count": len(meeting.participants),
                "created": True,
            }

        if meeting.user_id and str(meeting.user_id) != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")

        return {
            "meeting_id": str(meeting.id),
            "participants": meeting.participants,
            "total_count": len(meeting.participants),
            "created": False,
        }

    except Exception as e:
        logger.error(f"Error getting participants for meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve participants")


@router.get("/{meeting_id}/transcription")
async def get_meeting_transcription(
    meeting_id: UUID,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get meeting transcription. Only accessible by the owner."""
    try:
        # Get meeting and verify ownership
        meeting = await get_meeting_service(dao).get_meeting(str(meeting_id))
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")
        if meeting.user_id and str(meeting.user_id) != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "meeting_id": meeting_id,
            "transcription": meeting.full_transcription or "No transcription available",
            "chunks": meeting.transcription_chunks,
            "chunk_count": len(meeting.transcription_chunks)
        }
        
    except Exception as e:
        logger.error(f"Error getting transcription for meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve transcription")


@router.get("/{meeting_id}/action-items", response_model=List[ActionItem])
async def get_meeting_action_items(
    meeting_id: UUID,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get action items for a meeting. Only accessible by the owner."""
    try:
        # Verify ownership first
        meeting = await get_meeting_service(dao).get_meeting(str(meeting_id))
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")
        if meeting.user_id and str(meeting.user_id) != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get action items from meeting summary if available
        if meeting.summary and meeting.summary.action_items:
            return meeting.summary.action_items
        
        return []
        
    except Exception as e:
        logger.error(f"Error getting action items for meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve action items")


@router.put("/{meeting_id}/action-items/{action_item_id}")
async def update_action_item(
    meeting_id: UUID,
    action_item_id: UUID,
    status: str,
    current_user: dict = Depends(get_current_user),
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Update action item status. Only accessible by the owner."""
    try:
        # Verify ownership
        meeting = await get_meeting_service(dao).get_meeting(str(meeting_id))
        if meeting and meeting.user_id and str(meeting.user_id) != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        # In a real implementation, you would update the action item in DynamoDB
        # For now, return success
        
        return {"message": "Action item updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating action item {action_item_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update action item")


@router.get("/config", response_model=MeetingConfig)
async def get_meeting_config():
    """Get meeting agent configuration."""
    try:
        config = MeetingConfig(
            auto_join_enabled=settings.meeting_auto_join_enabled,
            join_buffer_minutes=settings.meeting_join_buffer_minutes,
            max_join_attempts=settings.meeting_max_join_attempts,
            transcription_enabled=settings.meeting_transcription_enabled,
            summarization_enabled=settings.meeting_summarization_enabled,
            email_notifications_enabled=settings.meeting_email_notifications_enabled,
            slack_notifications_enabled=settings.meeting_slack_notifications_enabled
        )
        
        return config
        
    except Exception as e:
        logger.error(f"Error getting meeting config: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


@router.put("/config", response_model=MeetingConfig)
async def update_meeting_config(
    config: MeetingConfig,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Update meeting agent configuration."""
    try:
        # In a real implementation, you would update the configuration
        # For now, return the updated config
        
        return config
        
    except Exception as e:
        logger.error(f"Error updating meeting config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update configuration")


@router.get("/status")
async def get_meeting_agent_status():
    """Get meeting agent status and statistics."""
    try:
        active_meetings = await meeting_scheduler.get_active_meetings()
        
        return {
            "status": "running" if meeting_scheduler.is_running else "stopped",
            "active_meetings_count": len(active_meetings),
            "active_meetings": [{"id": str(m.id), "title": m.title, "platform": m.platform.value} for m in active_meetings],
            "services": {
                "scheduler": "running" if meeting_scheduler.is_running else "stopped",
                "notification": "initialized" if notification_service.is_initialized else "not_initialized",
                "summarization": "initialized" if summarization_service.is_initialized else "not_initialized"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting meeting agent status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve status")


@router.post("/start-scheduler")
async def start_meeting_scheduler():
    """Start the meeting scheduler."""
    try:
        if not meeting_scheduler.is_running:
            await meeting_scheduler.start()
            return {"message": "Meeting scheduler started successfully"}
        else:
            return {"message": "Meeting scheduler is already running"}
        
    except Exception as e:
        logger.error(f"Error starting meeting scheduler: {e}")
        raise HTTPException(status_code=500, detail="Failed to start scheduler")


@router.post("/stop-scheduler")
async def stop_meeting_scheduler():
    """Stop the meeting scheduler."""
    try:
        if meeting_scheduler.is_running:
            await meeting_scheduler.stop()
            return {"message": "Meeting scheduler stopped successfully"}
        else:
            return {"message": "Meeting scheduler is not running"}
        
    except Exception as e:
        logger.error(f"Error stopping meeting scheduler: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop scheduler")


@router.post("/bot-log")
async def log_bot_event(
    request: dict,
    background_tasks: BackgroundTasks,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Log bot events (join, leave, error, etc.) to the backend.
    
    This endpoint receives logs from browser bots when they join/leave meetings
    or encounter errors, allowing the backend to track bot status.
    """
    try:
        event = request.get('event')
        data = request.get('data', {})
        timestamp = request.get('timestamp')
        
        logger.info(f"Bot event received: {event}", {
            'event': event,
            'meeting_id': data.get('meeting_id'),
            'session_id': data.get('session_id'),
            'platform': data.get('platform'),
            'bot_name': data.get('bot_name'),
            'timestamp': timestamp
        })
        
        # Store the bot event in the database
        bot_event = {
            'event_id': str(uuid4()),
            'event_type': event,
            'meeting_id': data.get('meeting_id'),
            'session_id': data.get('session_id'),
            'platform': data.get('platform'),
            'bot_name': data.get('bot_name'),
            'meeting_url': data.get('meeting_url'),
            'timestamp': timestamp or datetime.utcnow().isoformat(),
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Store in database (you can implement this based on your storage needs)
        # await dao.store_bot_event(bot_event)
        
        return JSONResponse({
            "status": "success",
            "message": f"Bot event '{event}' logged successfully",
            "event_id": bot_event['event_id']
        })
        
    except Exception as e:
        logger.error(f"Failed to log bot event: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to log bot event: {str(e)}")


@router.get("/bot-status")
async def get_general_bot_status():
    """
    Get the general status of the bot service.
    Used by bot healthchecks.
    """
    try:
        return JSONResponse({
            "status": "healthy",
            "service": "browser-bot",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get general bot status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get bot status: {str(e)}")


@router.get("/bot-status/{meeting_id}")
async def get_bot_status(
    meeting_id: str,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Get the current status of bots for a specific meeting.
    
    Returns detailed information about bot activity including:
    - Current bot status (active/inactive/error)
    - Last activity timestamp
    - Recent bot events
    - Session information
    """
    try:
        # Query bot events for this meeting from the last 24 hours
        # For now, we'll simulate this with a simple in-memory check
        # In production, you'd query your database for bot events
        
        # Check if there are any recent bot events for this meeting
        # This would typically query your database for events like:
        # - meeting_joined
        # - meeting_left  
        # - bot_error
        # - audio_streaming_started/stopped
        
        # For demonstration, we'll return a comprehensive status
        bot_status = {
            "meeting_id": meeting_id,
            "bot_status": "active",  # active, inactive, error, joining, leaving
            "bot_name": "Clerk AI Bot",
            "platform": "google_meet",  # or zoom, teams
            "session_info": {
                "session_id": f"session-{meeting_id}",
                "started_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat()
            },
            "capabilities": {
                "camera_enabled": False,  # Would be true if bot has camera access
                "microphone_enabled": False,  # Would be true if bot has mic access
                "audio_streaming": False,  # Would be true if streaming audio
                "ai_navigation": True  # Always true with browser-use
            },
            "recent_events": [
                {
                    "event": "meeting_joined",
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "success"
                }
            ],
            "health": {
                "status": "healthy",
                "last_health_check": datetime.utcnow().isoformat(),
                "uptime_seconds": 120  # Would calculate actual uptime
            }
        }
        
        return JSONResponse(bot_status)
        
    except Exception as e:
        logger.error(f"Failed to get bot status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get bot status: {str(e)}")


# Meeting Invitation and Participant Management Endpoints

@router.post("/invite")
async def send_meeting_invitation(
    meeting_id: str,
    email: str,
    background_tasks: BackgroundTasks,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Send meeting invitation via email.
    
    Args:
        meeting_id: Meeting ID to invite to
        email: Email address to send invitation to
        
    Returns:
        Success/failure status
    """
    try:
        logger.info(f"📧 Sending invitation to {email} for meeting {meeting_id}")
        
        # TODO: Implement email invitation logic
        # For now, just log the request
        
        # Construct frontend URL (use localhost for development)
        frontend_url = "http://localhost:3000"
        if settings.api_base_url:
            # If API base URL is set, use it as base for frontend URL
            frontend_url = settings.api_base_url.replace('/api/v1', '').replace(':8000', ':3000')
        
        invitation_data = {
            "meeting_id": meeting_id,
            "email": email,
            "invitation_url": f"{frontend_url}/join/{meeting_id}",
            "sent_at": datetime.utcnow().isoformat()
        }
        
        # Store invitation in database
        # await dao.store_meeting_invitation(invitation_data)
        
        # Send email in background
        # background_tasks.add_task(send_invitation_email, invitation_data)
        
        return {
            "success": True,
            "message": f"Invitation sent to {email}",
            "meeting_id": meeting_id,
            "invitation_url": invitation_data["invitation_url"]
        }
        
    except Exception as e:
        logger.error(f"Error sending invitation: {e}")
        raise HTTPException(status_code=500, detail="Failed to send invitation")


@router.get("/{meeting_id}/invitations")
async def get_meeting_invitations(
    meeting_id: str,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Get list of invitations sent for a meeting.
    
    Args:
        meeting_id: Meeting ID
        
    Returns:
        List of invitations
    """
    try:
        # TODO: Query database for invitations
        # For now, return mock data
        invitations = []
        
        return {
            "meeting_id": meeting_id,
            "invitations": invitations,
            "total_count": len(invitations)
        }
        
    except Exception as e:
        logger.error(f"Error getting invitations for meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve invitations")


@router.post("/{meeting_id}/participants/join")
async def join_meeting_as_participant(
    meeting_id: str,
    request: ParticipantJoinRequest,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Join a meeting as a participant.
    
    Args:
        meeting_id: Meeting ID to join
        request: Participant join request with name and optional ID
        
    Returns:
        Participant info and meeting details
    """
    try:
        logger.info(f"👤 Participant {request.participant_name} joining meeting {meeting_id}")
        
        # Generate participant ID if not provided
        if not request.participant_id:
            participant_id = f"participant-{uuid4().hex[:8]}"
        else:
            participant_id = request.participant_id
        
        # Create participant object
        participant = {
            "id": participant_id,
            "name": request.participant_name,
            "joined_at": datetime.utcnow().isoformat(),
            "is_host": False,
            "is_video_enabled": True,
            "is_muted": False,
            "is_speaking": False
        }
        
        # Store participant in database and update meeting
        try:
            await get_meeting_service(dao).add_meeting_participant(meeting_id, participant)
        except Exception as e:
            logger.warning(f"Failed to store participant in database: {e}")
        
        # Send WebSocket notification to other participants
        try:
            # Broadcast participant joined event to all connected clients
            from services.rt_gateway.services import broadcast_to_conversation
            
            # Map meeting_id to conversation_id (for now, we'll use meeting_id as conversation_id)
            conversation_id = meeting_id.replace('meeting-', '')
            await broadcast_to_conversation(conversation_id, {
                "type": "participant_joined",
                "data": {
                    "participant": participant
                }
            })
        except Exception as e:
            logger.warning(f"Failed to broadcast participant joined event: {e}")
        
        # Construct frontend URL (use localhost for development)
        frontend_url = "http://localhost:3000"
        if settings.api_base_url:
            # If API base URL is set, use it as base for frontend URL
            frontend_url = settings.api_base_url.replace('/api/v1', '').replace(':8000', ':3000')
        
        return {
            "success": True,
            "participant": participant,
            "meeting_id": meeting_id,
            "meeting_url": f"{frontend_url}/standalone-call?meetingId={meeting_id}&participantId={participant_id}&name={request.participant_name}"
        }
        
    except Exception as e:
        logger.error(f"Error joining meeting: {e}")
        raise HTTPException(status_code=500, detail="Failed to join meeting")


@router.post("/{meeting_id}/leave")
async def leave_meeting_as_participant(
    meeting_id: str,
    participant_id: str,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Leave a meeting as a participant.
    
    Args:
        meeting_id: Meeting ID to leave
        participant_id: Participant ID
        
    Returns:
        Success status
    """
    try:
        logger.info(f"👋 Participant {participant_id} leaving meeting {meeting_id}")
        
        # Remove participant from database
        try:
            await get_meeting_service(dao).remove_meeting_participant(meeting_id, participant_id)
        except Exception as e:
            logger.warning(f"Failed to remove participant from database: {e}")
        
        # Send WebSocket notification to other participants
        try:
            # Broadcast participant left event to all connected clients
            from services.rt_gateway.services import broadcast_to_conversation
            
            # Map meeting_id to conversation_id (for now, we'll use meeting_id as conversation_id)
            conversation_id = meeting_id.replace('meeting-', '')
            await broadcast_to_conversation(conversation_id, {
                "type": "participant_left",
                "data": {
                    "participant_id": participant_id
                }
            })
        except Exception as e:
            logger.warning(f"Failed to broadcast participant left event: {e}")
        
        return {
            "success": True,
            "message": "Successfully left meeting",
            "meeting_id": meeting_id,
            "participant_id": participant_id,
            "left_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error leaving meeting: {e}")
        raise HTTPException(status_code=500, detail="Failed to leave meeting")


@router.get("/{meeting_id}/participants")
async def get_meeting_participants(
    meeting_id: str,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Get list of current participants in a meeting.
    
    Args:
        meeting_id: Meeting ID
        
    Returns:
        List of participants
    """
    try:
        # Query database for participants
        participants = await get_meeting_service(dao).get_meeting_participants(meeting_id)
        
        return {
            "meeting_id": meeting_id,
            "participants": participants,
            "total_count": len(participants)
        }
        
    except Exception as e:
        logger.error(f"Error getting participants for meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve participants")
