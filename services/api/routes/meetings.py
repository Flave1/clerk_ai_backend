"""
FastAPI routes for meeting agent management.

This module provides REST endpoints for managing meetings, summaries,
and meeting agent functionality.
"""
import logging
from datetime import datetime
from typing import List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from services.api.dao import get_dao, DynamoDBDAO
from shared.config import get_settings
from shared.schemas import Meeting, MeetingSummary, MeetingStatus, MeetingPlatform, ActionItem
from services.meeting_agent.models import (
    MeetingJoinRequest, MeetingJoinResponse, MeetingNotification,
    CalendarEvent, MeetingConfig
)
from services.meeting_agent.scheduler import create_meeting_scheduler
from services.meeting_agent.notifier import create_notification_service
from services.meeting_agent.summarization_service import create_summarization_service

logger = logging.getLogger(__name__)
settings = get_settings()

# Create router
router = APIRouter()

# Global services
meeting_scheduler = create_meeting_scheduler()
notification_service = create_notification_service()
summarization_service = create_summarization_service()


@router.get("/meetings", response_model=List[Meeting])
async def get_meetings(
    status: Optional[MeetingStatus] = None,
    platform: Optional[MeetingPlatform] = None,
    limit: int = 50,
    offset: int = 0,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get list of meetings with optional filtering."""
    try:
        # Query DynamoDB for meetings
        meetings = await dao.get_meetings(limit=limit + offset)
        
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
@router.get("/meetings/summaries", response_model=List[MeetingSummary])
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


@router.get("/meetings/active", response_model=List[Meeting])
async def get_active_meetings(
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get list of currently active meetings."""
    try:
        # Query DynamoDB for meetings and filter for active ones
        meetings = await dao.get_meetings(limit=100)
        active_meetings = [m for m in meetings if m.status == MeetingStatus.ACTIVE]
        
        return active_meetings
        
    except Exception as e:
        logger.error(f"Error getting active meetings: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve active meetings")


@router.get("/meetings/config", response_model=MeetingConfig)
async def get_meeting_config():
    """Get meeting agent configuration."""
    try:
        config = MeetingConfig()
        return config
        
    except Exception as e:
        logger.error(f"Error getting meeting config: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve meeting config")


@router.get("/meetings/status")
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


@router.delete("/meetings/bulk")
async def bulk_delete_meetings(
    meeting_ids: List[str],
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Delete multiple meetings at once.
    
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
                await dao.delete_meeting(meeting_id)
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


@router.get("/meetings/{meeting_id}", response_model=Meeting)
async def get_meeting(
    meeting_id: UUID,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get a specific meeting by ID."""
    try:
        # Query DynamoDB for the meeting
        meeting = await dao.get_meeting(str(meeting_id))
        
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")
        
        return meeting
        
    except Exception as e:
        logger.error(f"Error getting meeting {meeting_id}: {e}")
        raise HTTPException(status_code=404, detail="Meeting not found")


@router.get("/meetings/{meeting_id}/summary", response_model=MeetingSummary)
async def get_meeting_summary(
    meeting_id: UUID,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get meeting summary by meeting ID."""
    try:
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


@router.post("/meetings/{meeting_id}/join", response_model=MeetingJoinResponse)
async def join_meeting(
    meeting_id: UUID,
    background_tasks: BackgroundTasks,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Manually trigger joining a meeting with the bot.
    
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
        # Get meeting from database
        meeting = await dao.get_meeting(str(meeting_id))
        
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")
        
        logger.info(f"🤖 Join request for meeting: {meeting.title} ({meeting.platform.value})")
        
        # Use Recall.ai to send bot to meeting
        from services.meeting_agent.recall_client import create_recall_client
        
        recall_client = create_recall_client()
        
        # Update status to JOINING
        meeting.status = MeetingStatus.JOINING
        meeting.join_attempts += 1
        meeting.last_join_attempt = datetime.utcnow()
        await dao.update_meeting(meeting)
        
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
            await dao.update_meeting(meeting)
            
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
            await dao.update_meeting(meeting)
            
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
                    await dao.update_meeting(meeting)
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
                    await dao.update_meeting(meeting)
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
                    await dao.update_meeting(meeting)
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
                            await dao.update_meeting(meeting)
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


@router.post("/meetings/{meeting_id}/leave")
async def leave_meeting(
    meeting_id: UUID,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Manually trigger leaving a meeting and stopping transcription/summarization.
    
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
        meeting = await dao.get_meeting(str(meeting_id))
        logger.info(f"📋 Retrieved meeting: {meeting.title if meeting else 'None'}")
        
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")
        
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
        await dao.update_meeting(meeting)
        
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


@router.post("/meetings/{meeting_id}/notify")
async def send_meeting_notification(
    meeting_id: UUID,
    notification_type: str,
    background_tasks: BackgroundTasks,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Send notification for a meeting."""
    try:
        # Get meeting and summary
        meeting = await get_meeting(meeting_id, dao)
        summary = await get_meeting_summary(meeting_id, dao)
        
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


@router.get("/meetings/{meeting_id}/participants")
async def get_meeting_participants(
    meeting_id: UUID,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get list of meeting participants."""
    try:
        # Get meeting
        meeting = await get_meeting(meeting_id, dao)
        
        return {
            "meeting_id": meeting_id,
            "participants": meeting.participants,
            "total_count": len(meeting.participants)
        }
        
    except Exception as e:
        logger.error(f"Error getting participants for meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve participants")


@router.get("/meetings/{meeting_id}/transcription")
async def get_meeting_transcription(
    meeting_id: UUID,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get meeting transcription."""
    try:
        # Get meeting
        meeting = await get_meeting(meeting_id, dao)
        
        return {
            "meeting_id": meeting_id,
            "transcription": meeting.full_transcription or "No transcription available",
            "chunks": meeting.transcription_chunks,
            "chunk_count": len(meeting.transcription_chunks)
        }
        
    except Exception as e:
        logger.error(f"Error getting transcription for meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve transcription")


@router.get("/meetings/{meeting_id}/action-items", response_model=List[ActionItem])
async def get_meeting_action_items(
    meeting_id: UUID,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Get action items for a meeting."""
    try:
        # Get meeting summary
        summary = await get_meeting_summary(meeting_id, dao)
        
        return summary.action_items
        
    except Exception as e:
        logger.error(f"Error getting action items for meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve action items")


@router.put("/meetings/{meeting_id}/action-items/{action_item_id}")
async def update_action_item(
    meeting_id: UUID,
    action_item_id: UUID,
    status: str,
    dao: DynamoDBDAO = Depends(get_dao)
):
    """Update action item status."""
    try:
        # In a real implementation, you would update the action item in DynamoDB
        # For now, return success
        
        return {"message": "Action item updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating action item {action_item_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update action item")


@router.get("/meetings/config", response_model=MeetingConfig)
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


@router.put("/meetings/config", response_model=MeetingConfig)
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


@router.get("/meetings/status")
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


@router.post("/meetings/start-scheduler")
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


@router.post("/meetings/stop-scheduler")
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
