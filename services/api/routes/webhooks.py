"""
Webhooks API routes for testing and integration.
"""
import logging
import subprocess
import os
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4, UUID

from fastapi import APIRouter, HTTPException, status, Depends, Header
from pydantic import BaseModel

from shared.schemas import JoinMeetingRequest, Meeting, MeetingPlatform, MeetingStatus
from services.api.dao import get_dao, DynamoDBDAO
from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


class JoinMeetingResponse(BaseModel):
    """Join meeting webhook response."""
    
    success: bool
    message: str
    status: str
    timestamp: str
    meeting_id: Optional[str] = None
    meeting_url: Optional[str] = None
    platform: Optional[str] = None
    voice_id: Optional[str] = None
    capabilities: Optional[dict] = None


async def _create_platform_meeting_url(platform: str, meeting_id: str) -> str:
    """
    Create a meeting URL using platform-specific clients.
    
    Returns the join URL for the meeting.
    """
    platform_lower = platform.lower()
    now = datetime.utcnow()
    start_time = now.isoformat() + 'Z'
    end_time = (now + timedelta(minutes=30)).isoformat() + 'Z'
    
    if platform_lower == "zoom":
        try:
            from services.meeting_agent.zoom_client import create_zoom_client
            client = create_zoom_client()
            await client.initialize()
            res = await client.create_meeting(
                title=f"Auray Zoom Meeting {meeting_id[:8]}",
                start_time=start_time,
                end_time=end_time,
                attendees=[],
                description="Created via webhook"
            )
            if res.get("success"):
                return res["meeting"]["join_url"]
            else:
                error_msg = res.get("error", "Zoom API call did not return a valid join URL")
                logger.error(f"Zoom API call succeeded but didn't return a valid URL: {error_msg}")
                raise HTTPException(status_code=500, detail=f"Failed to create Zoom meeting: {error_msg}")
        except Exception as e:
            logger.error(f"Zoom URL generation via API failed: {e}")
            raise HTTPException(status_code=500, detail=f"Zoom URL generation failed: {e}")
    
    elif platform_lower in ("teams", "microsoft_teams"):
        try:
            from services.meeting_agent.teams_client import create_teams_client
            client = create_teams_client()
            await client.initialize()
            await client.authenticate()
            res = await client.create_meeting(
                title=f"Auray Teams Meeting {meeting_id[:8]}",
                start_time=start_time,
                end_time=end_time,
                attendees=[],
                description="Created via webhook"
            )
            if res.get("success"):
                return res["meeting"]["join_url"]
            else:
                logger.warning(f"Teams API call failed: {res.get('error', 'Unknown error')}. Using fallback URL.")
                # Fallback to static Teams meeting URL
                return "https://teams.live.com/meet/9318960718018?p=J453ke6nEPHvg5kJGq"
        except Exception as e:
            logger.warning(f"Teams URL generation via API failed: {e}. Using fallback URL.")
            # Fallback to static Teams meeting URL
            return "https://teams.live.com/meet/9318960718018?p=J453ke6nEPHvg5kJGq"
    
    elif platform_lower == "google_meet":
        try:
            from services.meeting_agent.google_meet_client import create_google_meet_client
            client = create_google_meet_client()
            res = await client.create_meeting(
                title=f"Auray Google Meet {meeting_id[:8]}",
                start_time=start_time,
                end_time=end_time,
                attendees=[],
                description="Created via webhook"
            )
            if res.get("success"):
                return res["meeting"]["join_url"]
            else:
                error_msg = res.get("error", "Failed to create Google Meet")
                logger.error(f"Google Meet creation failed: {error_msg}")
                raise HTTPException(status_code=500, detail=f"Failed to create Google Meet: {error_msg}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Google Meet creation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create Google Meet: {e}")
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported platform: {platform}")


async def get_user_from_api_key(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    dao: DynamoDBDAO = Depends(get_dao)
) -> dict:
    """
    Authenticate using API key only (public endpoint secured by API key).
    Returns user info dict for webhook endpoints.
    """
    logger.info(f"Authorization header received: {authorization[:30] if authorization else None}...")
    
    if not authorization:
        logger.warning("No authorization header provided")
        raise HTTPException(
            status_code=401, 
            detail="API key required. Provide your API key in the Authorization header as 'Bearer <your_api_key>'"
        )
    
    if not authorization.startswith("Bearer "):
        logger.warning(f"Invalid authorization format: {authorization[:20]}...")
        raise HTTPException(
            status_code=401, 
            detail="Invalid authorization format. Use 'Bearer <your_api_key>'"
        )
    
    token = authorization[7:]  # Remove "Bearer " prefix
    
    if not token:
        logger.warning("Empty token after Bearer prefix")
        raise HTTPException(status_code=401, detail="API key is required")
    
    # API keys must start with "sk_"
    if not token.startswith("sk_"):
        logger.warning(f"Token doesn't start with 'sk_': {token[:20]}...")
        raise HTTPException(
            status_code=401, 
            detail="Invalid API key format. API keys must start with 'sk_live_'"
        )
    
    logger.info(f"Attempting API key validation for token starting with: {token[:15]}...")
    try:
        api_key = await dao.validate_api_key(token)
    except Exception as e:
        logger.error(f"Exception during API key validation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while validating the API key"
        )
    
    if not api_key:
        logger.warning(f"API key validation failed for token: {token[:15]}...")
        raise HTTPException(
            status_code=401, 
            detail="Invalid or expired API key. Please check your API key and try again."
        )
    
    logger.info(f"API key validated successfully for user {api_key.user_id}")
    
    # Get user from API key
    user = await dao.get_user_by_id(str(api_key.user_id))
    if not user:
        logger.error(f"User {api_key.user_id} not found for valid API key")
        raise HTTPException(status_code=401, detail="User not found")
    
    return {
        "user_id": str(user.id),
        "email": user.email,
        "name": user.name
    }


@router.post("/join_meeting")
async def join_meeting(
    request: JoinMeetingRequest,
    current_user: dict = Depends(get_user_from_api_key),
    dao: DynamoDBDAO = Depends(get_dao)
):
    """
    Join meeting webhook endpoint (public, secured by API key only).
    
    This endpoint:
    1. Creates a meeting URL using platform-specific clients
    2. Creates a meeting record in the database
    3. Spawns the browser bot process to join the meeting
    
    Authentication: API key only (Bearer token with API key starting with 'sk_live_')
    
    Returns a response indicating the meeting join status.
    """
    try:
        logger.info(f"Join meeting request received: {request.model_dump()}")
        
        # current_user is already authenticated via dependency
        user_id = UUID(current_user["user_id"])
        
        # Map platform string to enum
        platform_map = {
            "zoom": MeetingPlatform.ZOOM,
            "teams": MeetingPlatform.MICROSOFT_TEAMS,
            "microsoft_teams": MeetingPlatform.MICROSOFT_TEAMS,
            "google_meet": MeetingPlatform.GOOGLE_MEET,
        }
        platform_enum = platform_map.get(request.type.lower(), MeetingPlatform.GOOGLE_MEET)
        
        # Generate meeting ID
        meeting_id = uuid4()
        meeting_id_str = str(meeting_id)
        
        # Create meeting URL using platform client
        logger.info(f"Creating {request.type} meeting URL...")
        meeting_url = await _create_platform_meeting_url(request.type, meeting_id_str)
        
        # Extract external meeting ID from URL
        meeting_id_external = meeting_url.split("/")[-1].split("?")[0]
        
        # Create meeting record in database
        now = datetime.utcnow()
        meeting = Meeting(
            id=meeting_id,
            user_id=user_id,
            platform=platform_enum,
            meeting_url=meeting_url,
            meeting_id_external=meeting_id_external,
            title=f"Meeting {meeting_id_str[:8]}",
            description="Created via webhook",
            start_time=now,
            end_time=now + timedelta(minutes=30),
            organizer_email=settings.ai_email or "",
            participants=[],
            status=MeetingStatus.SCHEDULED,
            ai_email=settings.ai_email or "",
            audio_enabled=request.audio_record,
            video_enabled=request.video_record,
            recording_enabled=request.audio_record or request.video_record,
        )
        
        meeting = await dao.create_meeting(meeting)
        logger.info(f"Created meeting in DB: {meeting.id}")
        
        # Prepare environment variables for bot
        # Map SESSION_ID = meeting_id (as string)
        # Map BOT_NAME = bot_name from request
        # Browser bot is at the root level, not in clerk_backend
        # __file__ is at: clerk_backend/services/api/routes/webhooks.py
        # We need to go up 5 levels to get to the root: ../../../../../
        # 1: routes/, 2: api/, 3: services/, 4: clerk_backend/, 5: root/
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        bot_path = os.path.join(base_dir, "browser_bot", "bot_entry.js")
        
        # Log the path for debugging
        logger.info(f"Bot entry path: {bot_path}")
        logger.info(f"Base directory: {base_dir}")
        logger.info(f"Bot path exists: {os.path.exists(bot_path)}")
        
        if not os.path.exists(bot_path):
            logger.error(f"Bot entry file not found at: {bot_path}")
            meeting.status = MeetingStatus.FAILED
            await dao.update_meeting(meeting)
            raise HTTPException(status_code=500, detail=f"Bot entry file not found at: {bot_path}")
        
        env_vars = {
            'RT_GATEWAY_URL': os.getenv('RT_GATEWAY_URL', 'ws://localhost:8001'),
            'API_BASE_URL': os.getenv('API_BASE_URL', 'http://localhost:8000'),
            'SESSION_ID': meeting_id_str,  # Map SESSION_ID = meeting_id
            'MEETING_URL': meeting_url,
            'PLATFORM': request.type.lower(),
            'BOT_NAME': request.bot_name,  # Use bot_name from request
            'MEETING_ID': meeting_id_external,
            'LOG_LEVEL': 'info',
            'HEADLESS': 'false',
            'ENABLE_TTS_PLAYBACK': 'true',
            'ENABLE_AUDIO_CAPTURE': 'true' if request.audio_record else 'false',
        }
        
        # Spawn bot process
        logger.info(f"Spawning bot process for meeting {meeting_id_str}...")
        try:
            # Use subprocess.Popen to spawn process in background
            process = subprocess.Popen(
                ['node', bot_path],
                env={**os.environ, **env_vars},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(bot_path)
            )
            logger.info(f"Bot process spawned with PID: {process.pid}")
        except Exception as e:
            logger.error(f"Failed to spawn bot process: {e}")
            # Update meeting status to failed
            meeting.status = MeetingStatus.FAILED
            await dao.update_meeting(meeting)
            raise HTTPException(status_code=500, detail=f"Failed to spawn bot process: {str(e)}")
        
        # Update meeting status to JOINING
        meeting.status = MeetingStatus.JOINING
        await dao.update_meeting(meeting)
        
        # Return response
        return JoinMeetingResponse(
            success=True,
            message=f"Successfully created and joined {request.type} meeting. AI assistant is now active.",
            status="active",
            timestamp=datetime.utcnow().isoformat(),
            meeting_id=meeting_id_str,
            meeting_url=meeting_url,
            platform=request.type,
            voice_id=request.voice_id,
            capabilities={
                "transcript_enabled": request.transcript,
                "audio_recording_enabled": request.audio_record,
                "video_recording_enabled": request.video_record,
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process join meeting webhook: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to join meeting: {str(e)}"
        )

