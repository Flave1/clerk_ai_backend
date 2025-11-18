"""
Webhooks API routes for testing and integration.
"""
import logging
import subprocess
import os
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status, Depends, Header
from pydantic import BaseModel

from shared.schemas import JoinMeetingRequest, MeetingStatus, MeetingContext
from services.api.dao import get_dao, MongoDBDAO
from shared.config import get_settings
from services.meeting_agent.bot_orchestrator import get_bot_orchestrator
from services.api.service_meeting import ServiceMeeting
from services.api.service_meeting_context import ServiceMeetingContext
from services.api.meeting_url import create_platform_meeting_url

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


async def _launch_bot_subprocess_fallback(
    meeting_id: str,
    meeting_url: str,
    platform: str,
    bot_name: str,
    meeting_id_external: str,
    audio_record: bool
) -> bool:
    """
    FALLBACK: Launch browser bot using subprocess.Popen (original method).
    
    This is kept as a backup option if you want to switch back from Docker/ECS orchestration.
    To use this instead of the orchestrator, replace the orchestrator call in join_meeting()
    with a call to this function.
    
    Returns True if process spawned successfully, False otherwise.
    """
    try:
        # Find bot entry script
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        bot_path = os.path.join(base_dir, "browser_bot", "bot_entry.js")
        
        logger.info(f"Bot entry path: {bot_path}")
        
        if not os.path.exists(bot_path):
            logger.error(f"Bot entry file not found at: {bot_path}")
            return False
        
        # Normalize platform name for browser_bot
        platform_for_bot = platform.lower()
        if platform_for_bot == 'microsoft_teams':
            platform_for_bot = 'teams'
        
        env_vars = {
            'RT_GATEWAY_URL': os.getenv('RT_GATEWAY_URL'),
            'API_BASE_URL': os.getenv('API_BASE_URL'),
            'SESSION_ID': meeting_id,
            'MEETING_URL': meeting_url,
            'PLATFORM': platform_for_bot,
            'BOT_NAME': bot_name,
            'MEETING_ID': meeting_id_external,
            'LOG_LEVEL': 'info',
            'HEADLESS': 'true',
            'ENABLE_TTS_PLAYBACK': 'true',
            'ENABLE_AUDIO_CAPTURE': 'true' if audio_record else 'false',
        }
        
        # Create log file paths
        log_dir = os.path.join(base_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'bot_{meeting_id}.log')
        
        # Write startup info to log
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Bot process starting at {datetime.now().isoformat()}\n")
            f.write(f"Environment: {', '.join([f'{k}={v}' for k, v in env_vars.items()])}\n")
            f.write(f"{'='*80}\n")
        
        # Open log file for subprocess
        log_fh = open(log_file, 'a', buffering=1)
        
        try:
            process = subprocess.Popen(
                ['node', bot_path],
                env={**os.environ, **env_vars},
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(bot_path),
                start_new_session=True,
                close_fds=False,
            )
            
            # Write PID to log
            log_fh.write(f"Bot process PID: {process.pid}\n")
            log_fh.flush()
            
            logger.info(f"Bot process spawned with PID: {process.pid}, log: {log_file}")
            
            # Keep file handle open (child process inherits it)
            # Process is detached via start_new_session=True
            
            return True
            
        except Exception as e:
            log_fh.close()
            logger.error(f"Failed to spawn bot process: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error in subprocess fallback: {e}")
        return False


async def get_user_from_api_key(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    dao: MongoDBDAO = Depends(get_dao)
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
    dao: MongoDBDAO = Depends(get_dao)
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

        logger.info(f"User ID for join_meeting: {user_id}")
        
        # Create meeting record in database via service
        meeting_service = ServiceMeeting(dao)
        meeting = await meeting_service.create_meeting_record(
            user_id=user_id,
            meeting_type=request.type,
            meeting_url=request.meeting_url,
            audio_record=request.audio_record,
            video_record=request.video_record,
            transcript=request.transcript,
            voice_id=request.voice_id,
            bot_name=request.bot_name,
            context_id=request.context_id,
        )
        
        
        try:
            meeting_id_str = str(meeting.id)
            meeting_url = meeting.meeting_url
            orchestrator = get_bot_orchestrator()
            
            # Prepare additional environment variables specific to this request
            additional_env_vars = {
                'MEETING_ID': meeting_id_str,  # External meeting ID from URL
                'ENABLE_AUDIO_CAPTURE': 'true' if request.audio_record else 'false',
                'ENABLE_TTS_PLAYBACK': 'true',
            }
            
            # Add context_id if provided
            if request.context_id:
                additional_env_vars['CONTEXT_ID'] = request.context_id
            
            # Launch bot via orchestrator (will try Docker -> ECS -> subprocess in that order)
            bot_launched = await orchestrator.launch_bot(
                meeting_id=meeting_id_str,
                platform=request.type,  # Will be normalized internally
                meeting_url=meeting_url,
                bot_name=request.bot_name,
                session_id=meeting_id_str,  # Use meeting_id as session_id
                additional_env_vars=additional_env_vars
            )
            
            if not bot_launched:
                logger.error(f"Failed to launch browser bot for meeting: {meeting_id_str}")
                meeting.status = MeetingStatus.FAILED
                await meeting_service.update_meeting(meeting)
                raise HTTPException(
                    status_code=500,
                    detail="Failed to launch browser bot. Check logs for details."
                )
            
            logger.info(f"✅ Browser bot launched successfully for meeting: {meeting_id_str}")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to launch browser bot: {e}", exc_info=True)
            meeting.status = MeetingStatus.FAILED
            await meeting_service.update_meeting(meeting)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to launch browser bot: {str(e)}"
            )
        
        # Update meeting status to JOINING
        meeting.status = MeetingStatus.JOINING
        await meeting_service.update_meeting(meeting)
        
        # Return response
        response = JoinMeetingResponse(
            success=True,
            message=f"Successfully!! Aurray will join the meeting when let in into the meeting",
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
        logger.info(f"Join meeting response: {response.model_dump()}")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process join meeting webhook: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to join meeting: {str(e)}"
        )


@router.get("/voice_profiles")
async def get_voice_profiles(
    current_user: dict = Depends(get_user_from_api_key),
):
    """
    Get available voice profiles webhook endpoint (public, secured by API key only).
    
    Returns a list of available voice profiles that can be used for TTS.
    
    Authentication: API key only (Bearer token with API key starting with 'sk_live_')
    """
    try:
        logger.info("Get voice profiles request received")
        
        # For now, return only the default voice until voice profile setup is complete
        voices = [
            {
                "id": "f5HLTX707KIM4SzJYzSz",
                "name": "Aurray Alloy (default)",
                "provider": "default",
                "category": "default"
            }
        ]
        
        return {
            "success": True,
            "voices": voices,
            "count": len(voices),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get voice profiles: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get voice profiles: {str(e)}"
        )


@router.get("/meeting_contexts")
async def get_meeting_contexts_webhook(
    current_user: dict = Depends(get_user_from_api_key),
    dao: MongoDBDAO = Depends(get_dao),
):
    """
    Get meeting contexts webhook endpoint (public, secured by API key only).
    
    Returns a list of meeting contexts for the authenticated user.
    
    Authentication: API key only (Bearer token with API key starting with 'sk_live_')
    """
    try:
        logger.info("Get meeting contexts request received")
        user_id = current_user["user_id"]
        context_service = ServiceMeetingContext(dao)
        
        contexts = await context_service.get_contexts_by_user(user_id)
        
        return {
            "success": True,
            "contexts": [
                {
                    "id": str(ctx.id),
                    "name": ctx.name,
                    "voice_id": ctx.voice_id,
                    "context_description": ctx.context_description,
                    "tools_integrations": ctx.tools_integrations,
                    "meeting_role": ctx.meeting_role.value,
                    "tone_personality": ctx.tone_personality.value,
                    "custom_tone": ctx.custom_tone,
                    "created_at": ctx.created_at.isoformat(),
                    "updated_at": ctx.updated_at.isoformat(),
                }
                for ctx in contexts
            ],
            "count": len(contexts),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get meeting contexts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get meeting contexts: {str(e)}"
        )

