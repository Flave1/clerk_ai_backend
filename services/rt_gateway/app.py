"""
Real-time Gateway FastAPI application.
Handles LiveKit integration, STT, TTS, and LLM orchestration.
"""
import asyncio
import io
import json
import logging
import sys
import wave
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict

# Python 3.12 compatibility fix for pydantic v1 ForwardRef._evaluate()
if sys.version_info >= (3, 12):
    import pydantic.v1.typing as pydantic_v1_typing
    original_evaluate = pydantic_v1_typing.evaluate_forwardref
    def patched_evaluate_forwardref(type_, globalns, localns):
        # Python 3.12 requires recursive_guard as a keyword argument
        return type_._evaluate(globalns, localns, set(), recursive_guard=set())
    pydantic_v1_typing.evaluate_forwardref = patched_evaluate_forwardref

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware

from shared.config import get_settings
from pydantic import BaseModel
from shared.schemas import (Action, Conversation, LLMRequest, LLMResponse,
                            STTRequest, STTResponse, TTSRequest, TTSResponse,
                            Turn)

from .events import EventPublisher
from .livekit_bridge import LiveKitBridge
from .llm import LLMService
from .stt import STTService
from .tts import TTSService
from .turn_manager import TurnManager
from .telephony import TelephonyHandler
from .external_turn_manager import ExternalTurnManager

# Configure logging
logging.basicConfig(level=logging.INFO)  # Use INFO level to reduce verbose debug logs
logger = logging.getLogger(__name__)

# Suppress verbose HTTP client logs that show binary data
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

settings = get_settings()

# Global services
livekit_bridge: LiveKitBridge = None
turn_manager: TurnManager = None
external_turn_manager: ExternalTurnManager = None  # Optional external turn manager for meeting bot
stt_service: STTService = None
tts_service: TTSService = None
llm_service: LLMService = None
event_publisher: EventPublisher = None
telephony_handler: TelephonyHandler = None

# Active conversations
active_conversations: Dict[str, Conversation] = {}

# Active bot sessions (session_id -> WebSocket connection)
active_bot_sessions: Dict[str, WebSocket] = {}
# Session metadata (session_id -> meeting_id, platform, etc.)
bot_session_metadata: Dict[str, dict] = {}


async def broadcast_to_conversation(conversation_id: str, message: Dict[str, Any]):
    """Broadcast a message to all participants in a conversation."""
    try:
        if conversation_id in active_conversations:
            conversation = active_conversations[conversation_id]
            # Send to all participants in the conversation
            for participant_id in conversation.participants:
                if participant_id in turn_manager.conversation_websockets:
                    websocket = turn_manager.conversation_websockets[participant_id]
                    try:
                        await websocket.send_text(json.dumps(message))
                    except Exception as e:
                        logger.warning(f"Failed to send message to participant {participant_id}: {e}")
    except Exception as e:
        logger.error(f"Failed to broadcast message to conversation {conversation_id}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global livekit_bridge, turn_manager, external_turn_manager, stt_service, tts_service, llm_service, event_publisher, telephony_handler

    logger.info("Starting RT Gateway services...")

    try:
        # Initialize services
        event_publisher = EventPublisher()
        await event_publisher.initialize()

        stt_service = STTService()
        tts_service = TTSService()
        llm_service = LLMService()
        turn_manager = TurnManager(llm_service, event_publisher, tts_service)

        livekit_bridge = LiveKitBridge(
            turn_manager, stt_service, tts_service, event_publisher
        )
        
        # Initialize telephony handler
        telephony_handler = TelephonyHandler(turn_manager, livekit_bridge, event_publisher)
        await telephony_handler.initialize()
        
        # Wire the services together
        turn_manager.livekit_bridge = livekit_bridge

        await livekit_bridge.initialize()

        # Initialize external turn manager for meeting bot if enabled
        if settings.use_external_turn_manager:
            external_turn_manager = ExternalTurnManager(llm_service)
            await external_turn_manager.initialize()
            logger.info("✅ External turn manager initialized (enabled via config)")
        else:
            logger.info("ℹ️ External turn manager disabled (use USE_EXTERNAL_TURN_MANAGER=true to enable)")

        logger.info("RT Gateway services initialized successfully")
        yield

    except Exception as e:
        logger.error(f"Failed to initialize RT Gateway services: {e}")
        raise
    finally:
        logger.info("Shutting down RT Gateway services...")
        if livekit_bridge:
            await livekit_bridge.cleanup()
        if external_turn_manager:
            await external_turn_manager.cleanup()


app = FastAPI(
    title="AI Receptionist (CLERK) RT Gateway",
    description="Real-time audio/video processing and AI orchestration",
    version=settings.app_version,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "rt_gateway",
        "version": settings.app_version,
        "active_conversations": len(active_conversations),
    }


@app.post("/conversations/start")
async def start_conversation(request: Dict[str, str]) -> Dict[str, Any]:
    """Start a new conversation."""
    try:
        room_id = request.get("room_id")
        user_id = request.get("user_id")
        
        if not room_id or not user_id:
            raise HTTPException(status_code=400, detail="room_id and user_id are required")
        
        conversation = await turn_manager.start_conversation(room_id, user_id)
        active_conversations[conversation.id] = conversation

        return {
            "conversation_id": str(conversation.id),
            "room_id": room_id,
            "status": "started",
        }
    except Exception as e:
        logger.error(f"Failed to start conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversations/{conversation_id}/join")
async def join_conversation(conversation_id: str, request: Dict[str, str]) -> Dict[str, Any]:
    """Join an existing conversation."""
    try:
        user_id = request.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Check if conversation exists
        conversation = active_conversations.get(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Add user to the conversation
        await turn_manager.add_participant_to_conversation(conversation_id, user_id)

        # Broadcast participant joined event to all connected clients
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


@app.post("/conversations/{conversation_id}/end")
async def end_conversation(conversation_id: str) -> Dict[str, Any]:
    """End a conversation."""
    try:
        conversation = active_conversations.get(conversation_id)
        if not conversation:
            logger.warning(f"Conversation {conversation_id} not found in active_conversations")
            # Still try to clean up turn_manager state
            try:
                await turn_manager.end_conversation(conversation_id)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup turn_manager state: {cleanup_error}")
            
            return {"status": "ended", "note": "Conversation was not active"}

        # End conversation in turn_manager
        await turn_manager.end_conversation(conversation_id)
        
        # Remove from active conversations
        del active_conversations[conversation_id]

        logger.info(f"Successfully ended conversation {conversation_id}")
        return {"status": "ended"}
        
    except Exception as e:
        logger.error(f"Failed to end conversation {conversation_id}: {e}")
        # Try to clean up anyway
        try:
            if conversation_id in active_conversations:
                del active_conversations[conversation_id]
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> Dict[str, Any]:
    """Delete a conversation and all related data."""
    try:
        # First check if conversation exists in active conversations
        conversation = active_conversations.get(conversation_id)
        
        # If conversation is active, end it first
        if conversation:
            logger.info(f"Ending active conversation {conversation_id} before deletion")
            await turn_manager.end_conversation(conversation_id)
            del active_conversations[conversation_id]
        
        # Delete conversation from database (this will also delete related turns and actions)
        try:
            from services.api.dao import DynamoDBDAO
            dao = DynamoDBDAO()
            await dao.initialize()
            await dao.delete_conversation(conversation_id)
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


@app.post("/conversations/bulk-delete")
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
                # First check if conversation exists in active conversations
                conversation = active_conversations.get(conversation_id)
                
                # If conversation is active, end it first
                if conversation:
                    logger.info(f"Ending active conversation {conversation_id} before deletion")
                    await turn_manager.end_conversation(conversation_id)
                    del active_conversations[conversation_id]
                
                # Delete conversation from database
                try:
                    from services.api.dao import DynamoDBDAO
                    dao = DynamoDBDAO()
                    await dao.initialize()
                    await dao.delete_conversation(conversation_id)
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


@app.post("/stt/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    conversation_id: str = Form(...),
    turn_number: int = Form(default=1)
):
    """Transcribe audio to text."""
    try:
        # Read audio data
        audio_data = await audio.read()
        
        # Create STT request
        from shared.schemas import STTRequest
        request = STTRequest(
            audio_data=audio_data,
            conversation_id=conversation_id,
            turn_number=turn_number
        )
        
        response = await stt_service.transcribe(request)
        return response
    except Exception as e:
        logger.error(f"STT transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest) -> TTSResponse:
    """Synthesize text to speech."""
    try:
        response = await tts_service.synthesize(request)
        return response
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/generate", response_model=LLMResponse)
async def generate_response(request: LLMRequest) -> LLMResponse:
    """Generate LLM response."""
    try:
        response = await llm_service.generate_response(request)
        return response
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class BotSpeakRequest(BaseModel):
    """Request model for bot speak endpoint."""
    text: str
    voice_id: str = "default"
    model: str = "openai"  # Options: '11lab' or 'openai'


@app.post("/bot/speak/{session_id}")
async def speak_to_bot(session_id: str, request: BotSpeakRequest):
    """
    Test endpoint to send text to a bot for TTS.
    The bot must be connected via WebSocket at /ws/bot/{session_id}
    
    Args:
        session_id: Bot session ID (path parameter)
        request: Request body with text, voice_id, and model
    
    Example:
    POST /bot/speak/teams-session
    Body: {"text": "Hello world", "voice_id": "default", "model": "openai"}
    
    Note: For backward compatibility, query parameters are also supported.
    """
    try:
        # Check if bot session exists
        if session_id not in active_bot_sessions:
            raise HTTPException(
                status_code=404, 
                detail=f"Bot session '{session_id}' not found. Bot must be connected via WebSocket at /ws/bot/{session_id}. Start the bot first!"
            )
        
        websocket = active_bot_sessions[session_id]
        
        logger.info(f"Sending TTS to bot session {session_id}: '{request.text[:50]}...' (voice={request.voice_id}, model={request.model})")
        
        # Synthesize audio and send PCM chunks
        chunk_count = 0
        async for pcm_chunk in tts_service.synthesize_to_pcm(request.text, request.voice_id, request.model):
            await websocket.send_bytes(pcm_chunk)
            chunk_count += 1
        
        # Send completion marker
        await websocket.send_text(json.dumps({
            "type": "tts_complete",
            "session_id": session_id
        }))
        
        logger.info(f"Successfully sent TTS to bot session {session_id} ({chunk_count} chunks)")
        
        return {
            "status": "success",
            "message": "Text sent to bot for TTS",
            "session_id": session_id,
            "text": request.text,
            "voice_id": request.voice_id,
            "model": request.model,
            "chunks_sent": chunk_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to speak to bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    
    # Load existing conversation if it exists
    conversation = await turn_manager.load_existing_conversation(conversation_id)
    if not conversation:
        logger.error(f"Conversation {conversation_id} not found in database")
        await websocket.close(code=1000, reason="Conversation not found")
        return
    
    logger.info(f"Successfully loaded conversation {conversation_id} for WebSocket")
    
    # Register WebSocket with turn manager for TTS audio streaming
    turn_manager.register_websocket(conversation_id, websocket)

    try:
        while True:
            try:
                # Try to receive text message first
                try:
                    data = await websocket.receive_text()
                    
                    # Handle ping/pong for keepalive
                    if data == "ping":
                        await websocket.send_text("pong")
                        continue
                    
                    logger.info(f"Received text message: {data}")
                    
                    # Process text message through turn manager
                    await process_text_message(conversation_id, data, websocket)
                    
                except Exception:
                    # If text fails, try to receive binary data
                    try:
                        binary_data = await websocket.receive_bytes()
                        logger.info(f"Received binary audio data: {len(binary_data)} bytes")
                        
                        # Audio processing disabled - using frontend STT instead
                        await websocket.send_text("Audio received (STT handled on frontend)")
                        
                    except Exception as binary_error:
                        logger.error(f"Failed to receive binary data: {binary_error}")
                        break
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for conversation {conversation_id}")
                break
            except Exception as e:
                logger.error(f"Failed to receive message: {e}")
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for conversation {conversation_id}")
    except Exception as e:
        logger.error(f"WebSocket error for conversation {conversation_id}: {e}")
        await websocket.close()
    finally:
        # Unregister WebSocket when connection closes
        turn_manager.unregister_websocket(conversation_id)


async def process_text_message(conversation_id: str, message: str, websocket: WebSocket):
    """Process text message through the AI pipeline."""
    try:
        # Create a user turn
        from shared.schemas import Turn, TurnType
        from datetime import datetime, timezone
        from uuid import uuid4, UUID
        
        user_turn = Turn(
            id=uuid4(),
            conversation_id=UUID(conversation_id),  # Use conversation_id from URL
            turn_number=1,
            turn_type=TurnType.USER_SPEECH,
            content=message,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Process through turn manager
        logger.info(f"Processing turn for conversation {conversation_id}")
        logger.info(f"Conversation exists in turn_manager: {conversation_id in turn_manager.conversations}")
        logger.info(f"Available conversations in turn_manager: {list(turn_manager.conversations.keys())}")
        
        # Process the turn and get the AI response
        result = await turn_manager.process_turn_and_get_response(user_turn)
        
        if result:
            # Send text response
            if isinstance(result, tuple):
                # Extract text response from tuple
                ai_response, _ = result
                await websocket.send_text(ai_response)
                logger.info(f"Sent AI text response to WebSocket: {ai_response[:100]}...")
            else:
                # Direct text response
                await websocket.send_text(result)
                logger.info(f"Sent AI response to WebSocket: {result[:100]}...")
        else:
            # Fallback response
            response = f"I received your message: '{message}'. How can I help you today? You can ask me to schedule meetings, send Slack messages, or search for information."
            await websocket.send_text(response)
        
    except Exception as e:
        logger.error(f"Failed to process text message: {e}")
        await websocket.send_text("Sorry, I encountered an error processing your message.")


@app.websocket("/ws/bot/{session_id}")
async def bot_audio_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for browser bot audio streaming.
    Bot connects here to receive PCM audio chunks for TTS.
    """
    await websocket.accept()
    logger.info(f"Bot audio stream connected for session {session_id}")
    
    # Register this bot session
    active_bot_sessions[session_id] = websocket
    
    # Store session metadata
    bot_session_metadata[session_id] = {
        "meeting_id": session_id,  # Default to session_id, will be updated by bot_registration
        "bot_name": "Unknown Bot",
        "platform": "unknown",
        "connected_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Start session in external turn manager if enabled (will be updated with real meeting_id)
    if settings.use_external_turn_manager and external_turn_manager:
        meeting_id = bot_session_metadata[session_id]["meeting_id"]
        await external_turn_manager.start_session(session_id, meeting_id)
    
    try:
        while True:
            try:
                # Receive message from bot
                data = await websocket.receive()
                
                if 'text' in data:
                    # Text message (request for TTS, registration, etc.)
                    try:
                        message = json.loads(data['text'])
                        logger.info(f"Received message from bot: {message}")
                        
                        # Handle bot registration - includes meeting_id
                        if message.get('type') == 'bot_registration':
                            old_meeting_id = bot_session_metadata[session_id].get("meeting_id", session_id)
                            bot_session_metadata[session_id].update({
                                "meeting_id": message.get('meetingId', session_id),
                                "bot_name": message.get('botName', 'Unknown Bot'),
                                "platform": message.get('platform', 'unknown')
                            })
                            new_meeting_id = bot_session_metadata[session_id]["meeting_id"]
                            logger.info(f"Bot registered", {
                                "session_id": session_id,
                                "meeting_id": new_meeting_id,
                                "bot_name": bot_session_metadata[session_id]["bot_name"],
                                "platform": bot_session_metadata[session_id]["platform"]
                            })
                            # Update external turn manager if meeting_id changed
                            if settings.use_external_turn_manager and external_turn_manager and old_meeting_id != new_meeting_id:
                                # End old session and start new one with correct meeting_id
                                await external_turn_manager.end_session(session_id)
                                await external_turn_manager.start_session(session_id, new_meeting_id)
                                logger.info(f"Updated external turn manager session with meeting_id: {new_meeting_id}")
                            continue  # Don't process as TTS request
                        
                        if message.get('type') == 'tts_request':
                            text = message.get('text', '')
                            voice_id = message.get('voice_id', 'default')
                            model = message.get('model', '11lab')  # Default to ElevenLabs
                            
                            # Synthesize audio and send PCM chunks
                            async for pcm_chunk in tts_service.synthesize_to_pcm(text, voice_id, model):
                                # Check connection state before sending
                                if websocket.client_state.name != "CONNECTED":
                                    logger.debug(f"WebSocket not connected, stopping TTS stream. State: {websocket.client_state.name}")
                                    break
                                try:
                                    # Send binary PCM data
                                    await websocket.send_bytes(pcm_chunk)
                                except (WebSocketDisconnect, RuntimeError):
                                    logger.debug("WebSocket closed during TTS stream")
                                    break
                            
                            # Send completion marker (if still connected)
                            try:
                                if websocket.client_state.name == "CONNECTED":
                                    await websocket.send_text(json.dumps({
                                        'type': 'tts_complete',
                                        'session_id': session_id
                                    }))
                            except (WebSocketDisconnect, RuntimeError):
                                pass
                            
                        elif data['text'] == "ping":
                            try:
                                if websocket.client_state.name == "CONNECTED":
                                    await websocket.send_text("pong")
                            except (WebSocketDisconnect, RuntimeError):
                                break
                            
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON message from bot")
                        try:
                            if websocket.client_state.name == "CONNECTED":
                                await websocket.send_text(json.dumps({
                                    'type': 'error',
                                    'message': 'Invalid JSON'
                                }))
                        except (WebSocketDisconnect, RuntimeError):
                            break
                elif 'bytes' in data:
                    # Silently ignore binary data - no need to log
                    pass
                    
            except WebSocketDisconnect:
                logger.info(f"Bot audio stream disconnected for session {session_id}")
                break
            except RuntimeError as e:
                # Handle "Cannot call receive once a disconnect message has been received"
                if "disconnect" in str(e).lower() or "receive" in str(e).lower():
                    logger.info("WebSocket disconnected (receive error)")
                    break
                else:
                    logger.error(f"Runtime error in bot audio stream: {e}")
                    break
                
    except WebSocketDisconnect:
        logger.info(f"Bot audio stream disconnected for session {session_id}")
    except Exception as e:
        error_str = str(e).lower()
        if "disconnect" in error_str or "receive" in error_str or "close" in error_str:
            logger.info(f"Bot audio stream closed for session {session_id}: {e}")
        else:
            logger.error(f"Bot audio stream error for session {session_id}: {e}")
            # Don't try to close if already closed
            try:
                if websocket.client_state.name == "CONNECTED":
                    await websocket.close()
            except Exception:
                pass  # Already closed or closing
    finally:
        # Unregister bot session and cleanup metadata
        if session_id in active_bot_sessions:
            del active_bot_sessions[session_id]
        if session_id in bot_session_metadata:
            del bot_session_metadata[session_id]
        
        # End session in external turn manager if enabled
        if settings.use_external_turn_manager and external_turn_manager:
            await external_turn_manager.end_session(session_id)
        
        logger.info(f"Bot audio stream closed for session {session_id}")


@app.websocket("/ws/bot/audio/{session_id}")
async def bot_audio_input(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for receiving audio from bot.
    Flow: Receive audio → STT → LLM → TTS → Send to bot.
    """
    await websocket.accept()
    logger.info(f"Bot audio input stream connected for session {session_id}")
    
    # Track conversation context for LLM (per session) - fallback if external turn manager disabled
    if not hasattr(bot_audio_input, 'conversation_contexts'):
        bot_audio_input.conversation_contexts = {}
    
    if session_id not in bot_audio_input.conversation_contexts:
        bot_audio_input.conversation_contexts[session_id] = []
    
    conversation_context = bot_audio_input.conversation_contexts[session_id]
    
    # Audio buffer - accumulate audio for processing
    audio_buffer = b""
    min_buffer_size = 96000  # 3 seconds at 16kHz, 16-bit (16000 samples/sec * 2 bytes/sample * 3 sec = 96000 bytes)
    max_buffer_size = 192000  # 6 seconds max - prevents unbounded growth
    
    # Cooldown between transcriptions (6 seconds to prevent echo loop)
    import time
    last_transcription_time = 0
    transcription_cooldown = 6.0
    
    # Track if we're currently processing to avoid overlapping transcriptions
    processing = False
    
    try:
        await websocket.send_text(json.dumps({"type": "connected", "message": "Ready to receive audio"}))
        
        while True:
            try:
                data = await websocket.receive()
                
                if 'type' in data and data.get('type') == 'websocket.disconnect':
                    break
                
                if 'bytes' in data:
                    audio_chunk = data['bytes']
                    audio_buffer += audio_chunk
                    
                    # Process when we have enough audio (3 seconds) and not already processing
                    if len(audio_buffer) >= min_buffer_size and not processing:
                        # Check for speech activity before processing (VAD)
                        has_speech = stt_service.detect_speech_activity(audio_buffer, sample_rate=16000)
                        
                        if not has_speech:
                            # No speech detected, clear buffer and continue
                            logger.debug(f"⏭️ No speech detected in {len(audio_buffer)} bytes buffer, clearing")
                            audio_buffer = b""
                            continue
                        
                        # Speech detected - send audio directly to GPT-4o
                        logger.info(f"🎤 Speech detected in {len(audio_buffer)} bytes buffer, sending to GPT-4o...")
                        processing = True
                        audio_to_process = audio_buffer
                        # Keep a small overlap (0.5 seconds) to prevent cutting off sentences
                        overlap_bytes = 16000  # 0.5 seconds
                        audio_buffer = audio_buffer[-overlap_bytes:] if len(audio_buffer) > overlap_bytes else b""
                        
                        # Check cooldown period
                        current_time = time.time()
                        if current_time - last_transcription_time < transcription_cooldown:
                            logger.debug(f"Skipping audio processing (cooldown)")
                            processing = False
                            continue
                        
                        last_transcription_time = current_time
                        
                        # Send audio directly to GPT-4o and get text response
                        try:
                            full_response = ""
                            logger.info(f"🤖 Sending {len(audio_to_process)} bytes audio to GPT-4o...")
                            
                            async for llm_chunk in llm_service.generate_response_streaming(
                                audio_data=audio_to_process,
                                conversation_context=conversation_context
                            ):
                                full_response += llm_chunk
                            
                            if not full_response or not full_response.strip():
                                logger.warning("⚠️ GPT-4o returned empty response")
                                processing = False
                                continue
                            
                            # Update conversation context
                            conversation_context.append({"role": "user", "content": {"audio": len(audio_to_process)}})
                            conversation_context.append({"role": "assistant", "content": full_response})
                            if len(conversation_context) > 20:
                                conversation_context = conversation_context[-20:]
                            
                            logger.info(f"🎙️ GPT-4o Response: '{full_response}'")
                            
                        except Exception as llm_error:
                            logger.error(f"GPT-4o audio processing failed: {llm_error}", exc_info=True)
                            processing = False
                            continue
                        
                        processing = False
                        
                        # Send TTS response back to bot
                        if session_id in active_bot_sessions:
                            tts_websocket = active_bot_sessions[session_id]
                            
                            chunk_count = 0
                            async for pcm_chunk in tts_service.synthesize_to_pcm(full_response, "default", "openai"):
                                await tts_websocket.send_bytes(pcm_chunk)
                                chunk_count += 1
                            
                            await tts_websocket.send_text(json.dumps({
                                "type": "tts_complete",
                                "session_id": session_id
                            }))
                            
                            logger.info(f"✅ Sent TTS to bot session {session_id} ({chunk_count} chunks)")
                        else:
                            logger.warning(f"⚠️ Bot TTS WebSocket not connected for session {session_id}")
                    
                    # Prevent buffer from growing too large (check after each chunk)
                    if len(audio_buffer) > max_buffer_size:
                        logger.debug(f"📊 Buffer exceeded max size ({max_buffer_size} bytes), trimming to latest {min_buffer_size} bytes")
                        # Keep the most recent audio (last 3 seconds)
                        audio_buffer = audio_buffer[-min_buffer_size:]
                
                elif 'text' in data:
                    # Handle control messages
                    try:
                        message = json.loads(data['text'])
                        if message.get('type') == 'end_stream':
                            break
                    except json.JSONDecodeError:
                        pass
                        
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"Bot audio input stream disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Bot audio input stream error for session {session_id}: {e}")
    finally:
        # Clean up conversation context if not using external turn manager
        if session_id in bot_audio_input.conversation_contexts:
            del bot_audio_input.conversation_contexts[session_id]
        
        logger.info(f"Bot audio input stream closed for session {session_id}")


# Audio processing disabled - using frontend STT instead
# async def process_audio_data(conversation_id: str, audio_data: bytes, websocket: WebSocket):
#     """Process audio data through the AI pipeline."""
#     try:
#         logger.info(f"Processing audio data for conversation {conversation_id}: {len(audio_data)} bytes")
#         
#         # Process audio through LiveKit bridge for STT
#         await livekit_bridge.process_audio_data(conversation_id, audio_data)
#         
#         # Send acknowledgment
#         await websocket.send_text("Audio received and processing...")
#         
#     except Exception as e:
#         logger.error(f"Failed to process audio data: {e}")
#         await websocket.send_text("Sorry, I encountered an error processing your audio.")


# ============================================================================
# TELEPHONY WEBHOOK ENDPOINTS
# ============================================================================

@app.post("/telephony/incoming")
async def handle_incoming_call(request: Request):
    """Handle incoming phone call webhook from LiveKit Voice."""
    try:
        body = await request.body()
        
        # Verify webhook signature
        if not telephony_handler.verify_webhook_signature(request, body):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        # Parse webhook data
        webhook_data = await request.json()
        logger.info(f"Received incoming call webhook: {webhook_data}")
        
        # Handle the incoming call
        response = await telephony_handler.handle_incoming_call(webhook_data)
        return response
        
    except Exception as e:
        logger.error(f"Failed to handle incoming call webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/telephony/answered")
async def handle_call_answered(request: Request):
    """Handle call answered webhook from LiveKit Voice."""
    try:
        body = await request.body()
        
        # Verify webhook signature
        if not telephony_handler.verify_webhook_signature(request, body):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        # Parse webhook data
        webhook_data = await request.json()
        logger.info(f"Received call answered webhook: {webhook_data}")
        
        # Handle the call answered event
        response = await telephony_handler.handle_call_answered(webhook_data)
        return response
        
    except Exception as e:
        logger.error(f"Failed to handle call answered webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/telephony/ended")
async def handle_call_ended(request: Request):
    """Handle call ended webhook from LiveKit Voice."""
    try:
        body = await request.body()
        
        # Verify webhook signature
        if not telephony_handler.verify_webhook_signature(request, body):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        # Parse webhook data
        webhook_data = await request.json()
        logger.info(f"Received call ended webhook: {webhook_data}")
        
        # Handle the call ended event
        response = await telephony_handler.handle_call_ended(webhook_data)
        return response
        
    except Exception as e:
        logger.error(f"Failed to handle call ended webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/telephony/failed")
async def handle_call_failed(request: Request):
    """Handle call failed webhook from LiveKit Voice."""
    try:
        body = await request.body()
        
        # Verify webhook signature
        if not telephony_handler.verify_webhook_signature(request, body):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        # Parse webhook data
        webhook_data = await request.json()
        logger.info(f"Received call failed webhook: {webhook_data}")
        
        # Handle the call failed event
        response = await telephony_handler.handle_call_failed(webhook_data)
        return response
        
    except Exception as e:
        logger.error(f"Failed to handle call failed webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/telephony/calls")
async def get_active_calls():
    """Get all active phone calls."""
    try:
        calls = await telephony_handler.get_active_calls()
        return {"active_calls": calls, "count": len(calls)}
        
    except Exception as e:
        logger.error(f"Failed to get active calls: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/telephony/calls/{call_id}")
async def get_call_status(call_id: str):
    """Get status of a specific call."""
    try:
        call_status = await telephony_handler.get_call_status(call_id)
        if not call_status:
            raise HTTPException(status_code=404, detail="Call not found")
        
        return {"call_id": call_id, "status": call_status}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get call status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """
    Get conversation history for a meeting bot session.
    Requires USE_EXTERNAL_TURN_MANAGER=true to be enabled.
    """
    try:
        if not settings.use_external_turn_manager or not external_turn_manager:
            raise HTTPException(
                status_code=400, 
                detail="External turn manager is not enabled. Set USE_EXTERNAL_TURN_MANAGER=true"
            )
        
        history = await external_turn_manager.get_conversation_history(session_id)
        
        return {
            "status": "success",
            "session_id": session_id,
            "history": history,
            "turn_count": len(history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session history for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/telephony/calls/{call_id}/hangup")
async def hangup_call(call_id: str):
    """Manually hang up a call."""
    try:
        success = await telephony_handler.hangup_call(call_id)
        if not success:
            raise HTTPException(status_code=404, detail="Call not found or already ended")
        
        return {"status": "success", "message": f"Call {call_id} hung up successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to hangup call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.rt_gateway.app:app", host=settings.rt_gateway_host, port=settings.rt_gateway_port, reload=settings.debug
    )
