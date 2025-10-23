"""
Real-time Gateway FastAPI application.
Handles LiveKit integration, STT, TTS, and LLM orchestration.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware

from shared.config import get_settings
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Global services
livekit_bridge: LiveKitBridge = None
turn_manager: TurnManager = None
stt_service: STTService = None
tts_service: TTSService = None
llm_service: LLMService = None
event_publisher: EventPublisher = None
telephony_handler: TelephonyHandler = None

# Active conversations
active_conversations: Dict[str, Conversation] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global livekit_bridge, turn_manager, stt_service, tts_service, llm_service, event_publisher, telephony_handler

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

        logger.info("RT Gateway services initialized successfully")
        yield

    except Exception as e:
        logger.error(f"Failed to initialize RT Gateway services: {e}")
        raise
    finally:
        logger.info("Shutting down RT Gateway services...")
        if livekit_bridge:
            await livekit_bridge.cleanup()


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
