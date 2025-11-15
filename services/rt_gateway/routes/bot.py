"""
Bot WebSocket routes using OpenAI Realtime API.

This replaces the traditional STT → LLM → TTS pipeline with OpenAI's Realtime API
for unified speech-to-speech processing with near-zero latency.
"""
import asyncio
import json
import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import base64
from shared.config import get_settings
from .. import services as rt_services
from ..realtime_api import RealtimeAPIService

logger = logging.getLogger(__name__)
router = APIRouter()
ws_router = APIRouter()  # Separate router for WebSocket routes
settings = get_settings()

# Store active Realtime API sessions per session_id
active_realtime_sessions: Dict[str, RealtimeAPIService] = {}


class BotSpeakRequest(BaseModel):
    """Request model for bot speak endpoint."""
    text: str
    voice_id: str = "default"
    model: str = "openai"


async def handle_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle tool calls from Realtime API.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments
    
    Returns:
        Tool execution result
    """
    if tool_name == "saveNote":
        note_text = arguments.get("text", "")
        # TODO: Integrate with your note storage system
        return {"status": "ok", "message": "Note saved successfully"}
    
    elif tool_name == "createTask":
        task = arguments.get("task", "")
        assignee = arguments.get("assignee")
        # TODO: Integrate with your task management system
        return {"status": "ok", "message": "Task created successfully"}
    
    elif tool_name == "summarizeDiscussion":
        topic = arguments.get("topic", "")
        # TODO: Generate and return summary
        return {"status": "ok", "summary": f"Summary of {topic}"}
    
    else:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}


@ws_router.websocket("/bot/{session_id}")
async def bot_websocket(websocket: WebSocket, session_id: str):
    """
    Unified WebSocket endpoint for bot audio streaming.
    
    Handles BOTH input and output on a single connection:
    - Receives audio chunks (bytes) → forwards to Realtime API
    - Receives control messages (text) → handles registration, commits, etc.
    - Sends audio output (bytes) from Realtime API
    - Sends text messages (transcription, ai_response, etc.)
    """
    await websocket.accept()
    
    # Store WebSocket for Realtime API to send audio to
    rt_services.active_bot_sessions[session_id] = websocket
    rt_services.bot_session_metadata[session_id] = rt_services.bot_session_metadata.get(session_id, {})
    
    # Track transcript for sending to frontend
    current_transcript = ""
    current_ai_response = ""
    
    try:
        # Wait for registration message from frontend
        registration_received = False
        while not registration_received:
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=5.0)
                
                if "text" in data:
                    message = json.loads(data["text"])
                    # Support both old and new registration formats
                    msg_type = message.get("type")
                    if msg_type == "bot_registration" or msg_type == "register":
                        # Store registration info
                        rt_services.bot_session_metadata[session_id].update({
                            "meeting_id": message.get("meetingId") or message.get("meeting_id") or session_id,
                            "bot_name": message.get("botName") or message.get("bot_name") or "Aurray",
                            "platform": message.get("platform", "clerk"),
                            "audio_config": message.get("audioConfig") or message.get("audio_config") or {},
                            "participants": message.get("participants", []),
                            "topic": message.get("topic", "General meeting")
                        })
                        registration_received = True
                        
                        # NEW: Send connected message for new format
                        if msg_type == "register":
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "connected",
                                    "session_id": session_id
                                }))
                            except Exception as e:
                                pass
            except asyncio.TimeoutError:
                registration_received = True
        
        # Initialize or reuse Realtime API service
        # Check if session already exists (from previous connection)
        realtime_service = active_realtime_sessions.get(session_id)
        
        if not realtime_service:
            # No existing session - create new one
            await _initialize_realtime_session(session_id, websocket, [current_transcript], [current_ai_response])
            realtime_service = active_realtime_sessions.get(session_id)
        
        if not realtime_service:
            logger.error(f"Failed to initialize Realtime API for session {session_id}")
            await websocket.close(code=1008, reason="Failed to initialize Realtime API")
            return
        
        # Main message loop - handles both input and output
        while True:
            try:
                data = await websocket.receive()
                
                if 'type' in data and data.get('type') == 'websocket.disconnect':
                    break
                
                if 'bytes' in data:
                    # Raw PCM16 audio input chunk (16kHz or 24kHz) - base64 encode and forward to OpenAI
                    audio_chunk = data['bytes']
                    
                    if not realtime_service:
                        continue
                    
                    try:
                        # Base64 encode the raw PCM bytes
                        audio_base64 = base64.b64encode(audio_chunk).decode("utf-8")
                        
                        # Send to OpenAI Realtime API as input_audio_buffer.append
                        await realtime_service._send_message({
                            "type": "input_audio_buffer.append",
                            "audio": audio_base64
                        })
                    except Exception as e:
                        logger.error(f"Failed to send audio chunk to Realtime API: {e}", exc_info=True)
                        # Connection is likely closed, break the loop
                        break
                            
                elif 'text' in data:
                    try:
                        message = json.loads(data["text"])
                        msg_type = message.get("type")
                        
                        if msg_type == "bot_registration" or msg_type == "register":
                            # Already handled above, but acknowledge
                            pass
                        elif msg_type == "ping":
                            # Respond to ping
                            await websocket.send_text(json.dumps({"type": "pong"}))
                        elif msg_type == "input_audio_buffer.commit" or msg_type == "commit":
                            # User finished speaking - commit audio buffer
                            try:
                                await realtime_service._send_message({
                                    "type": "input_audio_buffer.commit"
                                })
                            except Exception as e:
                                logger.error(f"Failed to commit audio: {e}", exc_info=True)
                        elif msg_type == "response.cancel" or msg_type == "interrupt":
                            # User started speaking (interrupt current response)
                            try:
                                await realtime_service._send_message({
                                    "type": "response.cancel"
                                })
                            except Exception as e:
                                logger.error(f"Failed to interrupt: {e}", exc_info=True)
                        elif msg_type == "response.create":
                            # Trigger response generation after commit
                            try:
                                await realtime_service._send_message({
                                    "type": "response.create"
                                })
                            except Exception as e:
                                logger.error(f"Failed to create response: {e}", exc_info=True)
                        elif msg_type == "end_stream":
                            break
                        elif msg_type == "text":
                            # Text message (fallback for text input)
                            text_payload = message.get("content", "")
                            if text_payload:
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "message": "Text input not supported in audio mode"
                                }))
                        else:
                            logger.warning(f"Unknown message type: {msg_type}, full message: {message}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse WebSocket text message: {e}, raw data: {data.get('text', '')[:100]}")
                    except Exception as e:
                        logger.error(f"Error processing WebSocket text message: {e}", exc_info=True)
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in unified stream: {e}", exc_info=True)
                break
                
    except Exception as e:
        logger.error(f"Unified bot audio stream error for session {session_id}: {e}", exc_info=True)
    finally:
        # Cleanup - only if this is the last WebSocket for this session
        if session_id in rt_services.active_bot_sessions:
            # Check if this is the same WebSocket (not a new connection)
            if rt_services.active_bot_sessions[session_id] == websocket:
                del rt_services.active_bot_sessions[session_id]
                
                # Only disconnect Realtime API if no other WebSocket is using it
                # (In case of reconnection, a new WebSocket might have already been added)
                if session_id not in rt_services.active_bot_sessions and session_id in active_realtime_sessions:
                    try:
                        await active_realtime_sessions[session_id].disconnect()
                    except Exception:
                        pass
                    finally:
                        # Double-check session still exists before deleting
                        if session_id in active_realtime_sessions:
                            del active_realtime_sessions[session_id]


async def _initialize_realtime_session(
    session_id: str, 
    websocket: WebSocket,
    current_transcript_ref: list,
    current_ai_response_ref: list
):
    """Initialize Realtime API session for a given session_id."""
    session_metadata = rt_services.bot_session_metadata.get(session_id, {})
    meeting_id = session_metadata.get("meeting_id", session_id)
    meeting_context = {
        "meeting_id": meeting_id,
        "session_id": session_id,
        "participants": session_metadata.get("participants", []),
        "topic": session_metadata.get("topic", "General meeting")
    }
    
    # Initialize Realtime API service
    realtime_service = RealtimeAPIService()
    
    # Set up event handlers
    # Note: Handlers always use the current WebSocket from active_bot_sessions
    # This allows reconnection without recreating the Realtime API session
    async def on_audio_delta(audio_bytes: bytes):
        """Handle audio output from Realtime API."""
        # Always get the current WebSocket (handles reconnections)
        current_ws = rt_services.active_bot_sessions.get(session_id)
        if current_ws:
            try:
                # Send audio bytes directly to frontend
                asyncio.create_task(current_ws.send_bytes(audio_bytes))
            except Exception as e:
                logger.error(f"Error sending audio: {e}")
    
    async def on_transcript_delta(transcript_delta: str):
        """Handle real-time transcription and send to frontend."""
        # Update transcript reference
        if not current_transcript_ref:
            current_transcript_ref.append("")
        
        # Handle full transcript or delta
        if len(transcript_delta) > len(current_transcript_ref[0]):
            current_transcript_ref[0] = transcript_delta
        else:
            current_transcript_ref[0] += transcript_delta
        
        # Send transcription to frontend
        # Always get the current WebSocket (handles reconnections)
        current_ws = rt_services.active_bot_sessions.get(session_id)
        if current_ws:
            try:
                asyncio.create_task(current_ws.send_text(json.dumps({
                    "type": "transcription",
                    "session_id": session_id,
                    "meeting_id": meeting_id,
                    "content": current_transcript_ref[0]
                })))
            except Exception as e:
                logger.error(f"Error sending transcript: {e}")
    
    async def on_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool calls."""
        return await handle_tool_call(tool_name, arguments)
    
    async def on_response_done():
        """Handle response completion."""
        # Send completion message to frontend
        # Always get the current WebSocket (handles reconnections)
        current_ws = rt_services.active_bot_sessions.get(session_id)
        if current_ws:
            try:
                asyncio.create_task(current_ws.send_text(json.dumps({
                    "type": "tts_complete",
                    "session_id": session_id
                })))
            except Exception as e:
                logger.error(f"Error sending completion: {e}")
    
    async def on_error(error_message: str):
        """Handle errors."""
        logger.error(f"Realtime API error: {error_message}")
    
    # Set handlers
    realtime_service.on_audio_delta = on_audio_delta
    realtime_service.on_transcript_delta = on_transcript_delta
    realtime_service.on_tool_call = on_tool_call
    realtime_service.on_response_done = on_response_done
    realtime_service.on_error = on_error
    
    # Connect to Realtime API
    connected = await realtime_service.connect(
        session_id=session_id,
        meeting_context=meeting_context,
        voice="alloy",  # Can be customized per user
        instructions=None,  # Uses default from service
        tools=None  # Uses default tools
    )
    
    if connected:
        active_realtime_sessions[session_id] = realtime_service
    else:
        logger.error(f"Failed to connect Realtime API for {session_id}")

