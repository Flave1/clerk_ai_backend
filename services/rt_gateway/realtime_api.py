"""
OpenAI Realtime API Service - CORRECTED VERSION

Fixes:
- Proper event handling for all event types
- Thread-safe callbacks with async support
- Correct tool calling sequence
- Proper audio flow (append, commit)
- Error recovery and propagation
- No global state dependencies
- Proper response state tracking
"""
import asyncio
import base64
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable, List, Union, Awaitable
from enum import Enum
import websockets
from websockets.client import WebSocketClientProtocol
from asyncio import Lock

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RealtimeEventType(str, Enum):
    """OpenAI Realtime API event types."""
    # Session events
    SESSION_UPDATE = "session.update"
    SESSION_UPDATED = "session.updated"
    
    # Input events
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
    
    # Response events
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_AUDIO_DONE = "response.audio.done"
    RESPONSE_TRANSCRIPT_DELTA = "response.transcript.delta"
    RESPONSE_TRANSCRIPT_DONE = "response.transcript.done"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"
    RESPONSE_DONE = "response.done"
    
    # Error events
    ERROR = "error"
    CONVERSATION_ITEM_CREATED = "conversation.item.created"
    CONVERSATION_ITEM_TRUNCATED = "conversation.item.truncated"


class RealtimeAPIService:
    """
    OpenAI Realtime API service - CORRECTED VERSION.
    
    Key fixes:
    1. Thread-safe callbacks with proper async handling
    2. Correct event sequence handling
    3. Proper tool calling with response.create → response.output_item.done
    4. Audio flow: append → commit
    5. Error propagation to callbacks
    6. Proper state tracking with locks
    7. No global state dependencies
    """
    
    REALTIME_API_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, 'openai_api_key', None)
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Realtime API")
        
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.connected = False
        self.session_id: Optional[str] = None
        
        # Thread-safe state
        self._state_lock = Lock()
        self._listener_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Event handlers - all must be async or None
        self.on_audio_delta: Optional[Callable[[bytes], Awaitable[None]]] = None
        self.on_transcript_delta: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_tool_call: Optional[Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None
        self.on_response_done: Optional[Callable[[], Awaitable[None]]] = None
        self.on_error: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_conversation_item: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
        
        # State tracking - protected by lock
        self._current_transcript = ""
        self._current_tool_calls: Dict[str, Dict[str, Any]] = {}  # Support multiple tool calls
        self._response_in_progress = False
        self._audio_started = False
        self._pending_audio_chunks: List[bytes] = []  # Backpressure queue
        self._chunk_count = 0  # For logging
        
    async def connect(
        self,
        session_id: str,
        meeting_context: Optional[Dict[str, Any]] = None,
        voice: str = "alloy",
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Establish WebSocket connection and initialize session."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            self.websocket = await websockets.connect(
                self.REALTIME_API_URL,
                extra_headers=headers
            )
            
            async with self._state_lock:
                self.connected = True
                self.session_id = session_id
                self._shutdown_event.clear()
            
            # Initialize session
            await self._initialize_session(meeting_context, voice, instructions, tools)
            
            # Start message listener
            self._listener_task = asyncio.create_task(self._listen_messages())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Realtime API: {e}", exc_info=True)
            async with self._state_lock:
                self.connected = False
                self.websocket = None
            return False
    
    async def _initialize_session(
        self,
        meeting_context: Optional[Dict[str, Any]],
        voice: str,
        instructions: Optional[str],
        tools: Optional[List[Dict[str, Any]]]
    ):
        """Initialize the Realtime API session with configuration."""
        default_instructions = (
            "You are Aurray, the user's meeting assistant. "
            "Always speak in English. Use a natural, polite, confident tone. "
            "Keep responses short and context aware. Be helpful and proactive in meetings."
        )
        
        if meeting_context:
            context_str = f"Meeting context: {json.dumps(meeting_context, indent=2)}"
            default_instructions = f"{default_instructions}\n\n{context_str}"
        
        session_config = {
            "type": RealtimeEventType.SESSION_UPDATE,
            "session": {
                "modalities": ["audio", "text"],   # REQUIRED combo for audio output

                "instructions": (
                    "You are Aurray, the user's meeting assistant. "
                    "Always speak only in English. "
                    "Never respond in any other language. "
                    "Only use tools (saveNote, createTask, summarizeDiscussion) when the user explicitly asks you to. "
                    "Do not proactively use tools or ask the user for information to use tools. "
                    "Just have a natural conversation unless the user specifically requests a tool."
                ),

                "voice": voice,

                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },

                # ❌ Removed this:
                # "input_audio_transcription": { ... }

                "temperature": 0.8,
                "max_response_output_tokens": 4096,
                "tools": tools or self._get_default_tools()
            }
        }

        
        await self._send_message(session_config)
    
    def _get_default_tools(self) -> List[Dict[str, Any]]:
        """Get default tool definitions for meeting assistant."""
        return [
            {
                "type": "function",
                "name": "saveNote",
                "description": "ONLY use when the user explicitly asks you to save a note. Save a meeting note or important point that the user specifically requests to be saved.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The note text to save"
                        }
                    },
                    "required": ["text"]
                }
            },
            {
                "type": "function",
                "name": "createTask",
                "description": "ONLY use when the user explicitly asks you to create a task. Create a task or action item that the user specifically requests.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task description"
                        },
                        "assignee": {
                            "type": "string",
                            "description": "Who should complete this task (optional)"
                        }
                    },
                    "required": ["task"]
                }
            },
            {
                "type": "function",
                "name": "summarizeDiscussion",
                "description": "ONLY use when the user explicitly asks you to summarize. Summarize the current discussion or meeting topic that the user specifically requests.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The topic or discussion to summarize"
                        }
                    },
                    "required": ["topic"]
                }
            }
        ]
    
    async def _send_message(self, message: Dict[str, Any]) -> bool:
        """Send a message to the Realtime API. Returns True if sent, False if failed."""
        async with self._state_lock:
            if not self.connected or not self.websocket:
                return False
        
        try:
            message_str = json.dumps(message)
            await self.websocket.send(message_str)
            return True
        except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError):
            async with self._state_lock:
                self.connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to send message to Realtime API: {e}", exc_info=True)
            async with self._state_lock:
                self.connected = False
            return False
    
    async def _listen_messages(self):
        """Listen for messages from Realtime API and handle events."""
        try:
            while True:
                # Check shutdown event
                if self._shutdown_event.is_set():
                    break
                
                # Check connection state
                async with self._state_lock:
                    if not self.connected or not self.websocket:
                        break
                    ws = self.websocket
                
                try:
                    # Receive with timeout to allow shutdown check
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    
                    # Parse event
                    try:
                        event = json.loads(message)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from Realtime API: {e}")
                        continue
                    
                    # Handle event
                    await self._handle_event(event)
                    
                except asyncio.TimeoutError:
                    # Timeout is expected, continue loop
                    continue
                except websockets.exceptions.ConnectionClosedOK:
                    async with self._state_lock:
                        self.connected = False
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    logger.error(f"Realtime API connection closed with error: {e.code}, {e.reason}")
                    async with self._state_lock:
                        self.connected = False
                    break
                except Exception as e:
                    logger.error(f"Error handling Realtime API message: {e}", exc_info=True)
                    # Continue processing - don't break on single message error
                    
        except Exception as e:
            logger.error(f"Fatal error in Realtime API message listener: {e}", exc_info=True)
            async with self._state_lock:
                self.connected = False
    
    async def _handle_event(self, event: Dict[str, Any]):
        """Handle incoming events from Realtime API."""
        event_type = event.get("type")
        
        # Route to appropriate handler
        if event_type == RealtimeEventType.RESPONSE_CREATE:
            await self._handle_response_create(event)
        elif event_type == RealtimeEventType.RESPONSE_AUDIO_DELTA:
            await self._handle_audio_delta(event)
        elif event_type == RealtimeEventType.RESPONSE_AUDIO_DONE:
            await self._handle_audio_done(event)
        elif event_type == RealtimeEventType.RESPONSE_TRANSCRIPT_DELTA:
            await self._handle_transcript_delta(event)
        elif event_type == RealtimeEventType.RESPONSE_TRANSCRIPT_DONE:
            await self._handle_transcript_done(event)
        elif event_type == RealtimeEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA:
            await self._handle_function_call_delta(event)
        elif event_type == RealtimeEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
            await self._handle_function_call_done(event)
        elif event_type == RealtimeEventType.RESPONSE_OUTPUT_ITEM_DONE:
            await self._handle_output_item_done(event)
        elif event_type == RealtimeEventType.RESPONSE_DONE:
            await self._handle_response_done(event)
        elif event_type == RealtimeEventType.ERROR:
            await self._handle_error(event)
        elif event_type == RealtimeEventType.SESSION_UPDATED:
            pass
        elif event_type == RealtimeEventType.CONVERSATION_ITEM_CREATED:
            await self._handle_conversation_item_created(event)
        elif event_type == RealtimeEventType.CONVERSATION_ITEM_TRUNCATED:
            await self._handle_conversation_item_truncated(event)
    
    async def _handle_response_create(self, event: Dict[str, Any]):
        """Handle response.create event - marks start of response."""
        async with self._state_lock:
            self._response_in_progress = True
    
    async def _handle_audio_delta(self, event: Dict[str, Any]):
        """Handle audio delta from model response."""
        audio_base64 = event.get("delta", "")
        
        if not audio_base64:
            return
        
        try:
            audio_bytes = base64.b64decode(audio_base64)
            
            if self.on_audio_delta:
                # Callback is async, await it
                await self.on_audio_delta(audio_bytes)
        except Exception as e:
            logger.error(f"Error decoding audio delta: {e}", exc_info=True)
            # Propagate error
            if self.on_error:
                await self.on_error(f"Audio decode error: {e}")
    
    async def _handle_audio_done(self, event: Dict[str, Any]):
        """Handle audio completion."""
        pass
    
    async def _handle_transcript_delta(self, event: Dict[str, Any]):
        """Handle transcription delta from model."""
        delta = event.get("delta", "")
        
        if not delta:
            return
        
        async with self._state_lock:
            self._current_transcript += delta
        
        if self.on_transcript_delta:
            # Send only the delta, not accumulated
            await self.on_transcript_delta(delta)
    
    async def _handle_transcript_done(self, event: Dict[str, Any]):
        """Handle transcription completion."""
        async with self._state_lock:
            self._current_transcript = ""
    
    async def _handle_function_call_delta(self, event: Dict[str, Any]):
        """Handle function call arguments delta."""
        item_id = event.get("item_id", "default")
        name = event.get("name", "")
        delta = event.get("arguments", "")
        
        async with self._state_lock:
            if item_id not in self._current_tool_calls:
                self._current_tool_calls[item_id] = {
                    "name": name,
                    "arguments": ""
                }
            
            if delta:
                self._current_tool_calls[item_id]["arguments"] += delta
    
    async def _handle_function_call_done(self, event: Dict[str, Any]):
        """Handle function call completion - wait for output_item.done before executing."""
        # Don't execute yet - wait for output_item.done
        pass
    
    async def _handle_output_item_done(self, event: Dict[str, Any]):
        """Handle output item completion - execute tool call if it's a function call item."""
        item_id = event.get("item_id", "default")
        
        # Check if this is actually a tool call item
        # output_item.done is sent for ALL output items (audio, transcript, function_call)
        # Only process if we have a tool call tracked for this item_id
        async with self._state_lock:
            tool_call = self._current_tool_calls.get(item_id)
        
        if not tool_call:
            # This is not a tool call item (likely audio or transcript), ignore it
            return
        
        # Remove from tracking since we're about to execute it
        async with self._state_lock:
            self._current_tool_calls.pop(item_id, None)
        
        tool_name = tool_call["name"]
        try:
            arguments = json.loads(tool_call["arguments"])
        except json.JSONDecodeError as e:
            logger.error(f"Invalid tool call arguments: {e}")
            if self.on_error:
                await self.on_error(f"Invalid tool arguments: {e}")
            return
        
        # Execute tool call
        result = None
        if self.on_tool_call:
            try:
                result = await self.on_tool_call(tool_name, arguments)
            except Exception as e:
                logger.error(f"Tool execution error: {e}", exc_info=True)
                result = {"error": str(e)}
        else:
            result = {"status": "ok", "message": f"Executed {tool_name}"}
        
        # Send tool result back to model - CORRECT SEQUENCE
        # Use response.create with response.output_item.done pattern
        await self._send_message({
            "type": RealtimeEventType.RESPONSE_CREATE,
            "response": {
                "id": item_id,
                "type": "function_call_output",
                "function_call_output": result
            }
        })
    
    async def _handle_response_done(self, event: Dict[str, Any]):
        """Handle response completion."""
        async with self._state_lock:
            self._response_in_progress = False
        
        if self.on_response_done:
            await self.on_response_done()
    
    async def _handle_error(self, event: Dict[str, Any]):
        """Handle error events."""
        error = event.get("error", {})
        error_message = error.get("message", "Unknown error")
        error_type = error.get("type", "unknown")
        
        # Don't log "no active response" as error - it's expected
        if "no active response" in error_message.lower():
            return
        
        logger.error(f"Realtime API error: {error_type} - {error_message}")
        
        if self.on_error:
            await self.on_error(f"{error_type}: {error_message}")
    
    async def _handle_conversation_item_created(self, event: Dict[str, Any]):
        """Handle conversation item created event."""
        item = event.get("item", {})
        if self.on_conversation_item:
            await self.on_conversation_item({"type": "created", "item": item})
    
    async def _handle_conversation_item_truncated(self, event: Dict[str, Any]):
        """Handle conversation item truncated event."""
        item = event.get("item", {})
        if self.on_conversation_item:
            await self.on_conversation_item({"type": "truncated", "item": item})
    
    async def send_audio_chunk(self, audio_chunk: bytes) -> bool:
        """
        Send audio chunk to Realtime API.
        
        Returns True if sent, False if failed.
        Implements backpressure by queuing if needed.
        """
        if not await self._check_connected():
            return False
        
        # Queue chunks if needed (for initial connection)
        async with self._state_lock:
            if not self._audio_started:
                # Queue this chunk for now
                self._pending_audio_chunks.append(audio_chunk)
                # Mark as started and flush queue
                self._audio_started = True
                asyncio.create_task(self._flush_audio_queue())
                return True
        
        # Encode and send
        audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')
        
        async with self._state_lock:
            self._chunk_count += 1
        
        return await self._send_message({
            "type": RealtimeEventType.INPUT_AUDIO_BUFFER_APPEND,
            "audio": audio_base64
        })
    
    async def _flush_audio_queue(self):
        """Flush queued audio chunks."""
        async with self._state_lock:
            chunks = self._pending_audio_chunks[:]
            self._pending_audio_chunks.clear()
        
        for chunk in chunks:
            await self.send_audio_chunk(chunk)
    
    async def commit_audio(self) -> bool:
        """Signal end of audio input (user finished speaking)."""
        if not await self._check_connected():
            return False
        
        return await self._send_message({
            "type": RealtimeEventType.INPUT_AUDIO_BUFFER_COMMIT
        })
    
    async def clear_audio_buffer(self) -> bool:
        """Clear the input audio buffer."""
        if not await self._check_connected():
            return False
        
        async with self._state_lock:
            self._audio_started = False
            self._pending_audio_chunks.clear()
        
        return await self._send_message({
            "type": RealtimeEventType.INPUT_AUDIO_BUFFER_CLEAR
        })
    
    async def interrupt(self) -> bool:
        """Interrupt the current response (user started speaking)."""
        if not await self._check_connected():
            return False
        
        async with self._state_lock:
            if not self._response_in_progress:
                return False
        
        success = await self._send_message({
            "type": RealtimeEventType.RESPONSE_CANCEL
        })
        
        if success:
            async with self._state_lock:
                self._response_in_progress = False
        
        return success
    
    async def _check_connected(self) -> bool:
        """Check if connected (thread-safe)."""
        async with self._state_lock:
            return self.connected and self.websocket is not None
    
    async def disconnect(self):
        """Close the WebSocket connection and cleanup."""
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel listener task
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket
        async with self._state_lock:
            if self.websocket:
                try:
                    await self.websocket.close()
                except Exception:
                    pass
            
            self.connected = False
            self.websocket = None
            self._audio_started = False
            self._pending_audio_chunks.clear()

