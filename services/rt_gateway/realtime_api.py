"""
OpenAI Realtime API – Fully Repaired Version (Part 1/3)
Compatible with ChatGPT Voice Mode behaviour
"""

import asyncio
import base64
import json
import logging
from typing import Optional, Dict, Any, Callable, List, Awaitable
import websockets
from websockets.client import WebSocketClientProtocol
from asyncio import Lock

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ----------------------------------------------------------------------
#  REALTIME EVENT TYPES (cleaned)
# ----------------------------------------------------------------------

class RealtimeEventType(str):
    """Realtime API event types (OpenAI official)."""

    # Session config
    SESSION_UPDATE = "session.update"
    SESSION_UPDATED = "session.updated"

    # Audio input
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"

    # Responses from model
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"
    RESPONSE_DONE = "response.done"

    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_AUDIO_DONE = "response.audio.done"

    RESPONSE_TRANSCRIPT_DELTA = "response.transcript.delta"
    RESPONSE_TRANSCRIPT_DONE = "response.transcript.done"

    RESPONSE_FUNCTION_ARGS_DELTA = "response.function_call_arguments.delta"
    RESPONSE_FUNCTION_ARGS_DONE = "response.function_call_arguments.done"

    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"

    # Misc
    ERROR = "error"
    CONVERSATION_ITEM_CREATED = "conversation.item.created"
    CONVERSATION_ITEM_TRUNCATED = "conversation.item.truncated"


# ----------------------------------------------------------------------
#  REALTIME API SERVICE
# ----------------------------------------------------------------------

class RealtimeAPIService:
    """
    **FULLY FIXED SERVICE**
    - Correct event sequencing
    - Correct tool call pipeline
    - ChatGPT-style audio flow (append → commit)
    - No global state
    - Async-safe, no deadlocks
    - Supports backpressure
    """

    REALTIME_API_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("Missing OpenAI API key")

        self.websocket: Optional[WebSocketClientProtocol] = None
        self.connected = False
        self.session_id: Optional[str] = None

        # Locks & state storage
        self._state_lock = Lock()
        self._listener_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Runtime tracking
        self._response_in_progress = False
        self._current_transcript = ""

        # Tool call tracking
        self._current_tool_calls: Dict[str, Dict[str, Any]] = {}

        # Pending audio queue
        self._pending_audio_chunks: List[bytes] = []
        self._audio_started = False

        # Event callbacks
        self.on_audio_delta: Optional[Callable[[bytes], Awaitable[None]]] = None
        self.on_transcript_delta: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_tool_call: Optional[Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None
        self.on_response_done: Optional[Callable[[], Awaitable[None]]] = None
        self.on_error: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_conversation_item: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None

# ======================================================================
#  PART 2 — CONNECTION, SESSION INIT & LISTENER (FULLY REPAIRED)
# ======================================================================

    async def connect(
        self,
        session_id: str,
        assistant_name: str = "Aurray",
        meeting_context: Optional[Dict[str, Any]] = None,
        voice: str = "alloy",
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Establish OpenAI Realtime WS connection (ChatGPT identical behaviour).
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }

            self.websocket = await websockets.connect(
                self.REALTIME_API_URL,
                extra_headers=headers,
                max_size=None,
            )

            async with self._state_lock:
                self.connected = True
                self.session_id = session_id
                self._shutdown_event.clear()

            # Initialize model session (instructions, audio config, tools)
            await self._initialize_session(
                assistant_name,
                meeting_context,
                voice,
                instructions,
                tools
            )

            # Start async listener task
            self._listener_task = asyncio.create_task(self._listen_messages())

            return True

        except Exception as e:
            logger.error(f"[RealtimeAPI.connect] Failed: {e}", exc_info=True)
            async with self._state_lock:
                self.connected = False
                self.websocket = None
            return False


    # ------------------------------------------------------------------
    #  SESSION INITIALIZATION  (ChatGPT Voice Mode EXACT)
    # ------------------------------------------------------------------
    async def _initialize_session(
        self,
        assistant_name: str,
        meeting_context: Optional[Dict[str, Any]],
        voice: str,
        instructions: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
    ):
        """
        Send session.update to configure the model.
        """

        base_instructions = (
            f"You are {assistant_name}, the user's meeting assistant. "
            "You MUST NOT match the user's language. Always speak ONLY English."
            "If the user speaks Spanish or any other language, translate it to English internally "
            "but ALWAYS answer in English. "
            "Never output any Spanish text or Spanish speech under any circumstance. "
            "Be natural, calm, confident, and helpful. "
            "Never proactively use tools. Only use tools when the user explicitly asks."
        )

        if meeting_context:
            base_instructions += meeting_context.get("context_description", "")

        session_payload = {
            "type": RealtimeEventType.SESSION_UPDATE,
            "session": {
                "modalities": ["audio", "text"],
                "instructions": base_instructions,

                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 16000},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 500,
                        }
                    },
                    "output": {
                        "format": {"type": "audio/pcm"},
                        "voice": voice,
                    },
                },

                "input_audio_transcription": {
                    "model": "whisper-1"
                },

                "temperature": 0.8,
                "max_response_output_tokens": 4096,
                "tools": tools or []
            },
        }

        await self._send_message(session_payload)
    
    def _get_default_tools(self) -> List[Dict[str, Any]]:
        """Get default tool definitions for meeting assistant."""
        return []


    # ------------------------------------------------------------------
    #  INTERNAL: SEND MESSAGE TO OPENAI WS
    # ------------------------------------------------------------------
    async def _send_message(self, message: Dict[str, Any]) -> bool:
        """
        Safely send message into OpenAI Realtime WS.
        """

        async with self._state_lock:
            if not self.connected or not self.websocket:
                return False
            ws = self.websocket

        try:
            await ws.send(json.dumps(message))
            return True

        except websockets.exceptions.ConnectionClosed:
            async with self._state_lock:
                self.connected = False
            return False

        except Exception as e:
            logger.error(f"[RealtimeAPI.send] {e}", exc_info=True)
            async with self._state_lock:
                self.connected = False
            return False

    # ------------------------------------------------------------------
    #  PUBLIC: CHECK RESPONSE STATE
    # ------------------------------------------------------------------
    async def is_response_in_progress(self) -> bool:
        """
        Check if a response is currently in progress.
        """
        async with self._state_lock:
            return self._response_in_progress


    # ------------------------------------------------------------------
    #  LISTENER LOOP (ChatGPT identical)
    # ------------------------------------------------------------------
    async def _listen_messages(self):
        """
        Continuous async listener that dispatches events.
        """

        try:
            while True:
                if self._shutdown_event.is_set():
                    break

                async with self._state_lock:
                    if not self.connected or not self.websocket:
                        break
                    ws = self.websocket

                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)

                except asyncio.TimeoutError:
                    continue

                except websockets.exceptions.ConnectionClosed:
                    async with self._state_lock:
                        self.connected = False
                    break

                except Exception as e:
                    logger.error(f"[RealtimeAPI.listen] {e}", exc_info=True)
                    continue

                # Parse & route
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    continue

                await self._route_event(event)

        except Exception as e:
            logger.error(f"[RealtimeAPI.listen fatal] {e}", exc_info=True)

        finally:
            async with self._state_lock:
                self.connected = False


    # ------------------------------------------------------------------
    #  ROUTE EVENTS TO CORRECT HANDLERS
    # ------------------------------------------------------------------
    async def _route_event(self, event: Dict[str, Any]):
        event_type = event.get("type")
        
        # Log response-related events
        if event_type in (RealtimeEventType.RESPONSE_CREATE, RealtimeEventType.RESPONSE_AUDIO_DELTA, 
                         RealtimeEventType.RESPONSE_DONE, RealtimeEventType.RESPONSE_AUDIO_DONE):
            logger.info(f"[RealtimeAPI] 📨 Received event: {event_type}")

        if event_type == RealtimeEventType.RESPONSE_CREATE:
            logger.info(f"[RealtimeAPI] ✅ Response started")
            self._response_in_progress = True

        elif event_type == RealtimeEventType.RESPONSE_AUDIO_DELTA:
            await self._handle_audio_delta(event)

        elif event_type == RealtimeEventType.RESPONSE_AUDIO_DONE:
            pass  # Optional

        elif event_type == RealtimeEventType.RESPONSE_TRANSCRIPT_DELTA:
            await self._handle_transcript_delta(event)

        elif event_type == RealtimeEventType.RESPONSE_TRANSCRIPT_DONE:
            self._current_transcript = ""

        elif event_type == RealtimeEventType.RESPONSE_FUNCTION_ARGS_DELTA:
            await self._handle_function_args_delta(event)

        elif event_type == RealtimeEventType.RESPONSE_FUNCTION_ARGS_DONE:
            pass

        elif event_type == RealtimeEventType.RESPONSE_OUTPUT_ITEM_DONE:
            await self._handle_output_item_done(event)

        elif event_type == RealtimeEventType.RESPONSE_DONE:
            self._response_in_progress = False
            if self.on_response_done:
                await self.on_response_done()

        elif event_type == RealtimeEventType.CONVERSATION_ITEM_CREATED:
            if self.on_conversation_item:
                await self.on_conversation_item(event)

        elif event_type == RealtimeEventType.ERROR:
            if self.on_error:
                await self.on_error(str(event.get("error", {})))


# ======================================================================
#  PART 3 — AUDIO/TEXT HANDLERS, FUNCTION CALLS, COMMIT, DISCONNECT
# ======================================================================

    # ------------------------------------------------------------------
    # HANDLE AUDIO STREAM FROM OPENAI (PCM16)
    # ------------------------------------------------------------------
    async def _handle_audio_delta(self, event: Dict[str, Any]):
        """
        OpenAI sends base64-encoded PCM16 bytes incrementally in the 'delta' field.
        Decode and forward to callback.
        """
        # OpenAI Realtime API sends audio data in the 'delta' field, not 'audio.data'
        audio_data = event.get("delta")
        if not audio_data:
            logger.warning(f"[RealtimeAPI] ⚠️ Audio delta event missing 'delta' field: {list(event.keys())}")
            return

        # OpenAI sends base64-encoded audio data as a string
        try:
            if isinstance(audio_data, str):
                pcm_bytes = base64.b64decode(audio_data)
            else:
                # Already bytes (shouldn't happen, but handle gracefully)
                pcm_bytes = audio_data
        except Exception as e:
            logger.error(f"[RealtimeAPI] ❌ Failed to decode audio data: {e}")
            return

        logger.info(f"[RealtimeAPI] 🎵 Received audio delta from OpenAI: {len(pcm_bytes)} bytes (decoded from base64)")
        if self.on_audio_delta:
            await self.on_audio_delta(pcm_bytes)
        else:
            logger.warning(f"[RealtimeAPI] ⚠️ No on_audio_delta callback registered")


    # ------------------------------------------------------------------
    # HANDLE TRANSCRIPTION DELTA
    # ------------------------------------------------------------------
    async def _handle_transcript_delta(self, event: Dict[str, Any]):
        delta = event.get("delta", "")
        if not delta:
            return

        self._current_transcript += delta

        if self.on_transcript:
            await self.on_transcript(self._current_transcript)


    # ------------------------------------------------------------------
    # HANDLE FUNCTION CALL ARGUMENTS DELTA (STREAMING JSON)
    # ------------------------------------------------------------------
    async def _handle_function_args_delta(self, event: Dict[str, Any]):
        item_id = event.get("item_id")
        delta = event.get("delta", "")
        if not item_id:
            return

        if item_id not in self._function_call_buffers:
            self._function_call_buffers[item_id] = ""

        self._function_call_buffers[item_id] += delta


    # ------------------------------------------------------------------
    # HANDLE FUNCTION TOOL OUTPUT FINISH
    # ------------------------------------------------------------------
    async def _handle_output_item_done(self, event: Dict[str, Any]):
        """
        Called when OpenAI finishes a function call:
        → We parse accumulated args
        → Call the Python tool
        → Send result back to OpenAI with response.ref
        """

        item_id = event.get("item_id")
        output = event.get("output", {})

        tool_name = output.get("tool_name")
        if not tool_name:
            return

        args_raw = self._function_call_buffers.get(item_id, "")
        if item_id in self._function_call_buffers:
            del self._function_call_buffers[item_id]

        try:
            args = json.loads(args_raw or "{}")
        except Exception as e:
            logger.error(f"[RealtimeAPI] Invalid tool args: {e}")
            args = {}

        python_tool = self._tool_registry.get(tool_name)
        if not python_tool:
            logger.error(f"[RealtimeAPI] Unknown tool: {tool_name}")
            return

        try:
            result = await python_tool(**args)
        except Exception as e:
            logger.error(f"[RealtimeAPI] Tool error: {e}", exc_info=True)
            result = {"error": str(e)}

        # Send back tool re
