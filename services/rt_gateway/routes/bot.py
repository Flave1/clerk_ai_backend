"""
Aurray Voice Backend — ChatGPT Voice Mode Compatible
Author: Fully Repaired Version
"""

import asyncio
import json
import logging
import struct
import time
import base64

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..realtime_api import RealtimeAPIService
from .. import services as rt_services

logger = logging.getLogger(__name__)

# HTTP router (empty - bot only uses WebSocket routes)
router = APIRouter()

# WebSocket router
ws_router = APIRouter()

# Active Realtime API sessions
active_realtime_sessions = {}

# --- Utility: RMS ----------------------------------------------------------

def calculate_audio_rms(audio_bytes: bytes) -> float:
    """Return normalized RMS for PCM16 audio."""
    if len(audio_bytes) < 2:
        return 0.0
    
    samples = struct.unpack(f"<{len(audio_bytes)//2}h", audio_bytes)
    sum_sq = sum(s * s for s in samples)
    rms = (sum_sq / len(samples)) ** 0.5
    
    return min(rms / 32768.0, 1.0)
    

# ======================================================================
# ==  MAIN WEBSOCKET HANDLER  ==========================================
# ======================================================================

@ws_router.websocket("/bot/{session_id}")
async def bot_websocket(websocket: WebSocket, session_id: str):
    """
    FULL ChatGPT Voice Mode clone:
    - Single unified WebSocket
    - Audio streamed as raw PCM16 (Float32 → Int16 → bytes)
    - Automatic turn-taking
    - Automatic commit based on server VAD OR fallback backend VAD
    - Interrupt handled instantly
    - No premature commits
    """
    await websocket.accept()
    
    # Store frontend WebSocket
    rt_services.active_bot_sessions[session_id] = websocket
    # Only initialize if not already set (don't overwrite existing meeting_context from conversations.py)
    if session_id not in rt_services.bot_session_metadata:
        rt_services.bot_session_metadata[session_id] = {}

    
    # ChatGPT-like state
    buffer_started_at = None
    last_voice_ts = None
    voice_detected = False
    RMS_THRESHOLD = 0.005
    MIN_MS = 100
    GRACE_MS = 300
    
    try:
        # ==================================================================
        # STEP 1 — WAIT FOR REGISTER MESSAGE
        # ==================================================================
        logger.info(f"[AURRAY] Waiting for registration message from {session_id}")
        while True:
            msg = await websocket.receive()
            logger.info(f"[AURRAY] Registration loop: received data with keys={list(msg.keys())}")

            if "text" in msg:
                try:
                    body = json.loads(msg["text"])
                    logger.info(f"[AURRAY] Registration loop: parsed JSON, type={body.get('type')}")
                    # Ensure body is a dictionary
                    if not isinstance(body, dict):
                        logger.warning(f"Expected dict in registration, got {type(body).__name__}: {body}")
                        continue
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Failed to parse registration JSON: {e}, data: {msg.get('text', '')[:100]}")
                    continue
                    
                if body.get("type") in ("register", "bot_registration"):
                    logger.info(f"[AURRAY] ✅ Registration received for {session_id}, sending connected response")
                    await websocket.send_text(json.dumps({
                        "type": "connected",
                        "session_id": session_id
                    }))
                    logger.info(f"[AURRAY] ✅ Connected response sent for {session_id}")
                    break
            else:
                logger.warning(f"[AURRAY] Registration loop: received non-text data: {list(msg.keys())}")

        # ==================================================================
        # STEP 2 — INIT REALTIME SESSION
        # ==================================================================
        logger.info(f"[AURRAY] Checking for existing Realtime session for {session_id}")
        rt_session = active_realtime_sessions.get(session_id)
        
        if not rt_session:
            logger.info(f"[AURRAY] Initializing new Realtime session for {session_id}")
            await _init_rt_session(session_id)
            rt_session = active_realtime_sessions.get(session_id)
            
            if not rt_session:
                logger.error(f"[AURRAY] ❌ Failed to initialize Realtime session for {session_id}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Failed to initialize Realtime API session"
                }))
                return
            else:
                logger.info(f"[AURRAY] ✅ Realtime session initialized successfully for {session_id}")
        else:
            logger.info(f"[AURRAY] ✅ Using existing Realtime session for {session_id}")

        logger.info(f"[AURRAY] Realtime session ready for {session_id}")

        # ==================================================================
        # STEP 3 — MAIN LOOP
        # ==================================================================
        logger.info(f"[AURRAY] Starting main loop for {session_id}")
        chunk_count = 0
        text_count = 0
        while True:
            try:
                data = await websocket.receive()
            except WebSocketDisconnect:
                logger.info(f"[AURRAY] WebSocket disconnected for {session_id}")
                break
            except Exception as e:
                logger.error(f"[AURRAY] WS receive error: {e}", exc_info=True)
                break
            
            # Log received data structure (first 50 chunks, then every 50th)
            chunk_count += 1
            if chunk_count <= 50 or chunk_count % 50 == 0:
                logger.info(f"[AURRAY] 📦 Received data #{chunk_count}: keys={list(data.keys())}, has_bytes={'bytes' in data}, has_text={'text' in data}, data_type={type(data)}")
                if "bytes" not in data and "text" not in data:
                    logger.warning(f"[AURRAY] ⚠️ Unexpected data format: {data}")
                
            # -------------------------------------------------------------
            # A) AUDIO CHUNK
            # -------------------------------------------------------------
            # FastAPI WebSocket: binary data comes as {"bytes": b"..."}
            # Also check for direct bytes or other formats
            if "bytes" in data:
                chunk = data["bytes"]
                rms = calculate_audio_rms(chunk)
                
                if chunk_count <= 50 or chunk_count % 50 == 0:
                    logger.info(f"[AURRAY] 📥 Audio chunk #{chunk_count}: {len(chunk)} bytes, RMS: {rms:.4f}, voice_detected={voice_detected}, buffer_started={buffer_started_at is not None}")

                now = time.time()

                # First chunk starts buffer
                if buffer_started_at is None:
                    buffer_started_at = now

                # Detect actual voice
                if rms > RMS_THRESHOLD:
                    voice_detected = True
                    last_voice_ts = now

                # Check if response is in progress - if so, only forward if user voice detected (for interruption)
                is_response_in_progress = await rt_session.is_response_in_progress()
                should_forward = not is_response_in_progress or (rms > RMS_THRESHOLD)
                
                if not should_forward:
                    # Skip forwarding audio when bot is responding and no user voice detected
                    # This prevents bot's own voice from being sent back (feedback loop)
                    if chunk_count <= 10 or chunk_count % 50 == 0:
                        logger.info(f"[AURRAY] ⏭️ Skipping audio chunk #{chunk_count}: response in progress and no user voice detected")
                    continue

                # Forward audio to OpenAI
                try:
                    if chunk_count <= 10 or chunk_count % 50 == 0:
                        logger.info(f"[AURRAY] 🔄 Forwarding audio chunk #{chunk_count} to OpenAI Realtime API (response_in_progress={is_response_in_progress}, user_voice={rms > RMS_THRESHOLD})")
                    await rt_session._send_message({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode()
                    })
                    if chunk_count <= 10 or chunk_count % 50 == 0:
                        logger.info(f"[AURRAY] ✅ Successfully forwarded audio chunk #{chunk_count} to OpenAI")
                except Exception as e:
                    logger.error(f"[AURRAY] ❌ Failed to forward audio chunk #{chunk_count}: {e}", exc_info=True)
                    break
                            
                continue

            # -------------------------------------------------------------
            # B) TEXT MESSAGE
            # -------------------------------------------------------------
            if "text" in data:
                text_count += 1
                if text_count <= 10 or text_count % 10 == 0:
                    logger.info(f"[AURRAY] 📨 Text message #{text_count} (chunk #{chunk_count}): {data.get('text', '')[:200]}")
                try:
                    msg = json.loads(data["text"])
                    # Ensure msg is a dictionary
                    if not isinstance(msg, dict):
                        logger.warning(f"[AURRAY] Expected dict, got {type(msg).__name__}: {msg}")
                        continue
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"[AURRAY] Failed to parse message JSON: {e}, data: {data.get('text', '')[:200]}")
                    continue
                
                t = msg.get("type")
                logger.info(f"[AURRAY] 📨 Parsed message type: {t}")

                # Ping/pong
                if t == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    continue

                # User interrupts bot
                if t in ("interrupt", "response.cancel"):
                    await rt_session._send_message({"type": "response.cancel"})
                    buffer_started_at = None
                    voice_detected = False
                    last_voice_ts = None
                    continue

                # MANUAL COMMIT
                if t in ("commit", "input_audio_buffer.commit"):
                    now = time.time()
                    logger.info(f"[AURRAY] 🔄 Processing commit request: voice_detected={voice_detected}, buffer_started={buffer_started_at is not None}, last_voice_ts={last_voice_ts}")

                    # No voice detected → ignore
                    if not voice_detected:
                        logger.info(f"[AURRAY] ⏭️ Commit skipped: no voice detected")
                        buffer_started_at = None
                        continue

                    # Too short
                    if buffer_started_at:
                        dur_ms = (now - buffer_started_at) * 1000
                        if dur_ms < MIN_MS:
                            logger.info(f"[AURRAY] ⏭️ Commit skipped: buffer too short ({dur_ms:.1f}ms < {MIN_MS}ms)")
                            continue
                            
                    # Grace window
                    if last_voice_ts:
                        since_last = (now - last_voice_ts) * 1000
                        if since_last < GRACE_MS:
                            logger.info(f"[AURRAY] ⏭️ Commit skipped: grace period not met ({since_last:.1f}ms < {GRACE_MS}ms)")
                            continue
                            
                    # COMMIT
                    logger.info(f"[AURRAY] ✅ Commit conditions met, calling _safe_commit")
                    await _safe_commit(rt_session, websocket, session_id)
                    logger.info(f"[AURRAY] ✅ Commit completed successfully")

                    buffer_started_at = None
                    voice_detected = False
                    last_voice_ts = None
                    continue

                if t == "end_stream":
                    break
                
    except Exception as e:
        logger.error(f"[AURRAY] WebSocket error: {e}", exc_info=True)

    finally:
        # Cleanup
        ws = rt_services.active_bot_sessions.get(session_id)
        if ws is websocket:
            rt_services.active_bot_sessions.pop(session_id, None)

        svc = active_realtime_sessions.get(session_id)
        if svc:
            try:
                await svc.disconnect()
            except:
                pass
            active_realtime_sessions.pop(session_id, None)



# ======================================================================
# == HELPERS ===========================================================
# ======================================================================

async def _safe_commit(rt_session, websocket, session_id):
    """Commit safely and trigger response.create only once."""
    try:
        logger.info(f"[AURRAY] 📤 Sending input_audio_buffer.commit to OpenAI")
        await rt_session._send_message({"type": "input_audio_buffer.commit"})
        logger.info(f"[AURRAY] ✅ input_audio_buffer.commit sent successfully")

        # Avoid double response.create
        is_in_progress = await rt_session.is_response_in_progress()
        logger.info(f"[AURRAY] 🔍 Response in progress check: {is_in_progress}")
        if not is_in_progress:
            logger.info(f"[AURRAY] 📤 Sending response.create to OpenAI")
            await rt_session._send_message({"type": "response.create"})
            logger.info(f"[AURRAY] ✅ response.create sent successfully")
        else:
            logger.info(f"[AURRAY] ⏭️ Skipping response.create: response already in progress")

    except Exception as e:
        logger.error(f"[AURRAY] ❌ Commit failed: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
        except:
            pass


async def _init_rt_session(session_id: str):
    """Initialize Realtime API connection for this session."""
    logger.info(f"[AURRAY] _init_rt_session: Starting initialization for {session_id}")
    svc = RealtimeAPIService()
    meeting_context = rt_services.bot_session_metadata.get(session_id, {})
    logger.info(f"[AURRAY] _init_rt_session: Meeting context keys: {list(meeting_context.keys())}")

    async def send_audio(audio_bytes: bytes):
        ws = rt_services.active_bot_sessions.get(session_id)
        if ws:
            logger.info(f"[AURRAY] 🔊 Forwarding audio from OpenAI to frontend: {len(audio_bytes)} bytes")
            await ws.send_bytes(audio_bytes)
            logger.info(f"[AURRAY] ✅ Audio forwarded successfully")
        else:
            logger.warning(f"[AURRAY] _init_rt_session: No WebSocket found for {session_id} when sending audio")

    async def send_transcript(delta: str):
        ws = rt_services.active_bot_sessions.get(session_id)
        if ws:
            await ws.send_text(json.dumps({
                    "type": "transcription",
                    "session_id": session_id,
                "content": delta
            }))
        else:
            logger.warning(f"[AURRAY] _init_rt_session: No WebSocket found for {session_id} when sending transcript")

    async def done():
        ws = rt_services.active_bot_sessions.get(session_id)
        if ws:
            await ws.send_text(json.dumps({
                    "type": "tts_complete",
                    "session_id": session_id
            }))
        else:
            logger.warning(f"[AURRAY] _init_rt_session: No WebSocket found for {session_id} when sending done")

    svc.on_audio_delta = send_audio
    svc.on_transcript_delta = send_transcript
    svc.on_response_done = done

    logger.info(f"[AURRAY] _init_rt_session: Calling svc.connect for {session_id}")
    ok = await svc.connect(
        session_id=session_id,
        assistant_name=meeting_context.get("name", "Aurray"),
        meeting_context=meeting_context,
        voice="alloy"
    )
    
    if ok:
        logger.info(f"[AURRAY] _init_rt_session: ✅ Realtime API connection successful for {session_id}")
        active_realtime_sessions[session_id] = svc
    else:
        logger.error(f"[AURRAY] _init_rt_session: ❌ RT session failed for {session_id}")
