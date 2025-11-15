"""
Optimized audio processing pipeline for real-time communication.
Designed for low latency, high responsiveness, and smooth user experience.
"""
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Tuple
from collections import deque

from shared.schemas import STTRequest, STTResponse
from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AudioProcessor:
    """
    Optimized audio processor for real-time voice communication.
    
    Key optimizations:
    - Lower buffer threshold (1.5s instead of 3s) for faster response
    - Adaptive cooldown based on conversation state
    - Improved VAD with better thresholds
    - Efficient buffer management with overlap
    """
    
    def __init__(
        self,
        stt_service,
        llm_service,
        tts_service,
        sample_rate: int = 16000,
        min_buffer_ms: int = 1200,  # Increased to capture more complete sentences (1.2s)
        max_buffer_ms: int = 4000,  # Increased max buffer for longer sentences
        vad_threshold: float = 0.0005,  # More lenient VAD to capture speech
        min_confidence: float = 0.5,  # Higher confidence to avoid responding to noise
        base_cooldown: float = 0.5,  # Slightly longer cooldown to allow complete sentences
        active_cooldown: float = 0.2,  # Minimal cooldown when user is active
    ):
        self.stt_service = stt_service
        self.llm_service = llm_service
        self.tts_service = tts_service
        self.sample_rate = sample_rate
        self.bytes_per_ms = (sample_rate * 2) // 1000  # 16-bit PCM
        self.min_buffer_size = min_buffer_ms * self.bytes_per_ms
        self.max_buffer_size = max_buffer_ms * self.bytes_per_ms
        self.vad_threshold = vad_threshold
        self.min_confidence = min_confidence
        self.base_cooldown = base_cooldown
        self.active_cooldown = active_cooldown
        
        # State
        self.audio_buffer = deque(maxlen=100)
        self.buffer_size = 0
        self.last_transcription_time = 0.0
        self.last_audio_activity_time = 0.0
        self.processing = False
        self.processing_lock = asyncio.Lock()
        self.consecutive_speech_detections = 0
    
    def add_audio_chunk(self, chunk: bytes):
        """Add audio chunk to buffer with efficient management."""
        if len(chunk) == 0:
            return
        
        chunk_size = len(chunk)
        self.audio_buffer.append(chunk)
        self.buffer_size += chunk_size
        
        # Update activity timestamp
        self.last_audio_activity_time = time.time()
        
        # Enforce max buffer size (trim from front)
        while self.buffer_size > self.max_buffer_size and len(self.audio_buffer) > 0:
            removed = self.audio_buffer.popleft()
            self.buffer_size -= len(removed)
    
    def get_buffer(self) -> bytes:
        """Get current audio buffer."""
        return b''.join(self.audio_buffer)
    
    def clear_buffer(self, keep_overlap: int = 8000):
        """Clear buffer, keeping overlap for context continuity."""
        if keep_overlap > 0 and self.buffer_size > keep_overlap:
            buffer = self.get_buffer()
            overlap = buffer[-keep_overlap:]
            self.audio_buffer.clear()
            self.audio_buffer.append(overlap)
            self.buffer_size = len(overlap)
        else:
            self.audio_buffer.clear()
            self.buffer_size = 0
    
    def has_sufficient_audio(self) -> bool:
        """Check if buffer has enough audio for processing."""
        return self.buffer_size >= self.min_buffer_size
    
    def has_meaningful_audio(self, audio_data: bytes) -> bool:
        """Quick amplitude check - extremely lenient to let STT VAD do the real work."""
        if not audio_data or len(audio_data) < 640:
            return False
        
        try:
            import struct
            sample_count = len(audio_data) // 2
            if sample_count == 0:
                return False
            
            # Fast amplitude check - sample more of the buffer for better accuracy
            sample_size = min(sample_count, 2000)  # Sample more for better accuracy
            pcm_format = "<" + "h" * sample_size
            samples = struct.unpack(pcm_format, audio_data[:sample_size * 2])
            abs_mean = sum(abs(s) for s in samples) / float(len(samples))
            normalized_mean = abs_mean / 32768.0
            
            # More reasonable threshold (0.0005 = 0.05%) - filter out very quiet noise but allow speech
            # This balances between rejecting silence and allowing actual speech
            threshold = 0.0005
            result = normalized_mean >= threshold
            if not result:
                logger.debug(f"Amplitude check failed: {normalized_mean:.6f} < {threshold}")
            return result
        except Exception as e:
            logger.debug(f"Amplitude check exception: {e}, defaulting to True")
            return True  # Default to processing if check fails
    
    async def detect_speech_activity(self, audio_data: bytes) -> bool:
        """Detect speech activity using optimized VAD pipeline - very lenient."""
        # Check amplitude first - if there's ANY signal, we'll be lenient
        has_any_signal = self.has_meaningful_audio(audio_data)
        
        # Use STT service VAD if available (more sophisticated)
        if self.stt_service and hasattr(self.stt_service, 'detect_speech_activity'):
            try:
                result = self.stt_service.detect_speech_activity(audio_data, sample_rate=self.sample_rate)
                if result:
                    self.consecutive_speech_detections += 1
                    logger.debug(f"STT VAD: Speech detected (consecutive: {self.consecutive_speech_detections})")
                    return True
                else:
                    # STT VAD said no speech - trust it more, but if buffer is very large (2x) and we have signal,
                    # allow processing (might be quiet speech or background noise that STT can filter)
                    if has_any_signal and len(audio_data) >= self.min_buffer_size * 2.0:
                        logger.debug(f"STT VAD: No speech, but allowing due to signal presence and very large buffer (2x)")
                        self.consecutive_speech_detections += 1
                        return True
                    self.consecutive_speech_detections = 0
                    logger.debug(f"STT VAD: No speech detected")
                    return False
            except Exception as e:
                logger.warning(f"STT VAD failed: {e}, falling back to amplitude check", exc_info=True)
                # Fall through to amplitude check if STT VAD fails
        
        # Fallback: Use amplitude check if STT VAD not available or failed
        if has_any_signal:
            # Any signal detected, assume speech (very lenient)
            self.consecutive_speech_detections += 1
            logger.debug(f"Amplitude check passed (STT VAD not available), assuming speech")
            return True
        else:
            logger.debug(f"Amplitude check failed, no speech detected")
            return False
    
    def can_process(self) -> bool:
        """
        Adaptive cooldown: shorter when user is actively speaking.
        This allows rapid follow-up questions and natural conversation flow.
        """
        current_time = time.time()
        time_since_last = current_time - self.last_transcription_time
        time_since_activity = current_time - self.last_audio_activity_time
        
        # If user is actively speaking (audio received recently), use shorter cooldown
        if time_since_activity < 2.0:  # Audio activity within last 2 seconds
            cooldown = self.active_cooldown
        else:
            cooldown = self.base_cooldown
        
        can_process = time_since_last >= cooldown
        if not can_process:
            logger.debug(f"Cooldown active: {time_since_last:.2f}s / {cooldown:.2f}s")
        
        return can_process
    
    async def process_audio(
        self,
        session_id: str,
        meeting_id: str,
        conversation_context: list
    ) -> Optional[Tuple[str, str]]:
        """
        Process audio buffer and return (transcribed_text, llm_response).
        Optimized for low latency and real-time responsiveness.
        """
        async with self.processing_lock:
            logger.info(f"[{datetime.now(timezone.utc).isoformat()}] 🔒 Processing lock acquired | Buffer: {self.buffer_size} bytes")
            
            if self.processing:
                logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ⏭️ Already processing, skipping")
                return None
            
            if not self.has_sufficient_audio():
                logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ⏭️ Insufficient audio: {self.buffer_size}/{self.min_buffer_size} bytes")
                return None
            
            if not self.can_process():
                time_since_last = time.time() - self.last_transcription_time
                logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ⏭️ Cooldown active: {time_since_last:.2f}s")
                return None
            
            audio_data = self.get_buffer()
            
            # VAD check with detailed logging and bypass for large buffers
            vad_result = await self.detect_speech_activity(audio_data)
            
            # Bypass VAD ONLY if:
            # 1. Buffer is very large (2x minimum)
            # 2. We haven't processed recently (3+ seconds)
            # 3. There's ACTUAL audio signal (not complete silence)
            # This prevents processing silence just because buffer is large
            time_since_last = time.time() - self.last_transcription_time
            has_signal = self.has_meaningful_audio(audio_data)
            large_buffer_bypass = (
                len(audio_data) >= (self.min_buffer_size * 2) 
                and time_since_last > 3.0
                and has_signal  # CRITICAL: Only bypass if there's actual audio signal
            )
            
            if not vad_result and not large_buffer_bypass:
                # Log amplitude for debugging
                try:
                    import struct
                    sample_count = len(audio_data) // 2
                    if sample_count > 0:
                        sample_size = min(sample_count, 2000)
                        pcm_format = "<" + "h" * sample_size
                        samples = struct.unpack(pcm_format, audio_data[:sample_size * 2])
                        abs_mean = sum(abs(s) for s in samples) / float(len(samples))
                        normalized_mean = abs_mean / 32768.0
                        logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ⏭️ No speech detected (VAD) | Amplitude: {normalized_mean:.6f} | Buffer: {len(audio_data)} bytes, clearing buffer")
                    else:
                        logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ⏭️ No speech detected (VAD), clearing buffer")
                except Exception:
                    logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ⏭️ No speech detected (VAD), clearing buffer")
                self.clear_buffer()
                return None
            elif large_buffer_bypass:
                # Log signal strength for debugging
                try:
                    import struct
                    sample_count = len(audio_data) // 2
                    if sample_count > 0:
                        sample_size = min(sample_count, 2000)
                        pcm_format = "<" + "h" * sample_size
                        samples = struct.unpack(pcm_format, audio_data[:sample_size * 2])
                        abs_mean = sum(abs(s) for s in samples) / float(len(samples))
                        normalized_mean = abs_mean / 32768.0
                        logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ⚠️ VAD bypass: Large buffer ({len(audio_data)} bytes) with signal (amplitude: {normalized_mean:.6f}) and no processing for {time_since_last:.1f}s - processing anyway")
                    else:
                        logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ⚠️ VAD bypass: Large buffer ({len(audio_data)} bytes) and no processing for {time_since_last:.1f}s - processing anyway")
                except Exception:
                    logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ⚠️ VAD bypass: Large buffer ({len(audio_data)} bytes) and no processing for {time_since_last:.1f}s - processing anyway")
            
            # VAD passed - proceed with processing
            self.processing = True
            self.last_transcription_time = time.time()
            
            try:
                # STT transcription
                transcribed_text = None
                stt_start = time.time()
                
                if self.stt_service:
                    try:
                        stt_request = STTRequest(
                            audio_data=audio_data,
                            conversation_id=session_id,
                            turn_number=len(conversation_context) + 1,
                            language="en"
                        )
                        stt_result = await self.stt_service.transcribe(stt_request, model="deepgram")
                        
                        if stt_result and stt_result.text and stt_result.text.strip():
                            confidence = getattr(stt_result, 'confidence', 1.0) or 1.0
                            if confidence >= self.min_confidence:
                                transcribed_text = stt_result.text.strip()
                                stt_elapsed = (time.time() - stt_start) * 1000
                                logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ✅ STT: '{transcribed_text[:50]}...' | Confidence: {confidence:.2f} | Time: {stt_elapsed:.1f}ms")
                            else:
                                logger.warning(f"[{datetime.now(timezone.utc).isoformat()}] ⚠️ STT returned low confidence: {confidence:.2f} < {self.min_confidence} - skipping LLM")
                        else:
                            # STT returned empty text - this means no speech was detected
                            logger.warning(f"[{datetime.now(timezone.utc).isoformat()}] ⚠️ STT returned empty text (no speech detected) - this should not happen if VAD worked correctly")
                    except Exception as e:
                        logger.error(f"STT failed: {e}", exc_info=True)
                
                if not transcribed_text:
                    logger.warning(f"[{datetime.now(timezone.utc).isoformat()}] ⚠️ No transcription available - skipping LLM to prevent incorrect responses to silence/noise")
                    # Clear buffer completely when no transcription (likely silence/noise)
                    self.clear_buffer(keep_overlap=0)
                    return None
                
                # Check if transcription is complete enough to respond to
                # Avoid responding to incomplete questions/single words
                words = transcribed_text.strip().split()
                word_count = len(words)
                
                # Incomplete indicators:
                # - Less than 3 words (likely incomplete)
                # - Ends with incomplete question words ("what is", "how", "why", "when", "where", "who")
                # - Single word responses
                incomplete_question_words = {"what", "how", "why", "when", "where", "who", "which", "whose"}
                is_incomplete = False
                
                if word_count < 3:
                    # Check if it's a single word or very short phrase
                    if word_count == 1:
                        is_incomplete = True
                        logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ⏸️ Incomplete transcription (single word): '{transcribed_text}' - waiting for more audio")
                    elif word_count == 2:
                        # Check if it ends with an incomplete question word
                        last_word = words[-1].lower().rstrip('?')
                        if last_word in incomplete_question_words:
                            is_incomplete = True
                            logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ⏸️ Incomplete transcription (incomplete question): '{transcribed_text}' - waiting for more audio")
                
                if is_incomplete:
                    # Don't clear buffer - keep it so we can accumulate more audio
                    # Just return None to wait for more audio
                    logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ⏸️ Waiting for complete question/statement before responding")
                    return None
                
                # LLM response generation
                if not self.llm_service:
                    logger.error("LLM service not available")
                    return None
                
                llm_start = time.time()
                logger.info(f"[{datetime.now(timezone.utc).isoformat()}] 🤖 LLM generating from: '{transcribed_text[:50]}...'")
                
                full_response = ""
                async for chunk in self.llm_service.generate_meeting_response_streaming_with_context(
                    meeting_id,
                    text=transcribed_text,
                    conversation_context=conversation_context,
                ):
                    full_response += chunk
                
                llm_elapsed = (time.time() - llm_start) * 1000
                logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ✅ LLM: {len(full_response)} chars | Time: {llm_elapsed:.1f}ms")
                
                if not full_response or not full_response.strip():
                    logger.warning("Empty LLM response")
                    return None
                
                # Update conversation context
                conversation_context.append({"role": "user", "content": transcribed_text})
                conversation_context.append({"role": "assistant", "content": full_response})
                
                # Keep context manageable (last 20 messages)
                if len(conversation_context) > 20:
                    conversation_context[:] = conversation_context[-20:]
                
                # Clear buffer completely - no overlap to prevent old audio from previous sentence
                # being included in new transcriptions
                self.clear_buffer(keep_overlap=0)  # No overlap - prevent cross-contamination
                
                total_elapsed = (time.time() - stt_start) * 1000
                logger.info(f"[{datetime.now(timezone.utc).isoformat()}] ✅ Processing complete | Total: {total_elapsed:.1f}ms | STT→LLM: {transcribed_text[:30]}... → {len(full_response)} chars")
                
                return (transcribed_text, full_response)
                
            except Exception as e:
                logger.error(f"[{datetime.now(timezone.utc).isoformat()}] ❌ Processing error: {e}", exc_info=True)
                return None
            finally:
                self.processing = False
                logger.debug(f"[{datetime.now(timezone.utc).isoformat()}] 🔓 Processing lock released")
