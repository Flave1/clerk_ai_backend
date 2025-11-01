"""
Text-to-Speech service using ElevenLabs and AWS Polly.
"""
import asyncio
import io
import logging
from typing import Optional, Iterator
import struct
import base64

import boto3
import elevenlabs
from botocore.exceptions import ClientError
import pydub
from pydub.utils import make_chunks

from shared.config import get_settings
from shared.schemas import TTSRequest, TTSResponse

logger = logging.getLogger(__name__)
settings = get_settings()


class TTSService:
    """Text-to-Speech service."""

    def __init__(self):
        self.elevenlabs_client = None
        self.aws_polly = None
        self._initialize_services()

    def _initialize_services(self):
        """Initialize TTS services."""
        try:
            # Initialize ElevenLabs
            if settings.elevenlabs_api_key:
                elevenlabs.set_api_key(settings.elevenlabs_api_key)
                self.elevenlabs_client = elevenlabs
                logger.info("ElevenLabs client initialized")

            # Initialize AWS Polly
            if settings.aws_access_key_id:
                self.aws_polly = boto3.client(
                    "polly",
                    region_name=settings.aws_region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                )
                logger.info("AWS Polly client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize TTS services: {e}")
            raise

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize text to speech."""
        try:
            # Try ElevenLabs first, fallback to AWS Polly
            if self.elevenlabs_client:
                try:
                    return await self._synthesize_elevenlabs(request)
                except Exception as e:
                    logger.error(f"ElevenLabs failed, trying AWS Polly: {e}")
                    if self.aws_polly:
                        return await self._synthesize_polly(request)
                    else:
                        raise RuntimeError("ElevenLabs failed and no AWS Polly fallback")
            elif self.aws_polly:
                return await self._synthesize_polly(request)
            else:
                raise RuntimeError("No TTS service available")

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise

    async def _synthesize_elevenlabs(self, request: TTSRequest) -> TTSResponse:
        """Synthesize using ElevenLabs."""
        try:
            # Get voice ID (use default if not specified)
            voice_id = (
                request.voice_id
                if request.voice_id and request.voice_id != "default"
                else "21m00Tcm4TlvDq8ikWAM"  # Rachel voice - a commonly available ElevenLabs voice
            )

            # Generate audio using streaming approach
            audio_generator = elevenlabs.generate(
                text=request.text, 
                voice=voice_id, 
                model="eleven_monolingual_v1",
                stream=True
            )

            # Collect audio chunks
            audio_chunks = []
            for chunk in audio_generator:
                if isinstance(chunk, bytes):
                    audio_chunks.append(chunk)
                else:
                    logger.warning(f"Unexpected chunk type: {type(chunk)}")
            
            # Combine all chunks
            audio_bytes = b"".join(audio_chunks)
            
            logger.info(f"ElevenLabs generated {len(audio_bytes)} bytes of audio")

            # Estimate duration (rough calculation)
            duration = len(request.text) * 0.1  # ~100ms per character

            return TTSResponse(
                audio_data=audio_bytes, duration=duration, voice_id=voice_id
            )

        except Exception as e:
            logger.error(f"ElevenLabs synthesis failed: {e}")
            raise

    async def _synthesize_polly(self, request: TTSRequest) -> TTSResponse:
        """Synthesize using AWS Polly."""
        try:
            # Map voice ID to Polly voice
            voice_id = self._map_voice_to_polly(request.voice_id)

            # Generate speech
            response = self.aws_polly.synthesize_speech(
                Text=request.text, OutputFormat="mp3", VoiceId=voice_id, Engine="neural"
            )

            # Read audio data
            audio_data = response["AudioStream"].read()

            # Get duration from metadata (if available)
            duration = self._estimate_audio_duration(audio_data)

            return TTSResponse(
                audio_data=audio_data, duration=duration, voice_id=voice_id
            )

        except Exception as e:
            logger.error(f"AWS Polly synthesis failed: {e}")
            raise

    def _map_voice_to_polly(self, voice_id: str) -> str:
        """Map voice ID to AWS Polly voice."""
        voice_mapping = {
            "default": "Joanna",
            "male": "Matthew",
            "female": "Joanna",
            "british": "Emma",
            "australian": "Nicole",
        }
        return voice_mapping.get(voice_id, "Joanna")

    def _estimate_audio_duration(self, audio_data: bytes) -> float:
        """Estimate audio duration from MP3 data."""
        # This is a rough estimation
        # In production, you'd use a proper audio library to get exact duration
        try:
            # MP3 files are typically ~128kbps
            # This is a very rough calculation
            estimated_duration = len(audio_data) / 16000  # Rough bytes per second
            return max(0.1, estimated_duration)  # Minimum 100ms
        except:
            return 1.0  # Default duration

    async def synthesize_streaming(self, text: str, voice_id: str = "default"):
        """Streaming TTS synthesis for real-time playback."""
        try:
            if self.elevenlabs_client:
                # ElevenLabs streaming
                voice = elevenlabs.Voice(voice_id=voice_id)

                # Generate streaming audio
                audio_stream = elevenlabs.generate(text=text, voice=voice, stream=True)

                async for audio_chunk in audio_stream:
                    yield audio_chunk

        except Exception as e:
            logger.error(f"Streaming TTS failed: {e}")
            raise

    async def get_available_voices(self) -> list:
        """Get list of available voices."""
        try:
            voices = []

            if self.elevenlabs_client:
                # Get ElevenLabs voices
                elevenlabs_voices = elevenlabs.voices()
                voices.extend(
                    [
                        {
                            "id": voice.voice_id,
                            "name": voice.name,
                            "provider": "elevenlabs",
                            "category": voice.category,
                        }
                        for voice in elevenlabs_voices
                    ]
                )

            if self.aws_polly:
                # Get AWS Polly voices
                response = self.aws_polly.describe_voices()
                voices.extend(
                    [
                        {
                            "id": voice["Id"],
                            "name": voice["Name"],
                            "provider": "polly",
                            "language": voice["LanguageCode"],
                            "gender": voice["Gender"],
                        }
                        for voice in response["Voices"]
                    ]
                )

            return voices

        except Exception as e:
            logger.error(f"Failed to get available voices: {e}")
            return []

    async def synthesize_to_pcm(self, text: str, voice_id: str = "default", model: str = "openai") -> Iterator[bytes]:
        """
        Synthesize text to PCM audio samples and yield in chunks in REAL-TIME.
        Returns PCM float32 samples at 16kHz.
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            model: TTS provider to use - '11lab' (default, real-time streaming) or 'openai' (buffered)
        
        Default: ElevenLabs streaming for minimal latency.
        Fallback: OpenAI TTS (buffered, not real-time) if ElevenLabs unavailable.
        """
        try:
            # ElevenLabs REAL-TIME STREAMING (default - preferred for low latency)
            if model == "11lab" and self.elevenlabs_client:
                actual_voice_id = voice_id if voice_id != "default" else "21m00Tcm4TlvDq8ikWAM"
                
                # Stream MP3 chunks from ElevenLabs (low latency)
                audio_generator = elevenlabs.generate(
                    text=text, 
                    voice=actual_voice_id, 
                    model="eleven_monolingual_v1",
                    stream=True  # Enable streaming
                )
                
                # Real-time streaming: decode MP3 chunks as they arrive for minimal latency
                mp3_buffer = b""
                min_buffer_size = 4096  # Minimum bytes before attempting decode (very low latency)
                max_attempts_per_chunk = 3  # Try decoding a few times if buffer grows
                total_mp3_bytes = 0
                total_pcm_samples = 0
                
                for mp3_chunk in audio_generator:
                    if not isinstance(mp3_chunk, bytes):
                        continue
                        
                    mp3_buffer += mp3_chunk
                    total_mp3_bytes += len(mp3_chunk)
                    
                    # Attempt to decode as soon as we have minimum data (for ultra-low latency)
                    # MP3 frames can be as small as ~144 bytes, but we need enough for a complete frame
                    attempts = 0
                    while len(mp3_buffer) >= min_buffer_size and attempts < max_attempts_per_chunk:
                        try:
                            # Decode MP3 chunk to PCM immediately
                            audio_segment = pydub.AudioSegment.from_mp3(io.BytesIO(mp3_buffer))
                            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                            
                            # Convert to float32 PCM samples
                            samples = audio_segment.get_array_of_samples()
                            float_samples = [float(s) / 32768.0 for s in samples]
                            
                            # Yield PCM chunks immediately (10k samples per chunk)
                            # Skip tiny final chunks to prevent continuous clicking sounds
                            chunk_size = 10000
                            min_chunk_size = 1000  # Don't send chunks smaller than this
                            
                            for i in range(0, len(float_samples), chunk_size):
                                chunk = float_samples[i:i + chunk_size]
                                
                                # Only send if chunk is substantial (skip tiny final chunks)
                                if len(chunk) >= min_chunk_size:
                                    pcm_chunk = struct.pack(f'<{len(chunk)}f', *chunk)
                                    total_pcm_samples += len(chunk)
                                    yield pcm_chunk
                                else:
                                    # Discard tiny final chunks to prevent noise
                                    logger.debug(f"Discarding tiny chunk ({len(chunk)} samples) to prevent noise")
                                    total_pcm_samples += len(chunk)  # Still count for logging
                            
                            # Successfully decoded - clear buffer and continue
                            mp3_buffer = b""
                            break
                            
                        except Exception as decode_error:
                            # Decode failed - might be incomplete MP3 frame
                            # If buffer is getting large, try with current data anyway
                            if len(mp3_buffer) >= 16384:  # 16KB - likely complete by now
                                attempts += 1
                                if attempts >= max_attempts_per_chunk:
                                    # Buffer the incomplete chunk for next iteration
                                    break
                            else:
                                # Not enough data yet, wait for more chunks
                                break
                
                # Process any remaining buffered MP3 data
                if len(mp3_buffer) > 0:
                    try:
                        audio_segment = pydub.AudioSegment.from_mp3(io.BytesIO(mp3_buffer))
                        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                        samples = audio_segment.get_array_of_samples()
                        float_samples = [float(s) / 32768.0 for s in samples]
                        
                        # Only send if the final chunk is substantial (>= 1000 samples)
                        # Tiny final chunks cause continuous clicking sounds
                        min_chunk_size = 1000
                        if len(float_samples) >= min_chunk_size:
                            chunk_size = 10000
                            for i in range(0, len(float_samples), chunk_size):
                                chunk = float_samples[i:i + chunk_size]
                                pcm_chunk = struct.pack(f'<{len(chunk)}f', *chunk)
                                total_pcm_samples += len(chunk)
                                yield pcm_chunk
                        else:
                            logger.debug(f"Skipping tiny final MP3 chunk ({len(float_samples)} samples) to prevent noise")
                            total_pcm_samples += len(float_samples)  # Still count for logging
                    except Exception as e:
                        logger.warning(f"Failed to decode final MP3 chunk ({len(mp3_buffer)} bytes): {e}")
                
                logger.info(f"ElevenLabs streamed {total_mp3_bytes} bytes MP3, synthesized {total_pcm_samples} PCM samples in real-time")
                return
            # End ElevenLabs streaming block
            elif model == "11lab":
                # ElevenLabs requested but not available - fallback to OpenAI if available
                logger.warning("ElevenLabs requested but not available, falling back to OpenAI")
            
            # OpenAI TTS (option or fallback - buffered, not real-time)
            # Note: OpenAI TTS waits for complete response before sending (higher latency)
            if (model == "openai" or model == "11lab") and settings.openai_api_key:
                import openai
                openai.api_key = settings.openai_api_key
                
                provider_type = "requested" if model == "openai" else "fallback"
                logger.info(f"Using OpenAI TTS ({provider_type} - buffered mode)")
                
                # Call OpenAI TTS API
                response = openai.audio.speech.create(
                    model="tts-1",
                    voice=voice_id if voice_id != "default" else "alloy",
                    input=text
                )
                
                audio_bytes = response.content
                logger.info(f"OpenAI TTS generated {len(audio_bytes)} bytes of audio")
                
                # Decode MP3 to PCM
                audio_segment = pydub.AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                
                # Convert to float32 PCM samples
                samples = audio_segment.get_array_of_samples()
                float_samples = [float(s) / 32768.0 for s in samples]
                
                # Yield in chunks of 10000 samples
                # But skip tiny final chunks to prevent noise
                chunk_size = 10000
                min_chunk_size = 1000  # Don't send chunks smaller than this
                
                for i in range(0, len(float_samples), chunk_size):
                    chunk = float_samples[i:i + chunk_size]
                    # Only send if chunk is substantial (skip tiny final chunks)
                    if len(chunk) >= min_chunk_size:
                        pcm_chunk = struct.pack(f'<{len(chunk)}f', *chunk)
                        yield pcm_chunk
                    else:
                        logger.debug(f"Skipping tiny final chunk ({len(chunk)} samples) to prevent noise")
                
                logger.info(f"Synthesized {len(float_samples)} samples to PCM (OpenAI)")
                return
            
            # No TTS provider available
            raise RuntimeError(f"TTS provider '{model}' not available and no fallback configured")
                
        except Exception as e:
            logger.error(f"PCM synthesis failed: {e}")
            raise

    async def synthesize_text_to_pcm_bytes(self, text: str, voice_id: str = "default") -> bytes:
        """
        Synthesize text and return all PCM samples as bytes.
        For use when we need the full audio before sending.
        """
        pcm_samples = []
        async for chunk in self.synthesize_to_pcm(text, voice_id):
            pcm_samples.append(chunk)
        return b"".join(pcm_samples)
