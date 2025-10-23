"""
Text-to-Speech service using ElevenLabs and AWS Polly.
"""
import asyncio
import io
import logging
from typing import Optional

import boto3
import elevenlabs
from botocore.exceptions import ClientError

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
                if request.voice_id != "default"
                else "21m00Tcm4TlvDq8ikWAM"
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
