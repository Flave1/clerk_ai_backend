"""
Speech-to-Text service using Deepgram Nova-2 (default), OpenAI Whisper Realtime, and AWS Transcribe.
"""
import asyncio
import io
import logging
import os
import sys
import wave
from typing import Optional, AsyncIterator, Dict, Any
import struct

import boto3
import whisper
from botocore.exceptions import ClientError

from shared.config import get_settings
from shared.schemas import STTRequest, STTResponse

logger = logging.getLogger(__name__)
settings = get_settings()


class STTService:
    """Speech-to-Text service with real-time streaming support."""

    def __init__(self):
        self.whisper_model = None
        self.aws_transcribe = None
        self.deepgram_client = None
        self.openai_realtime_client = None
        self.elevenlabs_client = None
        self._initialize_services()

    def _initialize_services(self):
        """Initialize STT services."""
        try:
            # Initialize Deepgram (default for real-time streaming)
            deepgram_api_key = getattr(settings, 'deepgram_api_key', None)
            if not deepgram_api_key:
                deepgram_api_key = os.getenv('DEEPGRAM_API_KEY')
            
            if deepgram_api_key:
                try:
                    from deepgram import DeepgramClient, DeepgramClientOptions, PrerecordedOptions, FileSource
                    self.deepgram_client = DeepgramClient(deepgram_api_key)
                    logger.info("✅ Deepgram client initialized for real-time streaming")
                except ImportError:
                    logger.error("❌ Deepgram SDK not installed. Install with: pip install deepgram-sdk")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize Deepgram: {e}")
            else:
                logger.warning("⚠️ DEEPGRAM_API_KEY not found. Deepgram STT will not be available.")
                logger.warning("   Set DEEPGRAM_API_KEY environment variable or in settings to enable Deepgram STT.")
            
            # Initialize ElevenLabs Speech-to-Text (REQUIRED)
            elevenlabs_api_key = getattr(settings, 'elevenlabs_api_key', None)
            if not elevenlabs_api_key:
                elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
            
            if not elevenlabs_api_key:
                logger.error("❌ ELEVENLABS_API_KEY not found. ElevenLabs STT REQUIRED but not configured.")
                logger.error("   Please set ELEVENLABS_API_KEY environment variable or in settings.")
                logger.error("   Example: export ELEVENLABS_API_KEY='your_api_key_here'")
                raise RuntimeError("ELEVENLABS_API_KEY is required for ElevenLabs STT but not configured")
            
            try:
                import elevenlabs
                # Set API key using the same method as TTS service
                if hasattr(elevenlabs, 'set_api_key'):
                    elevenlabs.set_api_key(elevenlabs_api_key)
                    self.elevenlabs_client = elevenlabs
                    logger.info("✅ ElevenLabs Speech-to-Text client initialized successfully")
                else:
                    logger.error("❌ ElevenLabs SDK missing 'set_api_key' function. Please reinstall: pip install --upgrade elevenlabs")
                    raise RuntimeError("ElevenLabs SDK initialization failed - missing set_api_key")
            except ImportError as import_err:
                logger.error("❌ ElevenLabs SDK not installed.")
                logger.error(f"   Import error: {import_err}")
                logger.error("   Install with: pip install elevenlabs")
                raise
            except Exception as e:
                logger.error(f"❌ Failed to initialize ElevenLabs STT: {e}")
                logger.error(f"   Error details: {type(e).__name__}: {str(e)}")
                raise  # Raise to prevent service from continuing without ElevenLabs
            
            # Initialize OpenAI Whisper Realtime (fallback)
            if settings.openai_api_key:
                try:
                    from openai import OpenAI
                    self.openai_realtime_client = OpenAI(api_key=settings.openai_api_key)
                    logger.info("OpenAI Whisper API client initialized")
                except ImportError:
                    logger.warning("OpenAI SDK not installed. Install with: pip install openai")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI Whisper: {e}")
                    
                # Fallback: Initialize local Whisper model
                try:
                    self.whisper_model = whisper.load_model(getattr(settings, 'whisper_model', 'base'))
                    logger.info(f"Whisper model {settings.whisper_model} loaded (fallback)")
                except Exception as e:
                    logger.warning(f"Failed to load Whisper model: {e}")

            # Initialize AWS Transcribe (fallback)
            if settings.aws_access_key_id:
                try:
                    self.aws_transcribe = boto3.client(
                        "transcribe",
                        region_name=settings.aws_region,
                        aws_access_key_id=settings.aws_access_key_id,
                        aws_secret_access_key=settings.aws_secret_access_key,
                    )
                    logger.info("AWS Transcribe client initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize AWS Transcribe: {e}")

            if not self.deepgram_client and not self.openai_realtime_client and not self.whisper_model and not self.aws_transcribe and not self.elevenlabs_client:
                logger.warning("No STT service available. Install Deepgram/ElevenLabs SDK or configure OpenAI/AWS.")

        except Exception as e:
            logger.error(f"Failed to initialize STT services: {e}")
            # Don't raise - allow service to continue with limited functionality

    async def transcribe(self, request: STTRequest, model: str = "deepgram") -> STTResponse:
        """
        Transcribe audio to text using specified model.
        
        Args:
            request: STTRequest with audio data
            model: STT provider - "deepgram" (default), "elevenlabs", "openai", "whisper", or "aws"
        
        Returns:
            STTResponse with transcribed text
        """
        try:
            # Try Deepgram first (default)
            if model == "deepgram" and self.deepgram_client:
                return await self._transcribe_deepgram(request)
            # Try ElevenLabs
            elif model == "elevenlabs" and self.elevenlabs_client:
                return await self._transcribe_elevenlabs_request(request)
            # Try OpenAI Whisper API
            elif model == "openai" and self.openai_realtime_client:
                return await self._transcribe_openai(request)
            # Try local Whisper model
            elif model == "whisper" and self.whisper_model:
                return await self._transcribe_whisper(request)
            # Try AWS Transcribe
            elif model == "aws" and self.aws_transcribe:
                return await self._transcribe_aws(request)
            else:
                # Auto-fallback: try in priority order
                if self.deepgram_client:
                    return await self._transcribe_deepgram(request)
                elif self.elevenlabs_client:
                    return await self._transcribe_elevenlabs_request(request)
                elif self.openai_realtime_client:
                    return await self._transcribe_openai(request)
                elif self.whisper_model:
                    return await self._transcribe_whisper(request)
                elif self.aws_transcribe:
                    return await self._transcribe_aws(request)
                else:
                    raise RuntimeError("No STT service available")

        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            raise

    async def _transcribe_deepgram(self, request: STTRequest) -> STTResponse:
        """Transcribe using Deepgram Nova-2."""
        try:
            from deepgram import PrerecordedOptions, FileSource
            
            if not self.deepgram_client:
                raise RuntimeError("Deepgram client not initialized")
            
            # Validate audio data
            if not request.audio_data or len(request.audio_data) < 1600:
                logger.warning(f"⚠️ Audio data too short: {len(request.audio_data) if request.audio_data else 0} bytes")
                return STTResponse(text="", confidence=0.0, language=request.language, is_final=True)
            
            # Convert raw PCM audio to WAV format
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(request.audio_data)
            
            wav_buffer.seek(0)
            wav_data = wav_buffer.read()
            
            # Create Deepgram payload and options
            payload: FileSource = {"buffer": wav_data}
            options = PrerecordedOptions(
                model="nova-2",
                language=request.language,
                smart_format=True,
                punctuate=True,
            )
            
            # Transcribe
            logger.info(f"📡 Sending {len(wav_data)} bytes WAV to Deepgram...")
            response = self.deepgram_client.listen.rest.v("1").transcribe_file(payload, options)
            
            # Parse response
            if response and response.results and response.results.channels:
                channel = response.results.channels[0]
                if channel.alternatives:
                    alternative = channel.alternatives[0]
                    text = getattr(alternative, 'transcript', '') or getattr(alternative, 'text', '')
                    confidence = getattr(alternative, 'confidence', 0.9)
                    text = text.strip() if text else ""
                    
                    if text:
                        logger.info(f"✅ Deepgram: '{text}' (confidence: {confidence})")
                    else:
                        logger.warning(f"⚠️ Deepgram returned empty text (confidence: {confidence})")
                    
                    return STTResponse(
                        text=text,
                        confidence=confidence,
                        language=request.language,
                        is_final=True,
                    )
            
            # No transcription result
            logger.warning("⚠️ Deepgram returned no transcription results")
            return STTResponse(text="", confidence=0.0, language=request.language, is_final=True)
            
        except ImportError:
            logger.error("Deepgram SDK not installed. Install with: pip install deepgram-sdk")
            raise RuntimeError("Deepgram SDK not available")
        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}")
            raise
    
    async def _transcribe_elevenlabs_request(self, request: STTRequest) -> STTResponse:
        """Transcribe using ElevenLabs Speech-to-Text API with STTRequest."""
        try:
            text = await self._transcribe_elevenlabs(request.audio_data, request.language)
            
            return STTResponse(
                text=text,
                confidence=0.9,  # ElevenLabs doesn't provide confidence scores
                language=request.language,
                is_final=True,
            )
        except Exception as e:
            logger.error(f"ElevenLabs STT transcription failed: {e}")
            raise
    
    async def _transcribe_openai(self, request: STTRequest) -> STTResponse:
        """Transcribe using OpenAI Whisper API with STTRequest."""
        try:
            if not self.openai_realtime_client:
                raise RuntimeError("OpenAI client not initialized")
            
            # Create a WAV file in memory for OpenAI Whisper API
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(request.audio_data)
            
            wav_buffer.seek(0)
            
            # Create transcription request using OpenAI client
            response = self.openai_realtime_client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", wav_buffer.read(), "audio/wav"),
                language=request.language,
                response_format="verbose_json",
                temperature=0
            )
            
            text = response.text.strip() if response.text else ""
            confidence = getattr(response, 'confidence', 0.9)
            
            return STTResponse(
                text=text,
                confidence=confidence,
                language=request.language,
                is_final=True,
            )
            
        except Exception as e:
            logger.error(f"OpenAI Whisper transcription failed: {e}")
            raise

    async def _transcribe_whisper(self, request: STTRequest) -> STTResponse:
        """Transcribe using Whisper."""
        try:
            # Convert audio data to format Whisper expects
            audio_data = self._preprocess_audio(request.audio_data)

            # Transcribe
            result = self.whisper_model.transcribe(
                audio_data,
                language=request.language,
                fp16=False,  # Use fp32 for better compatibility
            )

            # Extract text and confidence
            text = result["text"].strip()

            # Whisper doesn't provide confidence scores directly
            # We can estimate based on segment-level confidence
            segments = result.get("segments", [])
            if segments:
                avg_confidence = sum(
                    seg.get("avg_logprob", -1) for seg in segments
                ) / len(segments)
                # Convert log probability to confidence (rough approximation)
                confidence = max(0, min(1, (avg_confidence + 1) / 2))
            else:
                confidence = 0.8  # Default confidence

            return STTResponse(
                text=text,
                confidence=confidence,
                language=request.language,
                is_final=True,
            )

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise

    async def _transcribe_aws(self, request: STTRequest) -> STTResponse:
        """Transcribe using AWS Transcribe."""
        try:
            import os
            import tempfile

            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(request.audio_data)
                temp_file_path = temp_file.name

            try:
                # Start transcription job
                job_name = f"clerk-{request.conversation_id}-{request.turn_number}"

                self.aws_transcribe.start_transcription_job(
                    TranscriptionJobName=job_name,
                    Media={"MediaFileUri": f"file://{temp_file_path}"},
                    MediaFormat="wav",
                    LanguageCode=request.language,
                    Settings={"ShowSpeakerLabels": False, "MaxSpeakerLabels": 1},
                )

                # Wait for job completion
                while True:
                    response = self.aws_transcribe.get_transcription_job(
                        TranscriptionJobName=job_name
                    )

                    status = response["TranscriptionJob"]["TranscriptionJobStatus"]

                    if status == "COMPLETED":
                        # Get transcription result
                        transcript_uri = response["TranscriptionJob"]["Transcript"][
                            "TranscriptFileUri"
                        ]

                        # Download and parse result
                        import requests

                        result_response = requests.get(transcript_uri)
                        result_json = result_response.json()

                        text = result_json["results"]["transcripts"][0]["transcript"]
                        confidence = result_json["results"]["items"][0]["alternatives"][
                            0
                        ]["confidence"]

                        return STTResponse(
                            text=text,
                            confidence=confidence,
                            language=request.language,
                            is_final=True,
                        )

                    elif status == "FAILED":
                        raise RuntimeError(
                            f"Transcription job failed: {response['TranscriptionJob']['FailureReason']}"
                        )

                    # Wait before checking again
                    await asyncio.sleep(1)

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

                # Clean up transcription job
                try:
                    self.aws_transcribe.delete_transcription_job(
                        TranscriptionJobName=job_name
                    )
                except ClientError:
                    pass  # Job might already be deleted

        except Exception as e:
            logger.error(f"AWS Transcribe failed: {e}")
            raise

    def _preprocess_audio(self, audio_data: bytes):
        """Preprocess audio data for Whisper."""
        try:
            import numpy as np
            
            # Simple conversion: assume the audio data is already in the right format
            # Try to load as numpy array directly
            try:
                # For WebM/audio data from browser, try to convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32) / 255.0
                
                # Normalize the audio
                if len(audio_array) > 0:
                    audio_array = (audio_array - 0.5) * 2  # Convert from [0,1] to [-1,1]
                
                return audio_array
            except Exception:
                # Fallback: return as-is and let Whisper handle it
                logger.warning("Could not convert audio data to numpy array, passing raw data")
                return audio_data

        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            # Final fallback: return raw data
            return audio_data

    async def transcribe_streaming(self, audio_chunks, language: str = "en"):
        """Streaming transcription for real-time processing."""
        try:
            # Buffer for accumulating audio chunks
            audio_buffer = b""
            chunk_count = 0
            
            async for audio_chunk in audio_chunks:
                audio_buffer += audio_chunk
                chunk_count += 1
                
                # Process every 5 chunks or when buffer reaches certain size
                if chunk_count >= 5 or len(audio_buffer) >= 16000:  # ~1 second at 16kHz
                    try:
                        # Create STT request for buffered audio
                        request = STTRequest(
                            audio_data=audio_buffer,
                            conversation_id="streaming",
                            turn_number=chunk_count,
                            language=language
                        )
                        
                        # Transcribe the buffered audio
                        response = await self.transcribe(request)
                        
                        if response and response.text.strip():
                            yield response
                        
                        # Reset buffer
                        audio_buffer = b""
                        chunk_count = 0
                        
                    except Exception as e:
                        logger.error(f"Streaming transcription error: {e}")
                        continue
            
            # Process any remaining audio in buffer
            if audio_buffer:
                try:
                    request = STTRequest(
                        audio_data=audio_buffer,
                        conversation_id="streaming",
                        turn_number=chunk_count,
                        language=language
                    )
                    
                    response = await self.transcribe(request)
                    if response and response.text.strip():
                        yield response
                        
                except Exception as e:
                    logger.error(f"Final streaming transcription error: {e}")
                    
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            raise

    async def transcribe_stream_realtime(
        self, 
        audio_stream: AsyncIterator[bytes], 
        language: str = "en",
        model: str = "deepgram"
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Real-time streaming transcription with VAD support.
        
        Args:
            audio_stream: Async iterator of PCM 16-bit audio chunks (bytes)
            language: Language code (default: "en")
            model: STT provider - "deepgram" (default), "openai", or "whisper"
        
        Yields:
            Dict with keys: text, is_final, confidence, language
        """
        try:
            # Deepgram Nova-2 Real-time Streaming (default)
            if model == "deepgram" and self.deepgram_client:
                async for result in self._transcribe_deepgram_stream(audio_stream, language):
                    yield result
                return
            
            # OpenAI Whisper Realtime API
            elif model == "openai" and self.openai_realtime_client:
                async for result in self._transcribe_openai_realtime_stream(audio_stream, language):
                    yield result
                return
            
            # Fallback to buffered transcription
            elif model == "whisper" and self.whisper_model:
                audio_buffer = b""
                async for chunk in audio_stream:
                    audio_buffer += chunk
                    if len(audio_buffer) >= 32000:  # ~1 second at 16kHz mono 16-bit
                        try:
                            request = STTRequest(
                                audio_data=audio_buffer,
                                conversation_id="realtime",
                                turn_number=0,
                                language=language
                            )
                            response = await self._transcribe_whisper(request)
                            if response.text.strip():
                                yield {
                                    "text": response.text,
                                    "is_final": response.is_final,
                                    "confidence": response.confidence,
                                    "language": response.language
                                }
                            audio_buffer = b""
                        except Exception as e:
                            logger.error(f"Whisper streaming error: {e}")
                return
            
            raise RuntimeError(f"STT model '{model}' not available")
            
        except Exception as e:
            logger.error(f"Real-time streaming transcription failed: {e}")
            raise

    async def _transcribe_deepgram_stream(
        self, 
        audio_stream: AsyncIterator[bytes], 
        language: str = "en"
    ) -> AsyncIterator[Dict[str, Any]]:
        """Transcribe using Deepgram Nova-2 real-time streaming."""
        try:
            from deepgram import DeepgramClient, PrerecordedOptions, FileSource
            
            # For streaming, we'll use Deepgram's streaming API
            # Buffer audio and send in chunks for real-time processing
            audio_buffer = b""
            chunk_duration_ms = 500  # Process every 500ms
            samples_per_chunk = int(16000 * chunk_duration_ms / 1000)  # 8000 samples = 500ms
            bytes_per_chunk = samples_per_chunk * 2  # 16-bit = 2 bytes per sample
            
            async for audio_chunk in audio_stream:
                audio_buffer += audio_chunk
                
                # Process when we have enough audio (~500ms)
                if len(audio_buffer) >= bytes_per_chunk:
                    try:
                        # Use Deepgram's prerecorded API with streaming-like processing
                        payload: FileSource = {
                            "buffer": audio_buffer[:bytes_per_chunk],
                        }
                        
                        options = PrerecordedOptions(
                            model="nova-2",
                            language=language,
                            smart_format=True,
                            punctuate=True,
                        )
                        
                        # Transcribe chunk
                        response = self.deepgram_client.listen.rest.v("1").transcribe_file(payload, options)
                        
                        if response.results and response.results.channels:
                            channel = response.results.channels[0]
                            if channel.alternatives:
                                transcript = channel.alternatives[0].transcript
                                if transcript.strip():
                                    yield {
                                        "text": transcript,
                                        "is_final": True,
                                        "confidence": getattr(channel.alternatives[0], 'confidence', 0.9),
                                        "language": language
                                    }
                        
                        # Keep remaining audio in buffer
                        audio_buffer = audio_buffer[bytes_per_chunk:]
                        
                    except Exception as e:
                        logger.error(f"Deepgram chunk transcription error: {e}")
                        audio_buffer = b""  # Reset on error
                        continue
            
            # Process remaining buffer
            if len(audio_buffer) > 3200:  # At least 100ms of audio
                try:
                    payload: FileSource = {"buffer": audio_buffer}
                    options = PrerecordedOptions(
                        model="nova-2",
                        language=language,
                        smart_format=True,
                        punctuate=True,
                    )
                    response = self.deepgram_client.listen.rest.v("1").transcribe_file(payload, options)
                    
                    if response.results and response.results.channels:
                        channel = response.results.channels[0]
                        if channel.alternatives:
                            transcript = channel.alternatives[0].transcript
                            if transcript.strip():
                                yield {
                                    "text": transcript,
                                    "is_final": True,
                                    "confidence": getattr(channel.alternatives[0], 'confidence', 0.9),
                                    "language": language
                                }
                except Exception as e:
                    logger.error(f"Deepgram final chunk transcription error: {e}")
                    
        except ImportError:
            logger.error("Deepgram SDK not installed. Install with: pip install deepgram-sdk")
            raise RuntimeError("Deepgram SDK not available")
        except Exception as e:
            logger.error(f"Deepgram streaming transcription failed: {e}")
            raise

    async def _transcribe_openai_realtime_stream(
        self, 
        audio_stream: AsyncIterator[bytes], 
        language: str = "en"
    ) -> AsyncIterator[Dict[str, Any]]:
        """Transcribe using OpenAI Whisper API."""
        try:
            if not self.openai_realtime_client:
                raise RuntimeError("OpenAI client not initialized")
            
            # Buffer audio chunks for transcription
            audio_buffer = b""
            # Minimum 1 second of audio: 16kHz * 2 bytes (16-bit) = 32000 bytes
            min_buffer_size = 32000  # 1 second at 16kHz, 16-bit PCM
            
            async for audio_chunk in audio_stream:
                audio_buffer += audio_chunk
                
                # Process when we have enough audio (at least 1 second)
                if len(audio_buffer) >= min_buffer_size:
                    try:
                        # Create a WAV file in memory for OpenAI Whisper API
                        wav_buffer = io.BytesIO()
                        with wave.open(wav_buffer, 'wb') as wav_file:
                            wav_file.setnchannels(1)  # Mono
                            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                            wav_file.setframerate(16000)  # 16kHz
                            wav_file.writeframes(audio_buffer)
                        
                        wav_buffer.seek(0)
                        
                        # Create transcription request using OpenAI client
                        response = self.openai_realtime_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=("audio.wav", wav_buffer.read(), "audio/wav"),
                            language=language,
                            response_format="verbose_json",
                            temperature=0
                        )
                        
                        if response.text.strip():
                            yield {
                                "text": response.text,
                                "is_final": True,
                                "confidence": getattr(response, 'confidence', 0.9),
                                "language": language
                            }
                        
                        audio_buffer = b""
                    except Exception as e:
                        logger.error(f"OpenAI Realtime transcription error: {e}")
                        continue
            
            # Process remaining audio
            if audio_buffer and len(audio_buffer) >= 3200:  # At least 100ms of audio
                try:
                    # Create WAV file for remaining audio
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                        wav_file.setframerate(16000)  # 16kHz
                        wav_file.writeframes(audio_buffer)
                    
                    wav_buffer.seek(0)
                    
                    response = self.openai_realtime_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=("audio.wav", wav_buffer.read(), "audio/wav"),
                        language=language,
                        response_format="verbose_json"
                    )
                    if response.text.strip():
                        yield {
                            "text": response.text,
                            "is_final": True,
                            "confidence": getattr(response, 'confidence', 0.9),
                            "language": language
                        }
                except Exception as e:
                    logger.error(f"OpenAI final transcription error: {e}")
                    
        except Exception as e:
            logger.error(f"OpenAI Realtime streaming transcription failed: {e}")
            raise

    async def _transcribe_elevenlabs(
        self,
        audio_buffer: bytes,
        language: str = "en"
    ) -> str:
        """Transcribe audio using ElevenLabs Speech-to-Text API."""
        try:
            if not self.elevenlabs_client:
                raise RuntimeError("ElevenLabs client not initialized")
            
            # Create a BytesIO object from the audio buffer
            audio_file = io.BytesIO(audio_buffer)
            
            # Determine language code for ElevenLabs (e.g., "eng" for English)
            lang_map = {
                "en": "eng",
                "es": "spa",
                "fr": "fra",
                "de": "deu",
                "it": "ita",
                "pt": "por",
                "ja": "jpn",
                "ko": "kor",
                "zh": "zho",
            }
            language_code = lang_map.get(language, "eng")
            
            # Transcribe using ElevenLabs Speech-to-Text API
            # Note: ElevenLabs STT API expects a file-like object or file path
            # Check if client has speech_to_text attribute (newer SDK) or use direct API call
            if hasattr(self.elevenlabs_client, 'speech_to_text') and hasattr(self.elevenlabs_client.speech_to_text, 'convert'):
                transcription = self.elevenlabs_client.speech_to_text.convert(
                    file=audio_file,
                    model_id="scribe_v1",
                    tag_audio_events=False,  # Optional: set to True if you want audio event tagging
                    language_code=language_code,
                    diarize=False,  # Optional: set to True if you want speaker diarization
                )
            else:
                # Fallback: try direct API call (older SDK or module-level)
                # This may need adjustment based on actual ElevenLabs STT API
                raise RuntimeError("ElevenLabs STT API not accessible - client structure not recognized")
            
            # Extract text from transcription response
            # ElevenLabs returns a SpeechToTextResponse object with a text attribute
            text = ""
            if hasattr(transcription, 'text'):
                text = transcription.text
            elif isinstance(transcription, str):
                text = transcription
            elif hasattr(transcription, 'transcript'):
                text = transcription.transcript
            elif hasattr(transcription, 'get'):
                # If it's a dict-like object
                text = transcription.get('text', '') or transcription.get('transcript', '')
            else:
                # Try string conversion as last resort
                text = str(transcription)
            
            # Clean up the text
            text = text.strip() if text else ""
            
            # Remove any extra whitespace or newlines
            text = ' '.join(text.split())
            
            return text
            
        except Exception as e:
            logger.error(f"ElevenLabs STT transcription failed: {e}")
            raise

    def detect_speech_activity(self, audio_chunk: bytes, sample_rate: int = 16000) -> bool:
        """
        Voice Activity Detection (VAD) to detect if audio chunk contains speech.
        Uses manual energy-based detection with zero-crossing rate and spectral analysis.
        
        Args:
            audio_chunk: PCM 16-bit audio data (bytes)
            sample_rate: Sample rate in Hz (default: 16000)
        
        Returns:
            True if speech is detected, False otherwise
        """
        try:
            import struct
            import numpy as np
            
            # Convert bytes to numpy array (16-bit signed integers)
            samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
            
            # Normalize to [-1, 1]
            if len(samples) > 0:
                samples = samples / 32768.0
            else:
                return False
            
            # Energy-based detection: calculate RMS (Root Mean Square) energy
            rms_energy = np.sqrt(np.mean(samples**2))
            # Moderate threshold - balance between detecting speech and filtering noise
            energy_threshold = 0.005  # Lowered from 0.01 to detect quieter speech
            
            # Also check for non-zero samples (simple activity detection)
            non_zero_ratio = np.count_nonzero(np.abs(samples) > 0.001) / len(samples) if len(samples) > 0 else 0  # Lowered from 0.01 to 0.001
            
            # Manual energy-based VAD (no external dependencies)
            # Calculate zero-crossing rate (ZCR) - speech typically has higher ZCR than noise
            zcr = 0.0
            if len(samples) > 1:
                # Count sign changes (zero crossings)
                sign_changes = np.sum(np.diff(np.sign(samples)) != 0)
                zcr = sign_changes / len(samples)
            
            # Calculate spectral centroid (rough approximation using energy distribution)
            # Speech has energy concentrated in certain frequency bands
            spectral_centroid = 0.0
            if len(samples) > 0:
                # Simple frequency domain analysis using FFT
                try:
                    fft = np.fft.rfft(samples)
                    fft_magnitude = np.abs(fft)
                    if np.sum(fft_magnitude) > 0:
                        # Calculate weighted average frequency
                        freqs = np.fft.rfftfreq(len(samples), 1.0 / sample_rate)
                        spectral_centroid = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)
                except Exception:
                    pass
            
            # Manual VAD decision using multiple features
            # Speech characteristics:
            # 1. Higher RMS energy (already calculated)
            # 2. Higher zero-crossing rate (typical range: 0.01-0.1 for speech vs 0.001-0.01 for noise)
            # 3. Significant non-zero samples (already calculated)
            # 4. Spectral centroid in speech range (typically 100-3000 Hz)
            
            energy_based = (rms_energy > energy_threshold) and (non_zero_ratio > 0.05)  # Lowered from 0.15 to 0.05
            zcr_based = 0.002 < zcr < 0.2  # ZCR in typical speech range (widened from 0.005-0.15)
            spectral_based = 50 < spectral_centroid < 4000  # Spectral centroid in speech range (widened from 100-3000)
            
            # Speech detected if energy is high AND either ZCR or spectral features indicate speech
            # Lowered RMS threshold from 0.005 to 0.002 to detect quieter speech
            has_speech = energy_based and (zcr_based or spectral_based) and (rms_energy > 0.002)
            
            # Log occasionally for debugging (always log when speech is detected)
            if has_speech:
                import time
                if not hasattr(self, '_last_log_time'):
                    self._last_log_time = 0
                
                # Log when speech is detected (but throttle to once per second)
                if (time.time() - self._last_log_time) > 1.0:
                    logger.info(f"✅ Speech activity detected: RMS={rms_energy:.4f}, non_zero={non_zero_ratio:.2%}, ZCR={zcr:.4f}, spectral={spectral_centroid:.1f}Hz, audio_len={len(audio_chunk)} bytes")
                    logger.info(f"   VAD result: energy_based={energy_based}, zcr_based={zcr_based}, spectral_based={spectral_based}, has_speech={has_speech}")
                    self._last_log_time = time.time()
            else:
                # Log when speech is NOT detected (throttled to avoid spam)
                import time
                if not hasattr(self, '_last_vad_false_time'):
                    self._last_vad_false_time = 0
                
                if (time.time() - self._last_vad_false_time) > 2.0:
                    logger.debug(f"❌ No speech: RMS={rms_energy:.4f}, non_zero={non_zero_ratio:.2%}, ZCR={zcr:.4f}, spectral={spectral_centroid:.1f}Hz")
                    logger.debug(f"   VAD checks: energy_based={energy_based} (RMS>{energy_threshold}, non_zero>{0.15}), zcr_based={zcr_based}, spectral_based={spectral_based}")
                    self._last_vad_false_time = time.time()
            
            return has_speech
                
        except Exception as e:
            logger.error(f"VAD detection failed: {e}")
            # Fallback: check if chunk has any non-zero data
            return len(audio_chunk) > 0 and any(b != 0 for b in audio_chunk[:100])
