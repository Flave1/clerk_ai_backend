"""
Speech-to-Text service using Whisper and AWS Transcribe.
"""
import asyncio
import io
import logging
import wave
from typing import Optional

import boto3
import whisper
from botocore.exceptions import ClientError

from shared.config import get_settings
from shared.schemas import STTRequest, STTResponse

logger = logging.getLogger(__name__)
settings = get_settings()


class STTService:
    """Speech-to-Text service."""

    def __init__(self):
        self.whisper_model = None
        self.aws_transcribe = None
        self._initialize_services()

    def _initialize_services(self):
        """Initialize STT services."""
        try:
            # Initialize Whisper
            if settings.openai_api_key:
                self.whisper_model = whisper.load_model(settings.whisper_model)
                logger.info(f"Whisper model {settings.whisper_model} loaded")

            # Initialize AWS Transcribe
            if settings.aws_access_key_id:
                self.aws_transcribe = boto3.client(
                    "transcribe",
                    region_name=settings.aws_region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                )
                logger.info("AWS Transcribe client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize STT services: {e}")
            raise

    async def transcribe(self, request: STTRequest) -> STTResponse:
        """Transcribe audio to text."""
        try:
            # Try Whisper first, fallback to AWS Transcribe
            if self.whisper_model:
                response = await self._transcribe_whisper(request)
            elif self.aws_transcribe:
                response = await self._transcribe_aws(request)
            else:
                raise RuntimeError("No STT service available")

            return response

        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
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

    async def transcribe_streaming(self, audio_stream, language: str = "en"):
        """Streaming transcription for real-time processing."""
        # This would implement streaming transcription
        # For now, return a placeholder
        logger.info("Streaming transcription not yet implemented")
        pass
