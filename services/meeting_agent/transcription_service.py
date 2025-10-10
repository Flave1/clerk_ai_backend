"""
Transcription service for real-time audio processing.

This module handles converting audio streams to text using OpenAI Whisper
and provides real-time transcription capabilities.
"""
import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict, Any, List
import io
import wave

import openai
from openai import AsyncOpenAI

from shared.config import get_settings
from shared.schemas import Meeting, TranscriptionChunk

logger = logging.getLogger(__name__)
settings = get_settings()


class TranscriptionService:
    """Service for real-time audio transcription."""
    
    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self.is_processing = False
        self.audio_buffer: List[bytes] = []
        self.chunk_size_seconds = 30  # Process audio in 30-second chunks
        self.sample_rate = 16000
        self.channels = 1
        
    async def initialize(self) -> None:
        """Initialize the transcription service."""
        logger.info("Initializing transcription service...")
        
        try:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
            
            logger.info("Transcription service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize transcription service: {e}")
            raise
    
    async def start_transcription(self, meeting: Meeting, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[TranscriptionChunk, None]:
        """Start transcribing audio from a meeting."""
        logger.info(f"Starting transcription for meeting: {meeting.title}")
        
        if not self.client:
            await self.initialize()
        
        self.is_processing = True
        self.audio_buffer = []
        
        try:
            async for audio_chunk in audio_stream:
                if not self.is_processing:
                    break
                
                # Add audio chunk to buffer
                self.audio_buffer.append(audio_chunk)
                
                # Process buffer when we have enough audio
                if len(self.audio_buffer) >= self._get_chunk_size():
                    transcription_chunk = await self._process_audio_chunk(meeting.id)
                    if transcription_chunk:
                        yield transcription_chunk
                    
                    # Clear buffer
                    self.audio_buffer = []
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            # Process any remaining audio in buffer
            if self.audio_buffer:
                transcription_chunk = await self._process_audio_chunk(meeting.id)
                if transcription_chunk:
                    yield transcription_chunk
        
        finally:
            self.is_processing = False
            logger.info("Transcription completed")
    
    def _get_chunk_size(self) -> int:
        """Calculate chunk size based on sample rate and duration."""
        # Assuming 16-bit audio (2 bytes per sample)
        bytes_per_sample = 2
        samples_per_chunk = self.sample_rate * self.chunk_size_seconds
        return samples_per_chunk * bytes_per_sample * self.channels
    
    async def _process_audio_chunk(self, meeting_id: str) -> Optional[TranscriptionChunk]:
        """Process a chunk of audio and return transcription."""
        try:
            if not self.audio_buffer:
                return None
            
            # Combine audio chunks
            audio_data = b''.join(self.audio_buffer)
            
            # Convert to WAV format for Whisper
            wav_data = self._convert_to_wav(audio_data)
            
            # Transcribe using OpenAI Whisper
            transcription = await self._transcribe_audio(wav_data)
            
            if transcription and transcription.strip():
                return TranscriptionChunk(
                    meeting_id=meeting_id,
                    text=transcription.strip(),
                    confidence=0.9,  # Whisper doesn't provide confidence scores
                    timestamp=datetime.utcnow(),
                    is_final=True
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return None
    
    def _convert_to_wav(self, audio_data: bytes) -> bytes:
        """Convert raw audio data to WAV format."""
        try:
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)
            
            wav_buffer.seek(0)
            return wav_buffer.read()
            
        except Exception as e:
            logger.error(f"WAV conversion error: {e}")
            return audio_data  # Return original data if conversion fails
    
    async def _transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio using OpenAI Whisper."""
        try:
            # Create audio file object
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav"
            
            # Transcribe using Whisper API
            response = await self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",  # Default to English
                response_format="text"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return None
    
    async def transcribe_file(self, file_path: str, language: str = "en") -> Optional[str]:
        """Transcribe an audio file."""
        try:
            if not self.client:
                await self.initialize()
            
            with open(file_path, "rb") as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="text"
                )
                
                return response
                
        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return None
    
    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None], language: str = "en") -> AsyncGenerator[str, None]:
        """Transcribe a continuous audio stream."""
        logger.info("Starting stream transcription...")
        
        if not self.client:
            await self.initialize()
        
        buffer = []
        chunk_size = self._get_chunk_size()
        
        try:
            async for audio_chunk in audio_stream:
                buffer.append(audio_chunk)
                
                # Process when buffer is full
                if len(buffer) >= chunk_size:
                    # Combine chunks
                    audio_data = b''.join(buffer)
                    
                    # Convert to WAV
                    wav_data = self._convert_to_wav(audio_data)
                    
                    # Transcribe
                    transcription = await self._transcribe_audio(wav_data)
                    
                    if transcription:
                        yield transcription
                    
                    # Clear buffer
                    buffer = []
            
            # Process remaining audio
            if buffer:
                audio_data = b''.join(buffer)
                wav_data = self._convert_to_wav(audio_data)
                transcription = await self._transcribe_audio(wav_data)
                
                if transcription:
                    yield transcription
                    
        except Exception as e:
            logger.error(f"Stream transcription error: {e}")
    
    async def stop_transcription(self) -> None:
        """Stop the transcription process."""
        logger.info("Stopping transcription...")
        self.is_processing = False
    
    async def get_transcription_stats(self) -> Dict[str, Any]:
        """Get transcription statistics."""
        return {
            'is_processing': self.is_processing,
            'buffer_size': len(self.audio_buffer),
            'chunk_size_seconds': self.chunk_size_seconds,
            'sample_rate': self.sample_rate,
            'channels': self.channels
        }
    
    async def cleanup(self) -> None:
        """Cleanup transcription service resources."""
        logger.info("Cleaning up transcription service...")
        
        try:
            await self.stop_transcription()
            self.audio_buffer = []
            
            logger.info("Transcription service cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Factory function for creating transcription service
def create_transcription_service() -> TranscriptionService:
    """Create a new transcription service instance."""
    return TranscriptionService()
