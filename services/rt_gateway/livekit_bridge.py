"""
LiveKit integration for real-time audio/video processing.
"""
import asyncio
import logging
from typing import Callable, Dict, Optional

import livekit
from livekit import api, rtc

from shared.config import get_settings
from shared.schemas import Conversation, Turn, TurnType

logger = logging.getLogger(__name__)
settings = get_settings()


class LiveKitBridge:
    """LiveKit bridge for real-time communication."""

    def __init__(self, turn_manager, stt_service, tts_service, event_publisher):
        self.turn_manager = turn_manager
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.event_publisher = event_publisher

        self.room = None
        self.rooms: Dict[str, rtc.Room] = {}
        self.participants: Dict[str, rtc.RemoteParticipant] = {}
        
        # Audio buffering for streaming STT
        self.audio_buffers: Dict[str, bytes] = {}  # conversation_id -> audio_buffer
        self.buffer_threshold = 16000  # ~1 second at 16kHz
        self.chunk_count: Dict[str, int] = {}  # conversation_id -> chunk_count

    async def initialize(self):
        """Initialize LiveKit connection."""
        try:
            # Set up LiveKit API client
            self.api_client = api.LiveKitAPI(
                url=settings.livekit_url,
                api_key=settings.livekit_api_key,
                api_secret=settings.livekit_api_secret,
            )

            logger.info("LiveKit bridge initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LiveKit bridge: {e}")
            raise

    async def join_room(
        self, room_id: str, participant_name: str = "AI Assistant"
    ) -> str:
        """Join a LiveKit room."""
        try:
            room = rtc.Room()

            # Set up event handlers
            room.on("participant_connected", self._on_participant_connected)
            room.on("participant_disconnected", self._on_participant_disconnected)
            room.on("track_subscribed", self._on_track_subscribed)
            room.on("track_unsubscribed", self._on_track_unsubscribed)
            room.on("data_received", self._on_data_received)

            # Generate token and connect
            token = (
                self.api_client.access_token()
                .with_identity(participant_name)
                .with_name(participant_name)
                .with_grants(
                    api.VideoGrants(
                        room_join=True,
                        room=room_id,
                        can_publish=True,
                        can_subscribe=True,
                    )
                )
                .to_jwt()
            )

            await room.connect(settings.livekit_url, token)
            await room.local_participant.enable_camera_and_microphone()

            self.rooms[room_id] = room
            logger.info(f"Joined room {room_id}")

            return room_id

        except Exception as e:
            logger.error(f"Failed to join room {room_id}: {e}")
            raise

    async def leave_room(self, room_id: str):
        """Leave a LiveKit room."""
        try:
            room = self.rooms.get(room_id)
            if room:
                await room.disconnect()
                del self.rooms[room_id]
                logger.info(f"Left room {room_id}")

        except Exception as e:
            logger.error(f"Failed to leave room {room_id}: {e}")

    async def process_audio_data(self, conversation_id: str, audio_data: bytes):
        """Process incoming audio data through the AI pipeline with buffering."""
        try:
            logger.info(f"Processing audio data for conversation {conversation_id}: {len(audio_data)} bytes")
            
            # Initialize buffer for this conversation if not exists
            if conversation_id not in self.audio_buffers:
                self.audio_buffers[conversation_id] = b""
                self.chunk_count[conversation_id] = 0
            
            # Add audio data to buffer
            self.audio_buffers[conversation_id] += audio_data
            self.chunk_count[conversation_id] += 1
            
            # Check if buffer is ready for processing
            if (len(self.audio_buffers[conversation_id]) >= self.buffer_threshold or 
                self.chunk_count[conversation_id] >= 5):
                
                # Process buffered audio
                await self._process_buffered_audio(conversation_id)
                
                # Reset buffer
                self.audio_buffers[conversation_id] = b""
                self.chunk_count[conversation_id] = 0
            
        except Exception as e:
            logger.error(f"Failed to process audio data: {e}")
            raise
    
    async def _process_buffered_audio(self, conversation_id: str):
        """Process buffered audio data through STT."""
        try:
            audio_buffer = self.audio_buffers.get(conversation_id, b"")
            if not audio_buffer:
                return
            
            logger.info(f"Processing buffered audio for conversation {conversation_id}: {len(audio_buffer)} bytes")
            
            # 1. Convert audio data to proper format for STT
            processed_audio = await self._preprocess_audio(audio_buffer)
            if not processed_audio:
                logger.warning("Failed to preprocess buffered audio data")
                return
            
            # 2. Send to STT service for transcription
            from shared.schemas import STTRequest
            stt_request = STTRequest(
                audio_data=processed_audio,
                conversation_id=conversation_id,
                turn_number=self.chunk_count.get(conversation_id, 1),
                language="en"
            )
            
            stt_response = await self.stt_service.transcribe(stt_request)
            if not stt_response or not stt_response.text.strip():
                logger.info("No speech detected in buffered audio")
                return
            
            logger.info(f"Transcribed buffered audio: {stt_response.text}")
            
            # 3. Create a user turn and process through turn manager
            from shared.schemas import Turn, TurnType
            from datetime import datetime, timezone
            from uuid import uuid4, UUID
            
            user_turn = Turn(
                id=uuid4(),
                conversation_id=UUID(conversation_id),
                turn_number=self.chunk_count.get(conversation_id, 1),
                turn_type=TurnType.USER_SPEECH,
                content=stt_response.text,
                timestamp=datetime.now(timezone.utc),
                confidence_score=stt_response.confidence
            )
            
            # 4. Process through turn manager (handles LLM and TTS)
            await self.turn_manager.process_turn(user_turn)
            
        except Exception as e:
            logger.error(f"Failed to process buffered audio: {e}")
            raise
    
    async def _preprocess_audio(self, audio_data: bytes) -> Optional[bytes]:
        """Preprocess audio data for STT service with quality optimization."""
        try:
            # Convert audio to the format expected by STT service
            # This might involve format conversion, resampling, etc.
            
            # For WebM/Opus audio from browser, we need to handle it properly
            # For now, we'll try to detect the format and convert if needed
            
            # Check if it's WebM format (common from browser MediaRecorder)
            if audio_data.startswith(b'\x1a\x45\xdf\xa3'):  # WebM signature
                logger.info("Detected WebM audio format")
                # For now, pass through - Whisper can handle WebM
                return audio_data
            
            # Check if it's WAV format
            elif audio_data.startswith(b'RIFF'):
                logger.info("Detected WAV audio format")
                return audio_data
            
            # Check if it's MP3 format
            elif audio_data.startswith(b'\xff\xfb') or audio_data.startswith(b'ID3'):
                logger.info("Detected MP3 audio format")
                return audio_data
            
            # Check if it's raw PCM data
            elif len(audio_data) > 0 and len(audio_data) % 2 == 0:
                logger.info("Detected raw PCM audio data")
                # Convert raw PCM to WAV format for better STT processing
                return await self._convert_pcm_to_wav(audio_data)
            
            # Default: assume it's raw audio data
            else:
                logger.info("Unknown audio format, passing through")
                return audio_data
                
        except Exception as e:
            logger.error(f"Failed to preprocess audio: {e}")
            return None
    
    async def _convert_pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """Convert raw PCM data to WAV format for better STT processing."""
        try:
            import wave
            import io
            
            # Assume 16-bit PCM, 16kHz sample rate, mono
            sample_rate = 16000
            channels = 1
            sample_width = 2  # 16-bit = 2 bytes
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data)
            
            wav_data = wav_buffer.getvalue()
            wav_buffer.close()
            
            logger.info(f"Converted PCM to WAV: {len(pcm_data)} -> {len(wav_data)} bytes")
            return wav_data
            
        except Exception as e:
            logger.error(f"Failed to convert PCM to WAV: {e}")
            return pcm_data  # Return original data as fallback

    async def send_tts_audio(self, room_id: str, audio_data: bytes):
        """Send TTS audio to a LiveKit room."""
        try:
            room = self.rooms.get(room_id)
            if room and room.local_participant:
                # Create audio track from TTS data
                audio_track = rtc.LocalAudioTrack.create_audio_track(
                    "ai-voice", audio_data
                )
                
                # Publish to room
                await room.local_participant.publish_track(audio_track)
                logger.info(f"Sent TTS audio to room {room_id}: {len(audio_data)} bytes")
                
        except Exception as e:
            logger.error(f"Failed to send TTS audio to room {room_id}: {e}")
    
    async def send_tts_audio_to_websocket(self, websocket, audio_data: bytes):
        """Send TTS audio directly to WebSocket client."""
        try:
            # Send binary audio data to WebSocket
            await websocket.send_bytes(audio_data)
            logger.info(f"Sent TTS audio to WebSocket: {len(audio_data)} bytes")
            
        except Exception as e:
            logger.error(f"Failed to send TTS audio to WebSocket: {e}")

    async def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Handle participant connection."""
        logger.info(f"Participant connected: {participant.identity}")
        self.participants[participant.identity] = participant

        # Publish event
        await self.event_publisher.publish_event(
            {
                "type": "participant_connected",
                "participant_id": participant.identity,
                "room_id": participant.room.name,
            }
        )

    async def _on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """Handle participant disconnection."""
        logger.info(f"Participant disconnected: {participant.identity}")
        self.participants.pop(participant.identity, None)

        # Publish event
        await self.event_publisher.publish_event(
            {
                "type": "participant_disconnected",
                "participant_id": participant.identity,
                "room_id": participant.room.name,
            }
        )

    async def _on_track_subscribed(
        self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        """Handle track subscription (audio/video)."""
        logger.info(f"Track subscribed: {track.kind} from {participant.identity}")
        
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            # Handle audio track for STT processing
            track.on("data", lambda data: self._on_audio_data(data, participant.identity))
            
            # Enable audio track for processing
            await track.set_enabled(True)
            logger.info(f"Enabled audio track from {participant.identity}")
            
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            # Handle video track if needed
            await track.set_enabled(True)
            logger.info(f"Enabled video track from {participant.identity}")

    async def _on_track_unsubscribed(
        self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        """Handle track unsubscription."""
        logger.info(f"Track unsubscribed: {track.kind} from {participant.identity}")
        
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            # Clean up audio processing for this participant
            logger.info(f"Cleaned up audio processing for {participant.identity}")
            
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            # Clean up video processing if needed
            logger.info(f"Cleaned up video processing for {participant.identity}")

    async def _on_data_received(self, data: bytes, participant: rtc.RemoteParticipant):
        """Handle data channel messages."""
        logger.info(f"Data received from {participant.identity}: {len(data)} bytes")
        # Handle text messages or control data

    async def _on_audio_data(self, audio_frame: rtc.AudioFrame, participant_id: str):
        """Handle incoming audio data for STT processing."""
        try:
            # Convert audio frame to bytes
            audio_data = audio_frame.data.tobytes()
            
            # Process through the audio pipeline
            await self.process_audio_data(participant_id, audio_data)
            
        except Exception as e:
            logger.error(f"Failed to process audio data from {participant_id}: {e}")

    async def cleanup(self):
        """Clean up LiveKit connections."""
        try:
            for room_id, room in self.rooms.items():
                await room.disconnect()
                logger.info(f"Disconnected from room {room_id}")
            
            self.rooms.clear()
            self.participants.clear()
            self.audio_buffers.clear()
            self.chunk_count.clear()
            
        except Exception as e:
            logger.error(f"Failed to cleanup LiveKit connections: {e}")
    
    async def cleanup_conversation(self, conversation_id: str):
        """Clean up audio buffers for a specific conversation."""
        try:
            if conversation_id in self.audio_buffers:
                del self.audio_buffers[conversation_id]
            if conversation_id in self.chunk_count:
                del self.chunk_count[conversation_id]
            logger.info(f"Cleaned up audio buffers for conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup conversation {conversation_id}: {e}")