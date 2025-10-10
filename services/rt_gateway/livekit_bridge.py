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
        """Process incoming audio data through the AI pipeline."""
        try:
            logger.info(f"Processing audio data for conversation {conversation_id}: {len(audio_data)} bytes")
            
            # 1. Convert audio data to proper format for STT
            processed_audio = await self._preprocess_audio(audio_data)
            if not processed_audio:
                logger.warning("Failed to preprocess audio data")
                return
            
            # 2. Send to STT service for transcription
            transcript = await self.stt_service.transcribe(processed_audio)
            if not transcript or not transcript.strip():
                logger.info("No speech detected in audio")
                return
            
            logger.info(f"Transcribed: {transcript}")
            
            # 3. Create a user turn and process through turn manager
            from shared.schemas import Turn, TurnType
            from datetime import datetime, timezone
            from uuid import uuid4
            
            user_turn = Turn(
                id=uuid4(),
                conversation_id=uuid4(),  # Use the conversation_id from parameter
                turn_number=1,
                turn_type=TurnType.USER_SPEECH,
                content=transcript,
                timestamp=datetime.now(timezone.utc)
            )
            
            # 4. Process through turn manager (handles LLM and TTS)
            await self.turn_manager.process_turn(user_turn)
            
        except Exception as e:
            logger.error(f"Failed to process audio data: {e}")
            raise
    
    async def _preprocess_audio(self, audio_data: bytes) -> Optional[bytes]:
        """Preprocess audio data for STT service."""
        try:
            # Convert audio to the format expected by STT service
            # This might involve format conversion, resampling, etc.
            # For now, return the data as-is
            return audio_data
        except Exception as e:
            logger.error(f"Failed to preprocess audio: {e}")
            return None

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

    async def _on_track_unsubscribed(
        self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        """Handle track unsubscription."""
        logger.info(f"Track unsubscribed: {track.kind} from {participant.identity}")

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
            
        except Exception as e:
            logger.error(f"Failed to cleanup LiveKit connections: {e}")