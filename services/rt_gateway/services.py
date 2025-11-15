"""
Global services for RT Gateway.
These are initialized in the unified app lifespan and made available to routes.
"""
import json
import logging
from typing import Any, Dict, Optional

from .events import EventPublisher
from .external_turn_manager import ExternalTurnManager
from .llm import LLMService
from .stt import STTService
from .tts import TTSService
from .turn_manager import TurnManager
from .state_manager import get_state_manager, StateManager
from .realtime_api import RealtimeAPIService

logger = logging.getLogger(__name__)

# Global service instances (initialized by unified_app)
turn_manager: Optional[TurnManager] = None
external_turn_manager: Optional[ExternalTurnManager] = None
stt_service: Optional[STTService] = None
tts_service: Optional[TTSService] = None
llm_service: Optional[LLMService] = None
event_publisher: Optional[EventPublisher] = None
state_manager: Optional[StateManager] = None
realtime_api_service: Optional[RealtimeAPIService] = None

# Legacy in-memory state (kept for backward compatibility during migration)
# TODO: Remove after full migration to Redis
active_conversations: dict = {}
active_bot_sessions: dict = {}
bot_session_metadata: dict = {}


async def broadcast_to_conversation(conversation_id: str, message: Dict[str, Any]):
    """Broadcast a message to all participants in a conversation using Redis pub/sub."""
    try:
        # Use Redis pub/sub if available
        if state_manager:
            await state_manager.broadcast_to_conversation(conversation_id, message)
        else:
            # Fallback to legacy method
            if turn_manager is None:
                logger.warning("Turn manager is not initialized, cannot broadcast message")
                return
            
            if conversation_id in active_conversations:
                conversation = active_conversations[conversation_id]
                participants = []
                if conversation_id in turn_manager.conversation_participants:
                    participants = turn_manager.conversation_participants[conversation_id]
                # Send to all participants in the conversation
                for participant_id in participants:
                    if participant_id in turn_manager.conversation_websockets:
                        websocket = turn_manager.conversation_websockets[participant_id]
                        try:
                            await websocket.send_text(json.dumps(message))
                        except Exception as e:
                            logger.warning(f"Failed to send message to participant {participant_id}: {e}")
    except Exception as e:
        logger.error(f"Failed to broadcast message to conversation {conversation_id}: {e}")
