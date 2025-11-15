"""
Turn management and dialogue orchestration.
"""
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from uuid import UUID

from shared.config import get_settings
from shared.schemas import (Action, ActionStatus, ActionType, Conversation,
                            ConversationStatus, Turn, TurnType)

if TYPE_CHECKING:
    from services.api.dao import DynamoDBDAO

logger = logging.getLogger(__name__)
settings = get_settings()


class TurnManager:
    """Manages conversation turns and dialogue flow."""

    def __init__(self, llm_service, event_publisher, tts_service=None, dao: Optional[Any] = None):
        self.llm_service = llm_service
        self.event_publisher = event_publisher
        self.tts_service = tts_service
        # Use injected DAO or get from global instance
        if dao is not None:
            self._dao = dao
        else:
            # Try to get from global instance (initialized in unified_app)
            try:
                from services.api.dao import get_dao
                self._dao = get_dao()
            except RuntimeError:
                # DAO not initialized yet, will be set later
                self._dao = None
        self.conversations: Dict[str, Conversation] = {}
        self.conversation_turns: Dict[str, List[Turn]] = {}
        self.conversation_state: Dict[str, str] = {}  # conversation_id -> state
        self.conversation_websockets: Dict[str, any] = {}  # conversation_id -> websocket
        self.conversation_participants: Dict[str, List[str]] = {}

    def register_websocket(self, conversation_id: str, websocket):
        """Register a WebSocket connection for a conversation."""
        self.conversation_websockets[conversation_id] = websocket
        logger.info(f"Registered WebSocket for conversation {conversation_id}")

    def unregister_websocket(self, conversation_id: str):
        """Unregister a WebSocket connection for a conversation."""
        if conversation_id in self.conversation_websockets:
            del self.conversation_websockets[conversation_id]
            logger.info(f"Unregistered WebSocket for conversation {conversation_id}")

    async def start_conversation(
        self,
        room_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """Start a new conversation."""
        try:
            # Convert user_id to UUID if it's not already a valid UUID
            try:
                user_uuid = UUID(user_id)
            except ValueError:
                # If user_id is not a valid UUID, generate a new one
                from uuid import uuid4
                user_uuid = uuid4()
                logger.info(f"Generated new UUID for user_id: {user_uuid}")
            conversation = Conversation(
                user_id=user_uuid,
                room_id=room_id,
                status=ConversationStatus.ACTIVE,
            )

            if metadata:
                conversation.metadata.update(metadata)

            self.conversations[str(conversation.id)] = conversation
            self.conversation_turns[str(conversation.id)] = []
            self.conversation_state[str(conversation.id)] = "greeting"
            self.conversation_participants[str(conversation.id)] = [user_id]

            # Publish event
            await self.event_publisher.publish_event(
                {
                    "type": "conversation_started",
                    "conversation_id": str(conversation.id),
                    "room_id": room_id,
                    "user_id": user_id,
                }
            )

            # Send greeting
            await self._send_greeting(conversation)

            logger.info(f"Started conversation {conversation.id}")
            return conversation

        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            raise

    async def end_conversation(self, conversation_id: str):
        """End a conversation."""
        try:
            conversation = self.conversations.get(conversation_id)
            if conversation:
                conversation.status = ConversationStatus.COMPLETED
                conversation.ended_at = datetime.utcnow()

                # Publish event (with error handling)
                try:
                    await self.event_publisher.publish_event(
                        {
                            "type": "conversation_ended",
                            "conversation_id": conversation_id,
                            "duration": (
                                conversation.ended_at - conversation.started_at
                            ).total_seconds(),
                        }
                    )
                except Exception as event_error:
                    logger.warning(f"Failed to publish conversation_ended event: {event_error}")

                # Cleanup
                if conversation_id in self.conversation_turns:
                    del self.conversation_turns[conversation_id]
                if conversation_id in self.conversation_state:
                    del self.conversation_state[conversation_id]
                if conversation_id in self.conversation_websockets:
                    del self.conversation_websockets[conversation_id]
                if conversation_id in self.conversation_participants:
                    del self.conversation_participants[conversation_id]

                # Audio buffers are managed internally, no external cleanup needed

                # Update conversation in database
                logger.info(f"Ended conversation {conversation_id}")
            else:
                logger.warning(f"Conversation {conversation_id} not found in turn_manager, but proceeding with cleanup")
                
                # Still clean up any remaining state
                if conversation_id in self.conversation_turns:
                    del self.conversation_turns[conversation_id]
                if conversation_id in self.conversation_state:
                    del self.conversation_state[conversation_id]

        except Exception as e:
            logger.error(f"Failed to end conversation {conversation_id}: {e}")
            raise

    async def add_participant_to_conversation(self, conversation_id: str, user_id: str):
        """Add a participant to an existing conversation."""
        try:
            conversation = self.conversations.get(conversation_id)
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found")
                return False

            participants = self.conversation_participants.setdefault(conversation_id, [])
            # Add user to conversation participants
            if user_id not in participants:
                participants.append(user_id)
                logger.info(f"Added participant {user_id} to conversation {conversation_id}")
                return True
            else:
                logger.info(f"Participant {user_id} already in conversation {conversation_id}")
                return True
            
        except Exception as e:
            logger.error(f"Error adding participant to conversation {conversation_id}: {e}")
            raise

    async def create_turn(
        self,
        conversation_id: str,
        content: str,
        turn_type: TurnType,
        confidence_score: Optional[float] = None,
    ) -> Turn:
        """Create a new turn."""
        try:
            conversation = self.conversations.get(conversation_id)
            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found")

            turn_number = len(self.conversation_turns.get(conversation_id, [])) + 1

            turn = Turn(
                conversation_id=UUID(conversation_id),
                turn_number=turn_number,
                turn_type=turn_type,
                content=content,
                confidence_score=confidence_score,
            )

            # Store turn
            if conversation_id not in self.conversation_turns:
                self.conversation_turns[conversation_id] = []
            self.conversation_turns[conversation_id].append(turn)

            # Publish event
            await self.event_publisher.publish_event(
                {
                    "type": "turn_created",
                    "conversation_id": conversation_id,
                    "turn_id": str(turn.id),
                    "turn_type": turn_type.value,
                    "content": content,
                }
            )

            logger.info(f"Created turn {turn.id} for conversation {conversation_id}")
            return turn

        except Exception as e:
            logger.error(f"Failed to create turn: {e}")
            raise

    async def process_turn(self, turn: Turn):
        """Process a turn through the dialogue system."""
        try:
            conversation_id = str(turn.conversation_id)
            conversation = self.conversations.get(conversation_id)

            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found")

            # Get conversation context
            context = await self._build_context(conversation_id)

            meeting_id = conversation.metadata.get("meeting_id") if conversation.metadata else None

            # Process through LLM
            response = await self.llm_service.process_turn(
                turn,
                context,
                meeting_id=meeting_id,
            )

            if response:
                # Create AI response turn
                ai_turn = await self.create_turn(
                    conversation_id=conversation_id,
                    content=response.content,
                    turn_type=TurnType.AI_RESPONSE,
                )

                # Handle tool calls if any
                if response.tool_calls:
                    await self._handle_tool_calls(
                        conversation_id, str(ai_turn.id), response.tool_calls
                    )

                # Update conversation state
                await self._update_conversation_state(conversation_id, turn, ai_turn)

                # Send TTS response
                await self._send_tts_response(conversation, response.content)

        except Exception as e:
            logger.error(f"Failed to process turn {turn.id}: {e}")
            # Create error turn
            await self.create_turn(
                conversation_id=str(turn.conversation_id),
                content="I'm sorry, I encountered an error. Please try again.",
                turn_type=TurnType.ERROR,
            )

    async def process_turn_and_get_response(self, turn: Turn) -> Optional[Union[str, tuple]]:
        """Process a turn and return the AI response."""
        try:
            conversation_id = str(turn.conversation_id)
            conversation = self.conversations.get(conversation_id)

            logger.info(f"Looking for conversation {conversation_id}")
            logger.info(f"Available conversations: {list(self.conversations.keys())}")

            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found")

            # Get conversation context
            context = await self._build_context(conversation_id)

            meeting_id = conversation.metadata.get("meeting_id") if conversation.metadata else None

            # Process through LLM
            response = await self.llm_service.process_turn(
                turn,
                context,
                meeting_id=meeting_id,
            )

            if response:
                # Create AI response turn
                ai_turn = await self.create_turn(
                    conversation_id=conversation_id,
                    content=response.content,
                    turn_type=TurnType.AI_RESPONSE,
                )

                # Handle tool calls if any
                if response.tool_calls:
                    await self._handle_tool_calls(
                        conversation_id, str(ai_turn.id), response.tool_calls
                    )

                # Update conversation state
                await self._update_conversation_state(conversation_id, turn, ai_turn)

                # Send TTS response
                await self._send_tts_response(conversation, response.content)

                # Return text response
                return response.content

        except Exception as e:
            logger.error(f"Failed to process turn {turn.id}: {e}")
            return "I'm sorry, I encountered an error. Please try again."
        
        return None

    async def _build_context(self, conversation_id: str) -> List[Dict[str, str]]:
        """Build conversation context for LLM."""
        turns = self.conversation_turns.get(conversation_id, [])
        context = []

        for turn in turns[-10:]:  # Last 10 turns for context
            role = "user" if turn.turn_type == TurnType.USER_SPEECH else "assistant"
            context.append({"role": role, "content": turn.content})

        return context

    async def _handle_tool_calls(
        self, conversation_id: str, turn_id: str, tool_calls: List[Dict]
    ):
        """Handle tool calls from LLM response."""
        try:
            for tool_call in tool_calls:
                # Parse arguments if they're a string
                parameters = tool_call["function"]["arguments"]
                if isinstance(parameters, str):
                    try:
                        import json
                        parameters = json.loads(parameters)
                        logger.info(f"Successfully parsed tool call arguments: {parameters}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse tool call arguments as JSON: {e}")
                        logger.error(f"Problematic arguments string: {repr(parameters)}")
                        parameters = {"raw_arguments": parameters}
                elif isinstance(parameters, dict):
                    logger.info(f"Tool call arguments already in dict format: {parameters}")
                else:
                    logger.warning(f"Unexpected parameters type: {type(parameters)}, value: {parameters}")
                    parameters = {"raw_arguments": str(parameters)}
                
                # Map LLM function names to ActionType enum values
                function_name = tool_call["function"]["name"]
                action_type_mapping = {
                    "create_calendar_event": ActionType.CALENDAR_CREATE,
                    "create_zoom_meeting": ActionType.CALENDAR_CREATE,  # Zoom meetings are also calendar events
                    "update_calendar_event": ActionType.CALENDAR_UPDATE,
                    "delete_calendar_event": ActionType.CALENDAR_DELETE,
                    "send_email": ActionType.EMAIL_SEND,
                    "send_slack_message": ActionType.SLACK_MESSAGE,
                    "update_crm": ActionType.CRM_UPDATE,
                    "search_knowledge": ActionType.KNOWLEDGE_SEARCH,
                }
                
                action_type = action_type_mapping.get(function_name)
                if not action_type:
                    logger.warning(f"Unknown function name: {function_name}")
                    continue
                
                action = Action(
                    conversation_id=UUID(conversation_id),
                    turn_id=UUID(turn_id),
                    action_type=action_type,
                    parameters=parameters,
                    status=ActionStatus.PENDING,
                )

                # Publish action to SQS for processing
                await self.event_publisher.publish_action(action)

        except Exception as e:
            logger.error(f"Failed to handle tool calls: {e}")
            logger.error(f"Tool call data: {tool_calls}")

    async def _update_conversation_state(
        self, conversation_id: str, user_turn: Turn, ai_turn: Turn
    ):
        """Update conversation state based on turns."""
        # Simple state machine - can be enhanced
        current_state = self.conversation_state.get(conversation_id, "active")

        # Update state based on content analysis
        user_content = user_turn.content.lower()

        if "goodbye" in user_content or "bye" in user_content:
            self.conversation_state[conversation_id] = "ending"
        elif "help" in user_content:
            self.conversation_state[conversation_id] = "help"
        else:
            self.conversation_state[conversation_id] = "active"

    async def _send_greeting(self, conversation: Conversation):
        """Send initial greeting."""
        greeting = "Hello! I'm your AI assistant. How can I help you today?"

        greeting_turn = await self.create_turn(
            conversation_id=str(conversation.id),
            content=greeting,
            turn_type=TurnType.AI_RESPONSE,
        )

        # TTS removed - text-only communication

    async def _send_tts_response(self, conversation: Conversation, text: str):
        """Send text-to-speech response."""
        try:
            logger.info(f"Sending TTS for conversation {conversation.id}: {text}")
            
            # Generate TTS audio using the TTS service
            from shared.schemas import TTSRequest
            
            # Get the latest turn number for this conversation
            conversation_id = str(conversation.id)
            turns = self.conversation_turns.get(conversation_id, [])
            turn_number = len(turns) + 1 if turns else 1
            
            tts_request = TTSRequest(
                text=text,
                voice_id="default",
                conversation_id=conversation.id,
                turn_number=turn_number
            )
            
            tts_response = await self.tts_service.synthesize(tts_request)
            
            if tts_response and tts_response.audio_data:
                logger.info(f"Generated TTS audio: {len(tts_response.audio_data)} bytes")
                
                # Send audio to WebSocket if available
                conversation_id = str(conversation.id)
                websocket = self.conversation_websockets.get(conversation_id)
                
                if websocket:
                    # Send TTS audio directly to WebSocket
                    try:
                        await websocket.send_bytes(tts_response.audio_data)
                        logger.info(f"Sent TTS audio to WebSocket for conversation {conversation_id}")
                    except Exception as ws_error:
                        logger.error(f"Failed to send TTS to WebSocket: {ws_error}")
            else:
                logger.warning("No TTS audio generated")
            
        except Exception as e:
            logger.error(f"Failed to send TTS response: {e}")

    async def load_existing_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load an existing conversation from the database."""
        try:
            # Use the DAO instance from TurnManager (already initialized)
            if self._dao is None:
                from services.api.dao import get_dao
                self._dao = get_dao()
            
            dao = self._dao
            conversation = await dao.get_conversation(conversation_id)
            
            if conversation:
                # Load conversation into memory
                self.conversations[str(conversation.id)] = conversation
                
                # Load existing turns for this conversation
                turns = await dao.get_conversation_turns(conversation_id)
                self.conversation_turns[str(conversation.id)] = turns
                
                # Set conversation state
                self.conversation_state[str(conversation.id)] = "active"
                
                logger.info(f"Loaded existing conversation {conversation_id} with {len(turns)} turns")
                return conversation
            else:
                logger.warning(f"Conversation {conversation_id} not found in database")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load existing conversation {conversation_id}: {e}")
            return None

    async def _save_conversation_to_db(self, conversation: Conversation):
        """Deprecated: Conversations are managed in-memory."""
        logger.debug("Conversation persistence disabled; skipping save.")

    async def _update_conversation_in_db(self, conversation: Conversation):
        """Deprecated: Conversations are managed in-memory."""
        logger.debug("Conversation persistence disabled; skipping update.")
