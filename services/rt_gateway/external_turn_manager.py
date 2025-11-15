"""
External Turn Manager for Meeting Bot.
Originally Redis-backed, now defaults to in-memory turn management for sessions.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4, UUID

from shared.config import get_settings
from shared.schemas import Turn, TurnType

logger = logging.getLogger(__name__)
settings = get_settings()


class ExternalTurnManager:
    """
    Lightweight turn manager for meeting bot sessions.
    Uses Redis for fast turn storage and retrieval.
    """

    def __init__(self, llm_service):
        """Initialize the external turn manager."""
        self.llm_service = llm_service
        self.redis_client: Optional[Any] = None
        self.sessions: Dict[str, dict] = {}  # In-memory session metadata
        self.ttl_hours = 24  # TTL for Redis data
        
    async def initialize(self):
        """Initialize Redis connection."""
        logger.info("External turn manager running in in-memory mode")
    
    async def cleanup(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.aclose()
            logger.info("External turn manager Redis connection closed")
    
    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session metadata."""
        return f"meeting:session:{session_id}:metadata"
    
    def _get_turn_key(self, session_id: str, turn_number: int) -> str:
        """Get Redis key for a specific turn."""
        return f"meeting:session:{session_id}:turn:{turn_number}"
    
    def _get_turn_list_key(self, session_id: str) -> str:
        """Get Redis key for turn list."""
        return f"meeting:session:{session_id}:turns"
    
    async def start_session(self, session_id: str, meeting_id: str) -> bool:
        """
        Start a new conversation session for the meeting bot.
        
        Args:
            session_id: Unique session identifier from the bot
            meeting_id: Meeting identifier
        
        Returns:
            True if session started successfully
        """
        try:
            session_data = {
                "session_id": session_id,
                "meeting_id": meeting_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "turn_count": 0,
                "status": "active"
            }
            
            # Store in-memory for quick access
            self.sessions[session_id] = session_data
            
            # Store in Redis with TTL
            if self.redis_client:
                await self.redis_client.set(
                    self._get_session_key(session_id),
                    json.dumps(session_data),
                    ex=self.ttl_hours * 3600  # TTL in seconds
                )
                # Initialize turn list
                await self.redis_client.set(
                    self._get_turn_list_key(session_id),
                    "[]",
                    ex=self.ttl_hours * 3600
                )
            
            logger.info(f"Started session {session_id} for meeting {meeting_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start session {session_id}: {e}")
            return False
    
    async def end_session(self, session_id: str) -> bool:
        """
        End a conversation session.
        Note: Data is kept in Redis with TTL for potential recovery.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if session ended successfully
        """
        try:
            # Update session status
            if session_id in self.sessions:
                self.sessions[session_id]["status"] = "ended"
                self.sessions[session_id]["ended_at"] = datetime.now(timezone.utc).isoformat()
                
                # Update in Redis
                if self.redis_client:
                    await self.redis_client.set(
                        self._get_session_key(session_id),
                        json.dumps(self.sessions[session_id]),
                        ex=self.ttl_hours * 3600
                    )
            
            # Remove from in-memory cache
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            logger.info(f"Ended session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to end session {session_id}: {e}")
            return False
    
    async def _get_session(self, session_id: str) -> Optional[dict]:
        """Get session data from memory or Redis."""
        # Try in-memory first
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Try Redis
        if self.redis_client:
            try:
                data = await self.redis_client.get(self._get_session_key(session_id))
                if data:
                    session_data = json.loads(data)
                    # Restore to in-memory cache
                    self.sessions[session_id] = session_data
                    return session_data
            except Exception as e:
                logger.error(f"Failed to load session from Redis: {e}")
        
        return None
    
    async def _save_turn(self, session_id: str, turn: Turn) -> bool:
        """Save a turn to Redis and increment turn count."""
        try:
            # Get current session
            session = await self._get_session(session_id)
            if not session:
                # Auto-create session if it doesn't exist
                logger.warning(f"Session {session_id} not found, creating...")
                await self.start_session(session_id, "unknown")
                session = await self._get_session(session_id)
            
            if not session:
                logger.error(f"Failed to create session {session_id}")
                return False
            
            # Increment turn count
            turn_number = session["turn_count"] + 1
            session["turn_count"] = turn_number
            
            # Create turn data
            turn_data = {
                "id": str(turn.id),
                "turn_number": turn_number,
                "turn_type": turn.turn_type.value,
                "content": turn.content,
                "confidence_score": turn.confidence_score,
                "timestamp": turn.timestamp.isoformat()
            }
            
            # Store in Redis
            if self.redis_client:
                # Store individual turn
                await self.redis_client.set(
                    self._get_turn_key(session_id, turn_number),
                    json.dumps(turn_data),
                    ex=self.ttl_hours * 3600
                )
                
                # Update session metadata
                await self.redis_client.set(
                    self._get_session_key(session_id),
                    json.dumps(session),
                    ex=self.ttl_hours * 3600
                )
                
                # Add to turn list
                turn_list_data = await self.redis_client.get(self._get_turn_list_key(session_id))
                turn_list = json.loads(turn_list_data) if turn_list_data else []
                turn_list.append({
                    "turn_number": turn_number,
                    "turn_key": self._get_turn_key(session_id, turn_number)
                })
                await self.redis_client.set(
                    self._get_turn_list_key(session_id),
                    json.dumps(turn_list),
                    ex=self.ttl_hours * 3600
                )
            
            # Update in-memory cache
            self.sessions[session_id] = session
            
            logger.debug(f"Saved turn {turn_number} for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save turn for session {session_id}: {e}")
            return False
    
    async def _get_recent_turns(self, session_id: str, limit: int = 10) -> List[Turn]:
        """
        Get recent turns for building conversation context.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of recent turns to retrieve
        
        Returns:
            List of Turn objects
        """
        try:
            turns = []
            
            # Get turn list from Redis
            if self.redis_client:
                turn_list_data = await self.redis_client.get(self._get_turn_list_key(session_id))
                if turn_list_data:
                    turn_list = json.loads(turn_list_data)
                    # Get most recent turns
                    recent_turns = turn_list[-limit:]
                    
                    # Fetch each turn
                    for turn_info in recent_turns:
                        turn_data = await self.redis_client.get(turn_info["turn_key"])
                        if turn_data:
                            turn_dict = json.loads(turn_data)
                            # Reconstruct Turn object
                            turn = Turn(
                                id=UUID(turn_dict["id"]),
                                conversation_id=uuid4(),  # Dummy UUID for meeting sessions
                                turn_number=turn_dict["turn_number"],
                                turn_type=TurnType(turn_dict["turn_type"]),
                                content=turn_dict["content"],
                                confidence_score=turn_dict.get("confidence_score"),
                                timestamp=datetime.fromisoformat(turn_dict["timestamp"])
                            )
                            turns.append(turn)
            
            # Sort by turn number
            turns.sort(key=lambda t: t.turn_number)
            return turns
            
        except Exception as e:
            logger.error(f"Failed to get recent turns for session {session_id}: {e}")
            return []
    
    async def _build_context(self, session_id: str) -> List[Dict[str, str]]:
        """
        Build conversation context from recent turns.
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of message dicts for LLM context
        """
        try:
            turns = await self._get_recent_turns(session_id, limit=20)
            context = []
            
            for turn in turns:
                if turn.turn_type == TurnType.USER_SPEECH:
                    context.append({"role": "user", "content": turn.content})
                elif turn.turn_type == TurnType.AI_RESPONSE:
                    context.append({"role": "assistant", "content": turn.content})
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to build context for session {session_id}: {e}")
            return []
    
    async def process_turn(
        self,
        session_id: str,
        user_text: str,
        meeting_id: str,
        confidence_score: Optional[float] = None
    ) -> Optional[str]:
        """
        Process a user turn and return AI response.
        This is the main method called by the meeting bot.
        
        Args:
            session_id: Session identifier
            user_text: Transcribed user speech
            meeting_id: Meeting identifier
            confidence_score: Optional STT confidence score
        
        Returns:
            AI response text, or None if processing failed
        """
        try:
            # Ensure session exists
            session = await self._get_session(session_id)
            if not session:
                logger.info(f"Starting new session {session_id}")
                await self.start_session(session_id, meeting_id)
            
            # Create user turn
            user_turn = Turn(
                id=uuid4(),
                conversation_id=uuid4(),  # Dummy UUID for meetings
                turn_number=0,  # Will be set by _save_turn
                turn_type=TurnType.USER_SPEECH,
                content=user_text,
                confidence_score=confidence_score,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Save user turn
            await self._save_turn(session_id, user_turn)
            
            # Build context
            context = await self._build_context(session_id)
            
            # Get LLM response
            logger.info(f"🤖 Generating LLM response for session {session_id}")
            full_response = ""
            async for llm_chunk in self.llm_service.generate_response_streaming(
                user_text,
                conversation_context=context
            ):
                full_response += llm_chunk
            
            # Create AI turn
            ai_turn = Turn(
                id=uuid4(),
                conversation_id=uuid4(),  # Dummy UUID for meetings
                turn_number=0,  # Will be set by _save_turn
                turn_type=TurnType.AI_RESPONSE,
                content=full_response,
                confidence_score=None,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Save AI turn
            await self._save_turn(session_id, ai_turn)
            
            logger.info(f"✅ Processed turn for session {session_id}")
            return full_response
            
        except Exception as e:
            logger.error(f"Failed to process turn for session {session_id}: {e}", exc_info=True)
            return None
    
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, any]]:
        """
        Get full conversation history for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of conversation messages
        """
        try:
            turns = await self._get_recent_turns(session_id, limit=1000)
            history = []
            
            for turn in turns:
                history.append({
                    "turn_number": turn.turn_number,
                    "type": turn.turn_type.value,
                    "content": turn.content,
                    "timestamp": turn.timestamp.isoformat(),
                    "confidence": turn.confidence_score
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history for session {session_id}: {e}")
            return []

