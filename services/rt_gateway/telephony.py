"""
LiveKit Voice telephony integration for handling phone calls.
"""
import asyncio
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import uuid4, UUID

from fastapi import HTTPException, Request
from livekit import api

from shared.config import get_settings
from shared.schemas import Conversation, ConversationStatus, Turn, TurnType

logger = logging.getLogger(__name__)
settings = get_settings()


class TelephonyHandler:
    """Handles LiveKit Voice telephony webhooks and call management."""

    def __init__(self, turn_manager, livekit_bridge, event_publisher):
        self.turn_manager = turn_manager
        self.livekit_bridge = livekit_bridge
        self.event_publisher = event_publisher
        self.active_calls: Dict[str, Dict[str, Any]] = {}  # call_id -> call_data
        self.api_client = None

    async def initialize(self):
        """Initialize LiveKit Voice API client."""
        try:
            if not all([settings.livekit_api_key, settings.livekit_api_secret]):
                logger.warning("LiveKit Voice credentials not configured")
                return

            self.api_client = api.LiveKitAPI(
                url=settings.livekit_url,
                api_key=settings.livekit_api_key,
                api_secret=settings.livekit_api_secret,
            )
            logger.info("LiveKit Voice API client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize LiveKit Voice API: {e}")
            raise

    def verify_webhook_signature(self, request: Request, body: bytes) -> bool:
        """Verify webhook signature for security."""
        try:
            if not settings.livekit_voice_webhook_secret:
                logger.warning("Webhook secret not configured, skipping signature verification")
                return True

            signature = request.headers.get("x-livekit-signature")
            if not signature:
                logger.error("Missing webhook signature")
                return False

            expected_signature = hmac.new(
                settings.livekit_voice_webhook_secret.encode(),
                body,
                hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(signature, expected_signature)

        except Exception as e:
            logger.error(f"Failed to verify webhook signature: {e}")
            return False

    async def handle_incoming_call(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming phone call webhook."""
        try:
            call_id = webhook_data.get("call_id")
            caller_number = webhook_data.get("caller_number")
            called_number = webhook_data.get("called_number")
            
            logger.info(f"Incoming call: {call_id} from {caller_number} to {called_number}")

            # Create conversation for this call
            conversation = await self._create_call_conversation(call_id, caller_number, called_number)
            
            # Store call data
            self.active_calls[call_id] = {
                "conversation_id": str(conversation.id),
                "caller_number": caller_number,
                "called_number": called_number,
                "start_time": datetime.now(timezone.utc),
                "status": "active"
            }

            # Create LiveKit room for this call
            room_id = f"call_{call_id}"
            await self.livekit_bridge.join_room(room_id, "AI Receptionist")

            # Send greeting
            await self._send_call_greeting(conversation, room_id)

            # Return response to LiveKit Voice
            return {
                "room": room_id,
                "participant_identity": "ai-receptionist",
                "participant_name": "AI Receptionist",
                "participant_metadata": json.dumps({
                    "conversation_id": str(conversation.id),
                    "call_id": call_id,
                    "caller_number": caller_number
                })
            }

        except Exception as e:
            logger.error(f"Failed to handle incoming call: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def handle_call_ended(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call ended webhook."""
        try:
            call_id = webhook_data.get("call_id")
            reason = webhook_data.get("reason", "unknown")
            
            logger.info(f"Call ended: {call_id}, reason: {reason}")

            # Update call status
            if call_id in self.active_calls:
                self.active_calls[call_id]["status"] = "ended"
                self.active_calls[call_id]["end_time"] = datetime.now(timezone.utc)
                self.active_calls[call_id]["end_reason"] = reason

                # Clean up LiveKit room
                room_id = f"call_{call_id}"
                await self.livekit_bridge.leave_room(room_id)

                # Update conversation status
                conversation_id = self.active_calls[call_id]["conversation_id"]
                await self._end_call_conversation(conversation_id, reason)

                # Remove from active calls
                del self.active_calls[call_id]

            return {"status": "success"}

        except Exception as e:
            logger.error(f"Failed to handle call ended: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def handle_call_answered(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call answered webhook."""
        try:
            call_id = webhook_data.get("call_id")
            participant_identity = webhook_data.get("participant_identity")
            
            logger.info(f"Call answered: {call_id} by {participant_identity}")

            if call_id in self.active_calls:
                self.active_calls[call_id]["answered_at"] = datetime.now(timezone.utc)
                self.active_calls[call_id]["participant_identity"] = participant_identity

            return {"status": "success"}

        except Exception as e:
            logger.error(f"Failed to handle call answered: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def handle_call_failed(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call failed webhook."""
        try:
            call_id = webhook_data.get("call_id")
            error = webhook_data.get("error", "unknown")
            
            logger.error(f"Call failed: {call_id}, error: {error}")

            if call_id in self.active_calls:
                self.active_calls[call_id]["status"] = "failed"
                self.active_calls[call_id]["error"] = error
                self.active_calls[call_id]["failed_at"] = datetime.now(timezone.utc)

                # Clean up
                room_id = f"call_{call_id}"
                await self.livekit_bridge.leave_room(room_id)
                del self.active_calls[call_id]

            return {"status": "success"}

        except Exception as e:
            logger.error(f"Failed to handle call failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _create_call_conversation(self, call_id: str, caller_number: str, called_number: str) -> Conversation:
        """Create a conversation for the phone call."""
        try:
            conversation_id = uuid4()
            
            conversation = Conversation(
                id=conversation_id,
                room_id=f"call_{call_id}",
                user_id=f"caller_{caller_number}",
                status=ConversationStatus.ACTIVE,
                metadata={
                    "call_id": call_id,
                    "caller_number": caller_number,
                    "called_number": called_number,
                    "call_type": "phone",
                    "start_time": datetime.now(timezone.utc).isoformat()
                }
            )

            # Add to turn manager
            self.turn_manager.conversations[str(conversation_id)] = conversation
            self.turn_manager.conversation_turns[str(conversation_id)] = []

            # Publish event
            await self.event_publisher.publish_event({
                "type": "call_started",
                "conversation_id": str(conversation_id),
                "call_id": call_id,
                "caller_number": caller_number,
                "called_number": called_number
            })

            logger.info(f"Created conversation {conversation_id} for call {call_id}")
            return conversation

        except Exception as e:
            logger.error(f"Failed to create call conversation: {e}")
            raise

    async def _end_call_conversation(self, conversation_id: str, reason: str):
        """End the call conversation."""
        try:
            conversation = self.turn_manager.conversations.get(conversation_id)
            if conversation:
                conversation.status = ConversationStatus.ENDED
                conversation.metadata["end_time"] = datetime.now(timezone.utc).isoformat()
                conversation.metadata["end_reason"] = reason

                # Publish event
                await self.event_publisher.publish_event({
                    "type": "call_ended",
                    "conversation_id": conversation_id,
                    "reason": reason
                })

                logger.info(f"Ended conversation {conversation_id}")

        except Exception as e:
            logger.error(f"Failed to end call conversation: {e}")

    async def _send_call_greeting(self, conversation: Conversation, room_id: str):
        """Send greeting message for the call."""
        try:
            greeting = "Hello! Thank you for calling. I'm your AI assistant. How can I help you today?"
            
            # Create greeting turn
            greeting_turn = await self.turn_manager.create_turn(
                conversation_id=str(conversation.id),
                content=greeting,
                turn_type=TurnType.AI_RESPONSE,
            )

            # Send TTS response
            await self.turn_manager._send_tts_response(conversation, greeting)

            logger.info(f"Sent greeting for call {conversation.room_id}")

        except Exception as e:
            logger.error(f"Failed to send call greeting: {e}")

    async def get_call_status(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active call."""
        return self.active_calls.get(call_id)

    async def get_active_calls(self) -> Dict[str, Dict[str, Any]]:
        """Get all active calls."""
        return self.active_calls.copy()

    async def cleanup_expired_calls(self, max_duration_minutes: int = 60):
        """Clean up expired calls."""
        try:
            current_time = datetime.now(timezone.utc)
            expired_calls = []

            for call_id, call_data in self.active_calls.items():
                start_time = call_data.get("start_time", current_time)
                duration = (current_time - start_time).total_seconds() / 60
                
                if duration > max_duration_minutes:
                    expired_calls.append(call_id)

            for call_id in expired_calls:
                logger.info(f"Cleaning up expired call: {call_id}")
                await self.handle_call_ended({"call_id": call_id, "reason": "timeout"})

        except Exception as e:
            logger.error(f"Failed to cleanup expired calls: {e}")

    async def hangup_call(self, call_id: str) -> bool:
        """Hang up an active call."""
        try:
            if call_id not in self.active_calls:
                logger.warning(f"Call {call_id} not found in active calls")
                return False

            # Use LiveKit Voice API to hang up the call
            if self.api_client:
                # This would require LiveKit Voice API call to hang up
                # Implementation depends on LiveKit Voice API specifics
                logger.info(f"Hanging up call {call_id}")
                
                # Clean up locally
                await self.handle_call_ended({"call_id": call_id, "reason": "manual_hangup"})
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to hangup call {call_id}: {e}")
            return False
