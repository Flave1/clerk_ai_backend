"""Unit tests for shared schemas."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4
from shared.schemas import Conversation, Turn, Action, User, RoomInfo, ConversationStatus, TurnType, ActionStatus, ActionType


class TestConversation:
    """Test Conversation schema."""

    def test_conversation_creation(self):
        """Test creating a conversation."""
        user_id = uuid4()
        conv = Conversation(
            user_id=user_id,
            room_id="room-456",
            status=ConversationStatus.ACTIVE,
        )
        assert conv.user_id == user_id
        assert conv.room_id == "room-456"
        assert conv.status == ConversationStatus.ACTIVE

    def test_conversation_serialization(self):
        """Test conversation serialization."""
        user_id = uuid4()
        conv = Conversation(
            user_id=user_id,
            room_id="room-456",
            status=ConversationStatus.ACTIVE,
        )
        data = conv.model_dump()
        assert "user_id" in data
        assert "room_id" in data
        assert "status" in data


class TestTurn:
    """Test Turn schema."""

    def test_turn_creation(self):
        """Test creating a turn."""
        conversation_id = uuid4()
        turn = Turn(
            conversation_id=conversation_id,
            turn_number=1,
            turn_type=TurnType.USER_SPEECH,
            content="Hello, world!",
        )
        assert turn.conversation_id == conversation_id
        assert turn.turn_type == TurnType.USER_SPEECH
        assert turn.content == "Hello, world!"


class TestAction:
    """Test Action schema."""

    def test_action_creation(self):
        """Test creating an action."""
        action_id = uuid4()
        conversation_id = uuid4()
        turn_id = uuid4()
        action = Action(
            id=action_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            action_type=ActionType.CALENDAR_CREATE,
            status=ActionStatus.PENDING,
            parameters={"title": "Test Event"},
        )
        assert action.id == action_id
        assert action.conversation_id == conversation_id
        assert action.action_type == ActionType.CALENDAR_CREATE
        assert action.status == ActionStatus.PENDING


class TestUser:
    """Test User schema."""

    def test_user_creation(self):
        """Test creating a user."""
        user_id = uuid4()
        user = User(
            id=user_id,
            name="John Doe",
            email="john@example.com",
            phone="+1234567890",
        )
        assert user.id == user_id
        assert user.name == "John Doe"
        assert user.email == "john@example.com"


class TestRoomInfo:
    """Test RoomInfo schema."""

    def test_room_info_creation(self):
        """Test creating room info."""
        room = RoomInfo(
            room_id="room-123",
            name="Conference Room A",
            participant_count=2,
            is_active=True,
        )
        assert room.room_id == "room-123"
        assert room.name == "Conference Room A"
        assert room.participant_count == 2
        assert room.is_active is True