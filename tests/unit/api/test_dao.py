"""Unit tests for DynamoDB Data Access Object."""

import pytest
from datetime import datetime, timezone
from services.api.dao import DynamoDBDAO


class TestConversationDAO:
    """Test Conversation DAO operations."""

    def test_create_conversation(self, dynamodb_tables):
        """Test creating a conversation."""
        dao = DynamoDBDAO()
        conv_data = {
            "id": "conv-123",
            "room_id": "room-456",
            "user_id": "user-789",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        result = dao.create_conversation(conv_data)
        assert result["id"] == "conv-123"

    def test_get_conversation(self, dynamodb_tables):
        """Test getting a conversation."""
        dao = DynamoDBDAO()
        conv_data = {
            "id": "conv-123",
            "room_id": "room-456",
            "user_id": "user-789",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        dao.create_conversation(conv_data)
        result = dao.get_conversation("conv-123")
        assert result["id"] == "conv-123"

    def test_list_conversations(self, dynamodb_tables):
        """Test listing conversations."""
        dao = DynamoDBDAO()
        conv_data = {
            "id": "conv-123",
            "room_id": "room-456",
            "user_id": "user-789",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        dao.create_conversation(conv_data)
        results = dao.list_conversations()
        assert len(results) == 1
        assert results[0]["id"] == "conv-123"


class TestTurnDAO:
    """Test Turn DAO operations."""

    def test_create_turn(self, dynamodb_tables):
        """Test creating a turn."""
        dao = DynamoDBDAO()
        turn_data = {
            "conversation_id": "conv-123",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "speaker": "user",
            "text": "Hello, world!",
        }
        result = dao.create_turn(turn_data)
        assert result["conversation_id"] == "conv-123"
        assert result["speaker"] == "user"


class TestActionDAO:
    """Test Action DAO operations."""

    def test_create_action(self, dynamodb_tables):
        """Test creating an action."""
        dao = DynamoDBDAO()
        action_data = {
            "id": "action-123",
            "conversation_id": "conv-456",
            "turn_id": "turn-789",
            "action_type": "calendar_create",
            "status": "pending",
            "parameters": {"title": "Test Event"},
        }
        result = dao.create_action(action_data)
        assert result["id"] == "action-123"
        assert result["action_type"] == "calendar_create"