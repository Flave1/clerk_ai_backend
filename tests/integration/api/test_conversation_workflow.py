"""Integration tests for conversation workflow."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4


class TestConversationWorkflow:
    """Test complete conversation workflow through API."""

    def test_create_conversation_and_add_turns(self, api_client, dynamodb_tables):
        """Test creating a conversation and adding turns."""
        # Create conversation
        user_id = str(uuid4())
        conversation_data = {
            "room_id": "room-123",
            "user_id": user_id,
            "status": "active",
        }
        conv_response = api_client.post("/api/v1/conversations", json=conversation_data)
        assert conv_response.status_code == 200
        conversation_id = conv_response.json()["id"]

        # Add a turn
        turn_data = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "speaker": "user",
            "text": "Hello, I need help with my calendar",
        }
        turn_response = api_client.post(f"/api/v1/conversations/{conversation_id}/turns", json=turn_data)
        assert turn_response.status_code == 200
        turn_id = turn_response.json()["id"]

        # Create an action
        action_data = {
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "action_type": "calendar_create",
            "status": "pending",
            "parameters": {"title": "Test Meeting", "start_time": "2024-01-01T11:00:00Z"},
        }
        action_response = api_client.post("/api/v1/actions", json=action_data)
        assert action_response.status_code == 200
        action_id = action_response.json()["id"]

        # Verify all entities were created
        conv_check = api_client.get(f"/api/v1/conversations/{conversation_id}")
        assert conv_check.status_code == 200

        action_check = api_client.get(f"/api/v1/actions/{action_id}")
        assert action_check.status_code == 200
        assert action_check.json()["action_type"] == "calendar_create"

    def test_conversation_status_updates(self, api_client, dynamodb_tables):
        """Test updating conversation status."""
        # Create conversation
        user_id = str(uuid4())
        conversation_data = {
            "room_id": "room-123",
            "user_id": user_id,
            "status": "active",
        }
        conv_response = api_client.post("/api/v1/conversations", json=conversation_data)
        conversation_id = conv_response.json()["id"]

        # Update status to completed
        update_data = {"status": "completed", "summary": "Meeting scheduled successfully"}
        update_response = api_client.put(f"/api/v1/conversations/{conversation_id}", json=update_data)
        assert update_response.status_code == 200
        
        # Verify update
        check_response = api_client.get(f"/api/v1/conversations/{conversation_id}")
        assert check_response.json()["status"] == "completed"
        assert check_response.json()["summary"] == "Meeting scheduled successfully"