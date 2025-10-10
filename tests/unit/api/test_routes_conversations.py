"""Unit tests for conversation API routes."""

import pytest
from datetime import datetime, timezone


class TestConversationRoutes:
    """Test conversation API endpoints."""

    def test_get_conversations(self, api_client):
        """Test getting all conversations."""
        response = api_client.get("/api/v1/conversations")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_create_conversation(self, api_client):
        """Test creating a conversation."""
        conversation_data = {
            "room_id": "room-123",
            "user_id": "user-456",
            "status": "active",
        }
        response = api_client.post("/api/v1/conversations", json=conversation_data)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["room_id"] == "room-123"
        assert data["user_id"] == "user-456"

    def test_get_conversation_by_id(self, api_client):
        """Test getting a conversation by ID."""
        # First create a conversation
        conversation_data = {
            "room_id": "room-123",
            "user_id": "user-456",
            "status": "active",
        }
        create_response = api_client.post("/api/v1/conversations", json=conversation_data)
        conversation_id = create_response.json()["id"]

        # Then get it
        response = api_client.get(f"/api/v1/conversations/{conversation_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == conversation_id

    def test_update_conversation(self, api_client):
        """Test updating a conversation."""
        # First create a conversation
        conversation_data = {
            "room_id": "room-123",
            "user_id": "user-456",
            "status": "active",
        }
        create_response = api_client.post("/api/v1/conversations", json=conversation_data)
        conversation_id = create_response.json()["id"]

        # Then update it
        update_data = {"status": "completed", "summary": "Test summary"}
        response = api_client.put(f"/api/v1/conversations/{conversation_id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["summary"] == "Test summary"