"""
WebSocket connection management for real-time dashboard updates.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Set

from fastapi import WebSocket, WebSocketDisconnect

from shared.schemas import WebSocketMessage

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_subscriptions: Dict[
            str, Set[str]
        ] = {}  # connection_id -> set of topics
        self.topic_subscribers: Dict[
            str, Set[str]
        ] = {}  # topic -> set of connection_ids
        self.initialized = False

    async def initialize(self):
        """Initialize the connection manager."""
        self.initialized = True
        logger.info("WebSocket connection manager initialized")

    async def connect(self, websocket: WebSocket, connection_id: str) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_subscriptions[connection_id] = set()

        logger.info(f"WebSocket connection established: {connection_id}")

    def disconnect(self, connection_id: str) -> None:
        """Remove a WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]

        # Remove from all topic subscriptions
        if connection_id in self.connection_subscriptions:
            for topic in self.connection_subscriptions[connection_id]:
                if topic in self.topic_subscribers:
                    self.topic_subscribers[topic].discard(connection_id)
            del self.connection_subscriptions[connection_id]

        logger.info(f"WebSocket connection closed: {connection_id}")

    async def subscribe(self, connection_id: str, topic: str) -> bool:
        """Subscribe a connection to a topic."""
        if connection_id not in self.active_connections:
            return False

        # Add to connection's subscriptions
        if connection_id not in self.connection_subscriptions:
            self.connection_subscriptions[connection_id] = set()
        self.connection_subscriptions[connection_id].add(topic)

        # Add to topic's subscribers
        if topic not in self.topic_subscribers:
            self.topic_subscribers[topic] = set()
        self.topic_subscribers[topic].add(connection_id)

        # Send confirmation
        await self.send_personal_message(
            {
                "type": "subscription_confirmed",
                "topic": topic,
                "timestamp": datetime.utcnow().isoformat(),
            },
            connection_id,
        )

        logger.info(f"Connection {connection_id} subscribed to topic {topic}")
        return True

    async def unsubscribe(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe a connection from a topic."""
        if connection_id not in self.active_connections:
            return False

        # Remove from connection's subscriptions
        if connection_id in self.connection_subscriptions:
            self.connection_subscriptions[connection_id].discard(topic)

        # Remove from topic's subscribers
        if topic in self.topic_subscribers:
            self.topic_subscribers[topic].discard(connection_id)

        # Send confirmation
        await self.send_personal_message(
            {
                "type": "unsubscription_confirmed",
                "topic": topic,
                "timestamp": datetime.utcnow().isoformat(),
            },
            connection_id,
        )

        logger.info(f"Connection {connection_id} unsubscribed from topic {topic}")
        return True

    async def send_personal_message(
        self, message: Dict[str, Any], connection_id: str
    ) -> bool:
        """Send a message to a specific connection."""
        if connection_id not in self.active_connections:
            return False

        try:
            websocket = self.active_connections[connection_id]
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            # Remove failed connection
            self.disconnect(connection_id)
            return False

    async def broadcast_to_topic(self, message: Dict[str, Any], topic: str) -> int:
        """Broadcast a message to all subscribers of a topic."""
        if topic not in self.topic_subscribers:
            return 0

        sent_count = 0
        failed_connections = []

        for connection_id in self.topic_subscribers[topic].copy():
            if await self.send_personal_message(message, connection_id):
                sent_count += 1
            else:
                failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            self.disconnect(connection_id)

        logger.info(f"Broadcasted message to {sent_count} connections on topic {topic}")
        return sent_count

    async def broadcast_to_all(self, message: Dict[str, Any]) -> int:
        """Broadcast a message to all active connections."""
        sent_count = 0
        failed_connections = []

        for connection_id in self.active_connections.copy():
            if await self.send_personal_message(message, connection_id):
                sent_count += 1
            else:
                failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            self.disconnect(connection_id)

        logger.info(f"Broadcasted message to {sent_count} connections")
        return sent_count

    async def publish_conversation_update(
        self, conversation_id: str, update_type: str, data: Dict[str, Any]
    ):
        """Publish conversation update to subscribers."""
        message = {
            "type": "conversation_update",
            "conversation_id": conversation_id,
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Broadcast to conversation-specific topic
        topic = f"conversation:{conversation_id}"
        await self.broadcast_to_topic(message, topic)

        # Broadcast to general conversations topic
        await self.broadcast_to_topic(message, "conversations")

    async def publish_action_update(
        self,
        action_id: str,
        conversation_id: str,
        update_type: str,
        data: Dict[str, Any],
    ):
        """Publish action update to subscribers."""
        message = {
            "type": "action_update",
            "action_id": action_id,
            "conversation_id": conversation_id,
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Broadcast to conversation-specific topic
        topic = f"conversation:{conversation_id}"
        await self.broadcast_to_topic(message, topic)

        # Broadcast to general actions topic
        await self.broadcast_to_topic(message, "actions")

    async def publish_room_update(
        self, room_id: str, update_type: str, data: Dict[str, Any]
    ):
        """Publish room update to subscribers."""
        message = {
            "type": "room_update",
            "room_id": room_id,
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Broadcast to room-specific topic
        topic = f"room:{room_id}"
        await self.broadcast_to_topic(message, topic)

        # Broadcast to general rooms topic
        await self.broadcast_to_topic(message, "rooms")

    async def publish_system_message(self, message_type: str, data: Dict[str, Any]):
        """Publish system-wide message."""
        message = {
            "type": "system_message",
            "message_type": message_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.broadcast_to_topic(message, "system")

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)

    def get_subscription_count(self, topic: str) -> int:
        """Get the number of subscribers for a topic."""
        return len(self.topic_subscribers.get(topic, set()))

    def get_topics(self) -> Set[str]:
        """Get all active topics."""
        return set(self.topic_subscribers.keys())

    async def handle_client_message(
        self, connection_id: str, message_data: Dict[str, Any]
    ):
        """Handle incoming message from client."""
        try:
            message_type = message_data.get("type")

            if message_type == "subscribe":
                topic = message_data.get("topic")
                if topic:
                    await self.subscribe(connection_id, topic)

            elif message_type == "unsubscribe":
                topic = message_data.get("topic")
                if topic:
                    await self.unsubscribe(connection_id, topic)

            elif message_type == "ping":
                await self.send_personal_message(
                    {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                    connection_id,
                )

            else:
                logger.warning(
                    f"Unknown message type from {connection_id}: {message_type}"
                )

        except Exception as e:
            logger.error(f"Error handling client message from {connection_id}: {e}")

    async def cleanup_stale_connections(self):
        """Clean up stale connections."""
        stale_connections = []

        for connection_id, websocket in self.active_connections.items():
            try:
                # Try to send a ping
                await websocket.send_text(json.dumps({"type": "ping"}))
            except:
                stale_connections.append(connection_id)

        # Remove stale connections
        for connection_id in stale_connections:
            self.disconnect(connection_id)

        if stale_connections:
            logger.info(f"Cleaned up {len(stale_connections)} stale connections")
