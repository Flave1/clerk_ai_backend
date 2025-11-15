"""
Distributed state manager using Redis.
Replaces in-memory dictionaries for horizontal scalability.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timezone, timedelta
import asyncio

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class StateManager:
    """Distributed state manager using Redis with in-memory fallback."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.use_redis = False
        self._fallback_store: Dict[str, Any] = {}  # In-memory fallback
        self._pubsub: Optional[Any] = None
        self._subscribers: Dict[str, List[callable]] = {}
        
    async def initialize(self) -> bool:
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using in-memory fallback")
            return False
            
        try:
            redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379/0')
            self.redis_client = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=False,  # Keep binary for audio data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            self.use_redis = True
            logger.info("✅ Redis state manager initialized")
            
            # Initialize pub/sub
            self._pubsub = self.redis_client.pubsub()
            await self._start_pubsub_listener()
            
            return True
        except Exception as e:
            logger.warning(f"Redis initialization failed, using in-memory fallback: {e}")
            self.use_redis = False
            return False
    
    async def _start_pubsub_listener(self):
        """Start listening to pub/sub messages."""
        if not self.use_redis or not self._pubsub:
            return
            
        async def listener():
            try:
                async for message in self._pubsub.listen():
                    if message['type'] == 'message':
                        channel = message['channel'].decode('utf-8')
                        data = json.loads(message['data'].decode('utf-8'))
                        await self._handle_pubsub_message(channel, data)
            except Exception as e:
                logger.error(f"Pub/sub listener error: {e}")
                await asyncio.sleep(1)
                if self.use_redis:
                    await self._start_pubsub_listener()
        
        asyncio.create_task(listener())
    
    async def _handle_pubsub_message(self, channel: str, data: Dict[str, Any]):
        """Handle incoming pub/sub message."""
        if channel.startswith('conversation:'):
            conversation_id = channel.split(':')[1]
            if conversation_id in self._subscribers:
                for callback in self._subscribers[conversation_id]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
                    except Exception as e:
                        logger.error(f"Error in pub/sub callback: {e}")
    
    async def cleanup(self):
        """Cleanup Redis connection."""
        if self._pubsub:
            try:
                await self._pubsub.unsubscribe()
                await self._pubsub.close()
            except Exception:
                pass
        
        if self.redis_client:
            try:
                await self.redis_client.aclose()
            except Exception:
                pass
    
    # Conversation Management
    async def set_conversation(self, conversation_id: str, data: Dict[str, Any], ttl: int = 3600):
        """Store conversation data."""
        key = f"conversation:{conversation_id}"
        serialized = json.dumps(data).encode('utf-8')
        
        if self.use_redis:
            try:
                await self.redis_client.setex(key, ttl, serialized)
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                self._fallback_store[key] = (data, datetime.now(timezone.utc) + timedelta(seconds=ttl))
        else:
            self._fallback_store[key] = (data, datetime.now(timezone.utc) + timedelta(seconds=ttl))
    
    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation data."""
        key = f"conversation:{conversation_id}"
        
        if self.use_redis:
            try:
                data = await self.redis_client.get(key)
                if data:
                    return json.loads(data.decode('utf-8'))
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # Fallback
        if key in self._fallback_store:
            data, expiry = self._fallback_store[key]
            if datetime.now(timezone.utc) < expiry:
                return data
            else:
                del self._fallback_store[key]
        
        return None
    
    async def delete_conversation(self, conversation_id: str):
        """Delete conversation data."""
        key = f"conversation:{conversation_id}"
        
        if self.use_redis:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        if key in self._fallback_store:
            del self._fallback_store[key]
    
    async def list_conversations(self, pattern: str = "conversation:*") -> List[str]:
        """List all conversation IDs."""
        if self.use_redis:
            try:
                keys = await self.redis_client.keys(pattern)
                return [k.decode('utf-8').split(':')[1] for k in keys if b':' in k]
            except Exception as e:
                logger.error(f"Redis keys error: {e}")
        
        # Fallback
        return [k.split(':')[1] for k in self._fallback_store.keys() if k.startswith('conversation:')]
    
    # Bot Session Management
    async def set_bot_session(self, session_id: str, websocket_id: str, metadata: Dict[str, Any], ttl: int = 7200):
        """Store bot session data."""
        key = f"bot_session:{session_id}"
        data = {
            'websocket_id': websocket_id,
            'metadata': metadata,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        serialized = json.dumps(data).encode('utf-8')
        
        if self.use_redis:
            try:
                await self.redis_client.setex(key, ttl, serialized)
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                self._fallback_store[key] = (data, datetime.now(timezone.utc) + timedelta(seconds=ttl))
        else:
            self._fallback_store[key] = (data, datetime.now(timezone.utc) + timedelta(seconds=ttl))
    
    async def get_bot_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get bot session data."""
        key = f"bot_session:{session_id}"
        
        if self.use_redis:
            try:
                data = await self.redis_client.get(key)
                if data:
                    return json.loads(data.decode('utf-8'))
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # Fallback
        if key in self._fallback_store:
            data, expiry = self._fallback_store[key]
            if datetime.now(timezone.utc) < expiry:
                return data
            else:
                del self._fallback_store[key]
        
        return None
    
    async def delete_bot_session(self, session_id: str):
        """Delete bot session data."""
        key = f"bot_session:{session_id}"
        
        if self.use_redis:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        if key in self._fallback_store:
            del self._fallback_store[key]
    
    # WebSocket Registry (ephemeral - no persistence needed)
    async def register_websocket(self, conversation_id: str, websocket_id: str):
        """Register a WebSocket connection."""
        key = f"ws:{conversation_id}:{websocket_id}"
        
        if self.use_redis:
            try:
                await self.redis_client.setex(key, 3600, b"1")  # 1 hour TTL
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        else:
            self._fallback_store[key] = (True, datetime.now(timezone.utc) + timedelta(hours=1))
    
    async def unregister_websocket(self, conversation_id: str, websocket_id: str):
        """Unregister a WebSocket connection."""
        key = f"ws:{conversation_id}:{websocket_id}"
        
        if self.use_redis:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        if key in self._fallback_store:
            del self._fallback_store[key]
    
    async def get_websockets_for_conversation(self, conversation_id: str) -> List[str]:
        """Get all WebSocket IDs for a conversation."""
        pattern = f"ws:{conversation_id}:*"
        
        if self.use_redis:
            try:
                keys = await self.redis_client.keys(pattern)
                return [k.decode('utf-8').split(':')[2] for k in keys]
            except Exception as e:
                logger.error(f"Redis keys error: {e}")
        
        # Fallback
        return [k.split(':')[2] for k in self._fallback_store.keys() if k.startswith(f"ws:{conversation_id}:")]
    
    # Pub/Sub for broadcasts
    async def subscribe_to_conversation(self, conversation_id: str, callback: callable):
        """Subscribe to conversation events."""
        if conversation_id not in self._subscribers:
            self._subscribers[conversation_id] = []
        
        self._subscribers[conversation_id].append(callback)
        
        if self.use_redis and self._pubsub:
            try:
                channel = f"conversation:{conversation_id}"
                await self._pubsub.subscribe(channel)
            except Exception as e:
                logger.error(f"Redis subscribe error: {e}")
    
    async def unsubscribe_from_conversation(self, conversation_id: str, callback: callable):
        """Unsubscribe from conversation events."""
        if conversation_id in self._subscribers:
            if callback in self._subscribers[conversation_id]:
                self._subscribers[conversation_id].remove(callback)
    
    async def broadcast_to_conversation(self, conversation_id: str, message: Dict[str, Any]):
        """Broadcast message to all subscribers of a conversation."""
        if self.use_redis:
            try:
                channel = f"conversation:{conversation_id}"
                await self.redis_client.publish(channel, json.dumps(message).encode('utf-8'))
            except Exception as e:
                logger.error(f"Redis publish error: {e}")
        
        # Also notify local subscribers
        if conversation_id in self._subscribers:
            for callback in self._subscribers[conversation_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    logger.error(f"Error in broadcast callback: {e}")
    
    # Participant Management
    async def add_participant(self, conversation_id: str, participant_id: str):
        """Add participant to conversation."""
        key = f"participants:{conversation_id}"
        
        if self.use_redis:
            try:
                await self.redis_client.sadd(key, participant_id)
                await self.redis_client.expire(key, 3600)
            except Exception as e:
                logger.error(f"Redis sadd error: {e}")
        else:
            if key not in self._fallback_store:
                self._fallback_store[key] = (set(), datetime.now(timezone.utc) + timedelta(hours=1))
            self._fallback_store[key][0].add(participant_id)
    
    async def remove_participant(self, conversation_id: str, participant_id: str):
        """Remove participant from conversation."""
        key = f"participants:{conversation_id}"
        
        if self.use_redis:
            try:
                await self.redis_client.srem(key, participant_id)
            except Exception as e:
                logger.error(f"Redis srem error: {e}")
        else:
            if key in self._fallback_store:
                self._fallback_store[key][0].discard(participant_id)
    
    async def get_participants(self, conversation_id: str) -> Set[str]:
        """Get all participants in a conversation."""
        key = f"participants:{conversation_id}"
        
        if self.use_redis:
            try:
                members = await self.redis_client.smembers(key)
                return {m.decode('utf-8') if isinstance(m, bytes) else m for m in members}
            except Exception as e:
                logger.error(f"Redis smembers error: {e}")
        
        # Fallback
        if key in self._fallback_store:
            return self._fallback_store[key][0]
        
        return set()


# Global state manager instance
_state_manager: Optional[StateManager] = None


async def get_state_manager() -> StateManager:
    """Get or create state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
        await _state_manager.initialize()
    return _state_manager

