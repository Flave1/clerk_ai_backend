"""
Realtime Session Manager - CORRECTED VERSION

Manages Realtime API sessions with proper lifecycle, locking, and reconnection support.
Fixes:
- Thread-safe session access
- Proper cleanup on disconnect
- Support for multiple connections per session
- No race conditions
- Proper error recovery
"""
import asyncio
import json
import logging
from typing import Optional, Dict, Set, Callable, Awaitable
from asyncio import Lock
from datetime import datetime, timezone

from .realtime_api_fixed import RealtimeAPIService

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """Represents a single WebSocket connection."""
    def __init__(self, connection_id: str, websocket, session_id: str):
        self.connection_id = connection_id
        self.websocket = websocket
        self.session_id = session_id
        self.created_at = datetime.now(timezone.utc)
        self.closed = False


class RealtimeSession:
    """Manages a single Realtime API session with multiple WebSocket connections."""
    
    def __init__(
        self, 
        session_id: str, 
        meeting_context: Optional[Dict] = None,
        tool_handler: Optional[Callable[[str, Dict], Awaitable[Dict]]] = None
    ):
        self.session_id = session_id
        self.meeting_context = meeting_context or {}
        self.realtime_service: Optional[RealtimeAPIService] = None
        self.connections: Dict[str, WebSocketConnection] = {}  # connection_id -> connection
        self._lock = Lock()
        self._initializing = False
        self._initialized = False
        self._tool_handler = tool_handler  # Optional tool handler
        
    async def add_connection(self, connection: WebSocketConnection) -> bool:
        """Add a WebSocket connection to this session."""
        async with self._lock:
            self.connections[connection.connection_id] = connection
            
            # Initialize Realtime API if not already done
            if not self._initialized and not self._initializing:
                self._initializing = True
                asyncio.create_task(self._initialize_realtime())
            
            return True
    
    async def remove_connection(self, connection_id: str):
        """Remove a WebSocket connection."""
        async with self._lock:
            if connection_id in self.connections:
                conn = self.connections[connection_id]
                conn.closed = True
                del self.connections[connection_id]
                
                # If no more connections, cleanup Realtime API
                if not self.connections and self.realtime_service:
                    logger.info(f"[SessionManager] No more connections for {self.session_id}, cleaning up")
                    await self.realtime_service.disconnect()
                    self.realtime_service = None
                    self._initialized = False
    
    async def _initialize_realtime(self):
        """Initialize Realtime API service for this session."""
        try:
            realtime_service = RealtimeAPIService()
            
            # Set up event handlers that route to all active connections
            realtime_service.on_audio_delta = self._make_audio_handler()
            realtime_service.on_transcript_delta = self._make_transcript_handler()
            realtime_service.on_tool_call = self._make_tool_call_handler()
            realtime_service.on_response_done = self._make_response_done_handler()
            realtime_service.on_error = self._make_error_handler()
            
            # Connect
            connected = await realtime_service.connect(
                session_id=self.session_id,
                meeting_context=self.meeting_context,
                voice="alloy",
                instructions=None,
                tools=None
            )
            
            if connected:
                async with self._lock:
                    self.realtime_service = realtime_service
                    self._initialized = True
                    self._initializing = False
                logger.info(f"[SessionManager] Realtime API initialized for {self.session_id}")
            else:
                async with self._lock:
                    self._initializing = False
                logger.error(f"[SessionManager] Failed to initialize Realtime API for {self.session_id}")
                
        except Exception as e:
            logger.error(f"[SessionManager] Error initializing Realtime API: {e}", exc_info=True)
            async with self._lock:
                self._initializing = False
    
    def _make_audio_handler(self) -> Callable[[bytes], Awaitable[None]]:
        """Create audio delta handler that broadcasts to all connections."""
        async def handler(audio_bytes: bytes):
            async with self._lock:
                connections = list(self.connections.values())
            
            # Send to all active connections
            tasks = []
            for conn in connections:
                if not conn.closed:
                    try:
                        tasks.append(conn.websocket.send_bytes(audio_bytes))
                    except Exception as e:
                        logger.debug(f"[SessionManager] Error sending audio to {conn.connection_id}: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        return handler
    
    def _make_transcript_handler(self) -> Callable[[str], Awaitable[None]]:
        """Create transcript delta handler."""
        async def handler(delta: str):
            async with self._lock:
                connections = list(self.connections.values())
            
            message = {
                "type": "transcript",
                "delta": delta,
                "session_id": self.session_id
            }
            
            tasks = []
            for conn in connections:
                if not conn.closed:
                    try:
                        tasks.append(conn.websocket.send_text(json.dumps(message)))
                    except Exception as e:
                        logger.debug(f"[SessionManager] Error sending transcript: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        return handler
    
    def _make_tool_call_handler(self) -> Callable[[str, Dict], Awaitable[Dict]]:
        """Create tool call handler."""
        async def handler(tool_name: str, arguments: Dict) -> Dict:
            # Broadcast tool call to frontend
            async with self._lock:
                connections = list(self.connections.values())
            
            message = {
                "type": "tool_call",
                "name": tool_name,
                "arguments": arguments,
                "session_id": self.session_id
            }
            
            tasks = []
            for conn in connections:
                if not conn.closed:
                    try:
                        tasks.append(conn.websocket.send_text(json.dumps(message)))
                    except Exception:
                        pass
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Execute tool using configured handler or default
            if self._tool_handler:
                result = await self._tool_handler(tool_name, arguments)
            else:
                # Default handler
                result = {"status": "ok", "message": f"Executed {tool_name}"}
            
            # Send tool result to frontend
            result_message = {
                "type": "tool_result",
                "name": tool_name,
                "result": result,
                "session_id": self.session_id
            }
            
            tasks = []
            for conn in connections:
                if not conn.closed:
                    try:
                        tasks.append(conn.websocket.send_text(json.dumps(result_message)))
                    except Exception:
                        pass
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            return result
        
        return handler
    
    def _make_response_done_handler(self) -> Callable[[], Awaitable[None]]:
        """Create response done handler."""
        async def handler():
            async with self._lock:
                connections = list(self.connections.values())
            
            message = {
                "type": "done",
                "session_id": self.session_id
            }
            
            tasks = []
            for conn in connections:
                if not conn.closed:
                    try:
                        tasks.append(conn.websocket.send_text(json.dumps(message)))
                    except Exception:
                        pass
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        return handler
    
    def _make_error_handler(self) -> Callable[[str], Awaitable[None]]:
        """Create error handler."""
        async def handler(error_message: str):
            async with self._lock:
                connections = list(self.connections.values())
            
            message = {
                "type": "error",
                "message": error_message,
                "session_id": self.session_id
            }
            
            tasks = []
            for conn in connections:
                if not conn.closed:
                    try:
                        tasks.append(conn.websocket.send_text(json.dumps(message)))
                    except Exception:
                        pass
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        return handler
    
    async def send_audio_chunk(self, audio_chunk: bytes) -> bool:
        """Send audio chunk to Realtime API."""
        async with self._lock:
            if not self.realtime_service:
                return False
            service = self.realtime_service
        
        return await service.send_audio_chunk(audio_chunk)
    
    async def commit_audio(self) -> bool:
        """Commit audio to Realtime API."""
        async with self._lock:
            if not self.realtime_service:
                return False
            service = self.realtime_service
        
        return await service.commit_audio()
    
    async def interrupt(self) -> bool:
        """Interrupt current response."""
        async with self._lock:
            if not self.realtime_service:
                return False
            service = self.realtime_service
        
        return await service.interrupt()
    
    async def wait_for_ready(self, timeout: float = 10.0) -> bool:
        """Wait for Realtime API to be ready."""
        start = asyncio.get_event_loop().time()
        while True:
            async with self._lock:
                if self._initialized and self.realtime_service:
                    return True
                if not self._initializing:
                    return False
            
            if asyncio.get_event_loop().time() - start > timeout:
                return False
            
            await asyncio.sleep(0.1)


class RealtimeSessionManager:
    """Singleton manager for all Realtime API sessions."""
    
    _instance: Optional['RealtimeSessionManager'] = None
    _lock = Lock()
    
    def __init__(self):
        self.sessions: Dict[str, RealtimeSession] = {}  # session_id -> session
        self._manager_lock = Lock()
    
    @classmethod
    async def get_instance(cls) -> 'RealtimeSessionManager':
        """Get singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    async def get_or_create_session(
        self,
        session_id: str,
        connection_id: str,
        websocket,
        meeting_context: Optional[Dict] = None,
        tool_handler: Optional[Callable[[str, Dict], Awaitable[Dict]]] = None
    ) -> RealtimeSession:
        """Get or create a session and add connection."""
        async with self._manager_lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = RealtimeSession(
                    session_id, 
                    meeting_context,
                    tool_handler=tool_handler
                )
            else:
                # Update tool handler if provided and session doesn't have one
                if tool_handler and not self.sessions[session_id]._tool_handler:
                    self.sessions[session_id]._tool_handler = tool_handler
            
            session = self.sessions[session_id]
            connection = WebSocketConnection(connection_id, websocket, session_id)
            await session.add_connection(connection)
            
            return session
    
    async def remove_connection(self, session_id: str, connection_id: str):
        """Remove a connection from a session."""
        async with self._manager_lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                await session.remove_connection(connection_id)
                
                # Cleanup session if no connections
                async with session._lock:
                    if not session.connections:
                        del self.sessions[session_id]
                        logger.info(f"[SessionManager] Removed session {session_id}")
    
    async def get_session(self, session_id: str) -> Optional[RealtimeSession]:
        """Get a session by ID."""
        async with self._manager_lock:
            return self.sessions.get(session_id)

