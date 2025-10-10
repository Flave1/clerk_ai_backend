"""
Shared Pydantic schemas for the AI Receptionist system.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ConversationStatus(str, Enum):
    """Conversation status enumeration."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class TurnType(str, Enum):
    """Turn type enumeration."""

    USER_SPEECH = "user_speech"
    AI_RESPONSE = "ai_response"
    SYSTEM_MESSAGE = "system_message"
    ERROR = "error"


class ActionStatus(str, Enum):
    """Action status enumeration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActionType(str, Enum):
    """Action type enumeration."""

    CALENDAR_CREATE = "calendar_create"
    CALENDAR_UPDATE = "calendar_update"
    CALENDAR_DELETE = "calendar_delete"
    EMAIL_SEND = "email_send"
    SLACK_MESSAGE = "slack_message"
    CRM_UPDATE = "crm_update"
    KNOWLEDGE_SEARCH = "knowledge_search"
    MEETING_JOIN = "meeting_join"
    MEETING_LEAVE = "meeting_leave"
    MEETING_TRANSCRIBE = "meeting_transcribe"
    MEETING_SUMMARIZE = "meeting_summarize"


class MeetingPlatform(str, Enum):
    """Meeting platform enumeration."""
    
    GOOGLE_MEET = "google_meet"
    ZOOM = "zoom"
    MICROSOFT_TEAMS = "microsoft_teams"


class MeetingStatus(str, Enum):
    """Meeting status enumeration."""
    
    SCHEDULED = "scheduled"
    JOINING = "joining"
    ACTIVE = "active"
    ENDED = "ended"
    FAILED = "failed"
    CANCELLED = "cancelled"


class User(BaseModel):
    """User model."""

    id: UUID = Field(default_factory=uuid4)
    email: str
    name: str
    phone: Optional[str] = None
    timezone: str = "UTC"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Conversation(BaseModel):
    """Conversation model."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    room_id: str
    status: ConversationStatus = ConversationStatus.ACTIVE
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    """Turn model for conversation turns."""

    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    turn_number: int
    turn_type: TurnType
    content: str
    audio_url: Optional[str] = None
    confidence_score: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """Action model for external integrations."""

    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    turn_id: Optional[UUID] = None
    action_type: ActionType
    status: ActionStatus = ActionStatus.PENDING
    parameters: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class RoomInfo(BaseModel):
    """LiveKit room information."""

    room_id: str
    name: str
    participant_count: int
    is_active: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WebSocketMessage(BaseModel):
    """WebSocket message schema."""

    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationUpdate(BaseModel):
    """Conversation update message."""

    conversation_id: UUID
    status: Optional[ConversationStatus] = None
    turn: Optional[Turn] = None
    action: Optional[Action] = None


class STTRequest(BaseModel):
    """Speech-to-text request."""

    audio_data: bytes
    conversation_id: UUID
    turn_number: int
    language: str = "en"


class STTResponse(BaseModel):
    """Speech-to-text response."""

    text: str
    confidence: float
    language: str
    is_final: bool


class TTSRequest(BaseModel):
    """Text-to-speech request."""

    text: str
    voice_id: str = "default"
    conversation_id: UUID
    turn_number: int


class TTSResponse(BaseModel):
    """Text-to-speech response."""

    audio_data: bytes
    duration: float
    voice_id: str


class LLMRequest(BaseModel):
    """LLM request."""

    conversation_id: UUID
    messages: List[Dict[str, str]]
    tools: Optional[List[Dict[str, Any]]] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None


class LLMResponse(BaseModel):
    """LLM response."""

    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None
    finish_reason: str


class ToolExecution(BaseModel):
    """Tool execution request."""

    tool_name: str
    parameters: Dict[str, Any]
    conversation_id: UUID
    turn_id: Optional[UUID] = None


class ToolResult(BaseModel):
    """Tool execution result."""

    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float


class MeetingParticipant(BaseModel):
    """Meeting participant model."""
    
    email: str
    name: Optional[str] = None
    is_organizer: bool = False
    response_status: str = "accepted"  # accepted, declined, tentative, needsAction


class ActionItem(BaseModel):
    """Action item extracted from meeting."""
    
    id: UUID = Field(default_factory=uuid4)
    description: str
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: str = "medium"  # low, medium, high
    status: str = "pending"  # pending, in_progress, completed, cancelled


class MeetingSummary(BaseModel):
    """Meeting summary model."""
    
    id: UUID = Field(default_factory=uuid4)
    meeting_id: UUID
    topics_discussed: List[str] = Field(default_factory=list)
    key_decisions: List[str] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    summary_text: str
    sentiment: Optional[str] = None  # positive, negative, neutral
    duration_minutes: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Meeting(BaseModel):
    """Meeting model for DynamoDB storage."""
    
    id: UUID = Field(default_factory=uuid4)
    platform: MeetingPlatform
    meeting_url: str
    meeting_id_external: str  # External platform meeting ID
    title: str
    description: Optional[str] = None
    start_time: datetime
    end_time: datetime
    organizer_email: str
    participants: List[MeetingParticipant] = Field(default_factory=list)
    status: MeetingStatus = MeetingStatus.SCHEDULED
    ai_email: str  # The AI's email address for calendar invites
    
    # Meeting content
    transcription_chunks: List[str] = Field(default_factory=list)
    full_transcription: Optional[str] = None
    summary: Optional[MeetingSummary] = None
    
    # Metadata
    calendar_event_id: Optional[str] = None
    join_attempts: int = 0
    last_join_attempt: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    joined_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    # Audio/Video settings
    audio_enabled: bool = True
    video_enabled: bool = False
    recording_enabled: bool = False


class TranscriptionChunk(BaseModel):
    """Real-time transcription chunk."""
    
    meeting_id: UUID
    chunk_id: UUID = Field(default_factory=uuid4)
    text: str
    speaker: Optional[str] = None
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_final: bool = False


class MeetingNotification(BaseModel):
    """Meeting notification model."""
    
    meeting_id: UUID
    notification_type: str  # summary, action_items, reminder
    recipients: List[str]  # Email addresses
    subject: str
    content: str
    sent_at: Optional[datetime] = None
    delivery_status: str = "pending"  # pending, sent, failed


class CalendarEvent(BaseModel):
    """Calendar event model for detecting meetings."""
    
    event_id: str
    title: str
    description: Optional[str] = None
    start_time: datetime
    end_time: datetime
    organizer_email: str
    attendees: List[MeetingParticipant] = Field(default_factory=list)
    meeting_url: Optional[str] = None
    platform: Optional[MeetingPlatform] = None
    calendar_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
