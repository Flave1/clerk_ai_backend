"""
Pydantic models for the Meeting Agent service.

This module imports models from shared schemas and adds any meeting-specific models.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# Import shared models
from shared.schemas import (
    Meeting,
    MeetingPlatform,
    MeetingStatus,
    MeetingSummary,
    ActionItem,
    MeetingParticipant,
    TranscriptionChunk,
    MeetingNotification,
    CalendarEvent,
)


class MeetingJoinRequest(BaseModel):
    """Request to join a meeting."""
    
    meeting_id: UUID
    platform: MeetingPlatform
    meeting_url: str
    meeting_id_external: str
    join_time: datetime = Field(default_factory=datetime.utcnow)


class MeetingJoinResponse(BaseModel):
    """Response from joining a meeting."""
    
    success: bool
    meeting_id: UUID
    error_message: Optional[str] = None
    join_time: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None  # Additional data like Fireflies transcript ID


class MeetingConfig(BaseModel):
    """Configuration for meeting agent."""
    
    # Join settings
    auto_join_enabled: bool = True
    join_buffer_minutes: int = 5  # Join 5 minutes before start
    max_join_attempts: int = 3
    
    # Transcription settings
    transcription_enabled: bool = True
    chunk_size_seconds: int = 30
    language: str = "en"
    
    # Summarization settings
    summarization_enabled: bool = True
    summary_frequency_minutes: int = 10  # Generate summary every 10 minutes
    final_summary_enabled: bool = True
    
    # Notification settings
    email_notifications_enabled: bool = True
    slack_notifications_enabled: bool = True
    notification_channels: List[str] = Field(default_factory=list)
    
    # Voice participation (optional)
    voice_participation_enabled: bool = False
    response_triggers: List[str] = Field(default_factory=list)
    
    # Storage settings
    store_audio: bool = False
    store_transcription: bool = True
    retention_days: int = 90


# Re-export shared models for convenience
__all__ = [
    "Meeting",
    "MeetingPlatform", 
    "MeetingStatus",
    "MeetingSummary",
    "ActionItem",
    "MeetingParticipant",
    "TranscriptionChunk",
    "MeetingNotification",
    "CalendarEvent",
    "MeetingJoinRequest",
    "MeetingJoinResponse",
    "MeetingConfig",
]