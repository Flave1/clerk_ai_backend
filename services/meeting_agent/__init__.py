"""
Meeting Agent Service for AI Receptionist.

This module provides automated meeting participation capabilities for Google Meet,
Zoom, and Microsoft Teams meetings. It includes real-time transcription,
summarization, and notification features.
"""

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

# Import meeting-specific models
from .models import (
    MeetingJoinRequest,
    MeetingJoinResponse,
    MeetingConfig,
)

# Import services
from .scheduler import MeetingScheduler
from .transcription_service import TranscriptionService
from .summarization_service import SummarizationService
from .notifier import NotificationService

__all__ = [
    # Shared models
    "Meeting",
    "MeetingPlatform", 
    "MeetingStatus",
    "MeetingSummary",
    "ActionItem",
    "MeetingParticipant",
    "TranscriptionChunk",
    "MeetingNotification",
    "CalendarEvent",
    # Meeting-specific models
    "MeetingJoinRequest",
    "MeetingJoinResponse",
    "MeetingConfig",
    # Services
    "MeetingScheduler",
    "TranscriptionService",
    "SummarizationService",
    "NotificationService",
]
