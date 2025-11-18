"""
Meeting service helpers for reusable meeting operations.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from shared.config import get_settings
from shared.schemas import (
    Meeting,
    MeetingParticipant,
    MeetingPlatform,
    MeetingStatus,
    MeetingSummary,
    ActionItem,
)

from .service_meeting_context import ServiceMeetingContext
from .dao import MongoDBDAO
from .meeting_url import create_platform_meeting_url


logger = logging.getLogger(__name__)
settings = get_settings()


class ServiceMeeting:
    """Service layer for meeting creation and management."""

    def __init__(self, dao: MongoDBDAO):
        self.dao = dao
        self.context_service = ServiceMeetingContext(dao)

    async def create_meeting_record(
        self,
        *,
        user_id: UUID,
        meeting_type: str,
        meeting_url: Optional[str] = None,
        audio_record: bool = False,
        video_record: bool = False,
        transcript: bool = False,
        voice_id: Optional[str] = None,
        bot_name: Optional[str] = None,
        context_id: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[MeetingStatus] = None,
    ) -> Meeting:
        """Create a meeting record for the given user."""

        logger.info(
            "Creating meeting record for user %s (type=%s)",
            user_id,
            meeting_type,
        )

        platform_map = {
            "zoom": MeetingPlatform.ZOOM,
            "teams": MeetingPlatform.MICROSOFT_TEAMS,
            "microsoft_teams": MeetingPlatform.MICROSOFT_TEAMS,
            "google_meet": MeetingPlatform.GOOGLE_MEET,
            "aurray": MeetingPlatform.AURRAY,
        }
        platform_enum = platform_map.get(meeting_type.lower(), MeetingPlatform.AURRAY)

        meeting_id = uuid4()
        meeting_id_str = str(meeting_id)

        if meeting_url:
            logger.info("Using provided meeting URL: %s", meeting_url)
        else:
            logger.info("Creating %s meeting URL...", meeting_type)
            meeting_url = await create_platform_meeting_url(meeting_type, meeting_id_str)

        now = datetime.utcnow()

        meeting = Meeting(
            id=meeting_id,
            user_id=user_id,
            platform=platform_enum,
            meeting_url=meeting_url,
            meeting_id_external=meeting_id_str,
            title=f"Meeting {meeting_id_str[:8]}",
            description=description or "Meeting created via webhook",
            start_time=now,
            end_time=now + timedelta(minutes=30),
            organizer_email=settings.ai_email or "",
            participants=[],
            status=status or MeetingStatus.SCHEDULED,
            ai_email=settings.ai_email or "",
            audio_enabled=audio_record,
            video_enabled=video_record,
            recording_enabled=audio_record or video_record,
            transcript=transcript,
            voice_id=voice_id,
            bot_name=bot_name,
            context_id=context_id,
        )

        meeting = await self.context_service.apply_default_context_to_meeting(meeting)
        return await self.save_meeting(meeting)

    async def save_meeting(self, meeting: Meeting) -> Meeting:
        """Persist a meeting to MongoDB."""
        if self.dao.meetings_collection is None:
            raise RuntimeError("Meetings collection not initialized")

        return await self.dao.create_meeting(meeting)

    async def get_meeting(self, meeting_id: str) -> Optional[Meeting]:
        """Get a meeting by ID."""
        if self.dao.meetings_collection is None:
            raise RuntimeError("Meetings collection not initialized")

        return await self.dao.get_meeting(meeting_id)

    async def get_meetings(self, limit: int = 10, user_id: Optional[str] = None) -> List[Meeting]:
        """Get meetings with optional user filtering."""
        if self.dao.meetings_collection is None:
            raise RuntimeError("Meetings collection not initialized")

        return await self.dao.get_meetings(user_id=user_id, limit=limit)

    async def get_meeting_by_url(self, meeting_url: str) -> Optional[Meeting]:
        """Get a meeting by its URL."""
        if self.dao.meetings_collection is None:
            raise RuntimeError("Meetings collection not initialized")

        # Use MongoDB query to find by URL
        doc = await self.dao.meetings_collection.find_one({"meeting_url": meeting_url})
        if not doc:
            return None
        # Use the internal method to convert document to Meeting
        from uuid import UUID
        from shared.schemas import MeetingParticipant, MeetingSummary, ActionItem
        
        # Convert participants
        participants = []
        for p in doc.get("participants", []):
            participants.append(MeetingParticipant(
                email=p["email"],
                name=p.get("name"),
                is_organizer=p.get("is_organizer", False),
                response_status=p.get("response_status", "accepted"),
            ))
        
        # Convert summary if exists
        summary = None
        if doc.get("summary"):
            s = doc["summary"]
            action_items = []
            for ai in s.get("action_items", []):
                action_items.append(ActionItem(
                    id=UUID(ai["id"]),
                    description=ai["description"],
                    assignee=ai.get("assignee"),
                    due_date=datetime.fromisoformat(ai["due_date"]) if ai.get("due_date") else None,
                    priority=ai.get("priority", "medium"),
                    status=ai.get("status", "pending"),
                ))
            summary = MeetingSummary(
                id=UUID(s["id"]),
                meeting_id=UUID(s["meeting_id"]),
                topics_discussed=s.get("topics_discussed", []),
                key_decisions=s.get("key_decisions", []),
                action_items=action_items,
                summary_text=s["summary_text"],
                sentiment=s.get("sentiment"),
                duration_minutes=s.get("duration_minutes"),
                created_at=datetime.fromisoformat(s["created_at"]) if isinstance(s.get("created_at"), str) else s.get("created_at"),
            )
        
        return Meeting(
            id=UUID(doc["id"]),
            user_id=UUID(doc["user_id"]) if doc.get("user_id") else None,
            platform=MeetingPlatform(doc["platform"]) if isinstance(doc["platform"], str) else doc["platform"],
            meeting_url=doc["meeting_url"],
            meeting_id_external=doc["meeting_id_external"],
            title=doc["title"],
            description=doc.get("description"),
            start_time=doc["start_time"] if isinstance(doc["start_time"], datetime) else datetime.fromisoformat(doc["start_time"]),
            end_time=doc["end_time"] if isinstance(doc["end_time"], datetime) else datetime.fromisoformat(doc["end_time"]),
            organizer_email=doc["organizer_email"],
            participants=participants,
            status=MeetingStatus(doc["status"]) if isinstance(doc["status"], str) else doc["status"],
            ai_email=doc["ai_email"],
            transcription_chunks=doc.get("transcription_chunks", []),
            full_transcription=doc.get("full_transcription"),
            summary=summary,
            calendar_event_id=doc.get("calendar_event_id"),
            join_attempts=doc.get("join_attempts", 0),
            last_join_attempt=doc["last_join_attempt"] if isinstance(doc.get("last_join_attempt"), datetime) else (datetime.fromisoformat(doc["last_join_attempt"]) if doc.get("last_join_attempt") else None),
            error_message=doc.get("error_message"),
            bot_joined=doc.get("bot_joined", False),
            created_at=doc["created_at"] if isinstance(doc["created_at"], datetime) else datetime.fromisoformat(doc["created_at"]),
            updated_at=doc["updated_at"] if isinstance(doc["updated_at"], datetime) else datetime.fromisoformat(doc["updated_at"]),
            joined_at=doc["joined_at"] if isinstance(doc.get("joined_at"), datetime) else (datetime.fromisoformat(doc["joined_at"]) if doc.get("joined_at") else None),
            ended_at=doc["ended_at"] if isinstance(doc.get("ended_at"), datetime) else (datetime.fromisoformat(doc["ended_at"]) if doc.get("ended_at") else None),
            audio_enabled=doc.get("audio_enabled", True),
            video_enabled=doc.get("video_enabled", False),
            recording_enabled=doc.get("recording_enabled", False),
            transcript=doc.get("transcript", False),
            voice_id=doc.get("voice_id"),
            bot_name=doc.get("bot_name"),
            context_id=doc.get("context_id"),
        )

    async def delete_meeting(self, meeting_id: str) -> bool:
        """Delete a meeting."""
        if self.dao.meetings_collection is None:
            raise RuntimeError("Meetings collection not initialized")

        result = await self.dao.meetings_collection.delete_one({"id": meeting_id})
        logger.info(f"Deleted meeting: {meeting_id}")
        return result.deleted_count > 0

    async def update_meeting(self, meeting: Meeting) -> Meeting:
        """Update a meeting in MongoDB."""
        if self.dao.meetings_collection is None:
            raise RuntimeError("Meetings collection not initialized")

        meeting.updated_at = datetime.utcnow()
        meeting = await self.context_service.apply_default_context_to_meeting(meeting)

        return await self.dao.update_meeting(meeting)

    async def add_meeting_participant(self, meeting_id: str, participant: Dict[str, Any]) -> bool:
        """Add a participant to a meeting."""
        meeting = await self.get_meeting(meeting_id)
        if not meeting:
            logger.error(f"Meeting {meeting_id} not found")
            return False

        participant_obj = MeetingParticipant(
            email=participant["email"],
            name=participant.get("name", participant["email"]),
            is_organizer=participant.get("is_organizer", False),
            response_status=participant.get("response_status", "accepted"),
        )
        meeting.participants = list(meeting.participants or []) + [participant_obj]
        meeting.updated_at = datetime.utcnow()
        await self.update_meeting(meeting)
        return True

    async def remove_meeting_participant(self, meeting_id: str, participant_id: str) -> bool:
        """Remove a participant from a meeting."""
        meeting = await self.get_meeting(meeting_id)
        if not meeting:
            logger.error(f"Meeting {meeting_id} not found")
            return False

        meeting.participants = [
            participant
            for participant in meeting.participants or []
            if getattr(participant, "id", None) != participant_id
        ]
        meeting.updated_at = datetime.utcnow()
        await self.update_meeting(meeting)
        return True

    async def get_meeting_participants(self, meeting_id: str) -> List[Dict[str, Any]]:
        """Get participants for a meeting."""
        meeting = await self.get_meeting(meeting_id)
        if not meeting:
            logger.error(f"Meeting {meeting_id} not found")
            return []
        return [
            {
                "email": participant.email,
                "name": participant.name,
                "is_organizer": participant.is_organizer,
                "response_status": participant.response_status,
            }
            for participant in meeting.participants or []
        ]


__all__ = ["ServiceMeeting"]
