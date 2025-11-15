"""
Meeting service helpers for reusable meeting operations.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from boto3.dynamodb.conditions import Attr, Key

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
from .dao import DynamoDBDAO
from .meeting_url import create_platform_meeting_url


logger = logging.getLogger(__name__)
settings = get_settings()


class ServiceMeeting:
    """Service layer for meeting creation and management."""

    def __init__(self, dao: DynamoDBDAO):
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
        """Persist a meeting to DynamoDB."""
        if not self.dao.meetings_table:
            raise RuntimeError("Meetings table not initialized")

        item = self._meeting_to_item(meeting)
        self.dao.meetings_table.put_item(Item=item)
        logger.info(f"Saved meeting: {meeting.id}")
        return meeting

    async def get_meeting(self, meeting_id: str) -> Optional[Meeting]:
        """Get a meeting by ID."""
        if not self.dao.meetings_table:
            raise RuntimeError("Meetings table not initialized")

        response = self.dao.meetings_table.get_item(Key={"id": meeting_id})
        if "Item" not in response:
            return None
        return self._item_to_meeting(response["Item"])

    async def get_meetings(self, limit: int = 10, user_id: Optional[str] = None) -> List[Meeting]:
        """Get meetings with optional user filtering."""
        if not self.dao.meetings_table:
            raise RuntimeError("Meetings table not initialized")

        if user_id:
            response = self.dao.meetings_table.query(
                IndexName="user-id-start-time-index",
                KeyConditionExpression=Key("user_id").eq(user_id),
                ScanIndexForward=False,
                Limit=limit,
            )
            items = response.get("Items", [])
        else:
            response = self.dao.meetings_table.scan(Limit=limit)
            items = response.get("Items", [])

        return [self._item_to_meeting(item) for item in items]

    async def get_meeting_by_url(self, meeting_url: str) -> Optional[Meeting]:
        """Get a meeting by its URL."""
        if not self.dao.meetings_table:
            raise RuntimeError("Meetings table not initialized")

        response = self.dao.meetings_table.scan(
            FilterExpression=Attr("meeting_url").eq(meeting_url),
            Limit=1,
        )
        items = response.get("Items", [])
        if not items:
            return None
        return self._item_to_meeting(items[0])

    async def delete_meeting(self, meeting_id: str) -> bool:
        """Delete a meeting."""
        if not self.dao.meetings_table:
            raise RuntimeError("Meetings table not initialized")

        self.dao.meetings_table.delete_item(Key={"id": meeting_id})
        logger.info(f"Deleted meeting: {meeting_id}")
        return True

    async def update_meeting(self, meeting: Meeting) -> Meeting:
        """Update a meeting in DynamoDB."""
        if not self.dao.meetings_table:
            raise RuntimeError("Meetings table not initialized")

        meeting.updated_at = datetime.utcnow()
        meeting = await self.context_service.apply_default_context_to_meeting(meeting)

        item = self._meeting_to_item(meeting)
        self.dao.meetings_table.put_item(Item=item)
        logger.info(f"Updated meeting: {meeting.id}")
        return meeting

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

    def _meeting_to_item(self, meeting: Meeting) -> Dict[str, Any]:
        """Serialize a meeting into a DynamoDB item."""
        participants_data = [
            {
                "email": p.email,
                "name": p.name,
                "is_organizer": p.is_organizer,
                "response_status": p.response_status,
            }
            for p in (meeting.participants or [])
        ]

        summary_data = None
        if meeting.summary:
            if isinstance(meeting.summary, dict):
                summary_data = meeting.summary
            elif isinstance(meeting.summary, MeetingSummary):
                summary_data = meeting.summary.dict()
                summary_data["meeting_id"] = str(summary_data.get("meeting_id", meeting.id))
                summary_data["action_items"] = [
                    ai.dict() if isinstance(ai, ActionItem) else ai
                    for ai in summary_data.get("action_items", [])
                ]

        item = {
            "id": str(meeting.id),
            "user_id": str(meeting.user_id) if meeting.user_id else "",
            "platform": meeting.platform.value,
            "meeting_url": meeting.meeting_url,
            "meeting_id_external": meeting.meeting_id_external,
            "title": meeting.title,
            "description": meeting.description or "",
            "start_time": meeting.start_time.isoformat(),
            "end_time": meeting.end_time.isoformat(),
            "organizer_email": meeting.organizer_email,
            "participants": participants_data,
            "status": meeting.status.value,
            "ai_email": meeting.ai_email,
            "transcription_chunks": meeting.transcription_chunks or [],
            "full_transcription": meeting.full_transcription or "",
            "summary": summary_data,
            "calendar_event_id": meeting.calendar_event_id or "",
            "join_attempts": meeting.join_attempts,
            "last_join_attempt": meeting.last_join_attempt.isoformat() if meeting.last_join_attempt else None,
            "error_message": meeting.error_message or "",
            "created_at": meeting.created_at.isoformat(),
            "updated_at": meeting.updated_at.isoformat(),
            "joined_at": meeting.joined_at.isoformat() if meeting.joined_at else None,
            "ended_at": meeting.ended_at.isoformat() if meeting.ended_at else None,
            "audio_enabled": meeting.audio_enabled,
            "video_enabled": meeting.video_enabled,
            "recording_enabled": meeting.recording_enabled,
            "bot_joined": meeting.bot_joined,
            "transcript": meeting.transcript,
            "voice_id": meeting.voice_id or "",
            "bot_name": meeting.bot_name or "",
            "context_id": meeting.context_id or "",
        }

        item = {k: v for k, v in item.items() if v is not None}
        return item

    def _item_to_meeting(self, item: Dict[str, Any]) -> Meeting:
        """Convert a DynamoDB meeting item into a Meeting model."""
        participants = [
            MeetingParticipant(
                email=p["email"],
                name=p["name"],
                is_organizer=p["is_organizer"],
                response_status=p["response_status"],
            )
            for p in item.get("participants", [])
        ]

        summary_data = item.get("summary")
        summary = None
        if summary_data:
            if isinstance(summary_data, dict):
                summary_data = {**summary_data}
                summary_data.setdefault("meeting_id", item["id"])
                action_items = []
                for ai in summary_data.get("action_items", []):
                    if isinstance(ai, dict):
                        action_items.append(ActionItem(**ai))
                    else:
                        action_items.append(ai)
                try:
                    summary = MeetingSummary(
                        meeting_id=summary_data.get("meeting_id"),
                        topics_discussed=summary_data.get("topics_discussed", summary_data.get("topics", [])),
                        key_decisions=summary_data.get("key_decisions", summary_data.get("decisions", [])),
                        action_items=action_items,
                        summary_text=summary_data.get("summary_text", ""),
                        sentiment=summary_data.get("sentiment", "neutral"),
                        duration_minutes=summary_data.get("duration_minutes"),
                        created_at=datetime.fromisoformat(summary_data["created_at"])
                        if summary_data.get("created_at")
                        else datetime.utcnow(),
                    )
                except Exception as exc:
                    logger.warning(f"Failed to parse summary for meeting {item['id']}: {exc}")
                    summary = None
            else:
                summary = summary_data

        return Meeting(
            id=item["id"],
            user_id=UUID(item["user_id"]) if item.get("user_id") else None,
            platform=MeetingPlatform(item["platform"]),
            meeting_url=item["meeting_url"],
            meeting_id_external=item.get("meeting_id_external", ""),
            title=item["title"],
            description=item.get("description", ""),
            start_time=datetime.fromisoformat(item["start_time"]),
            end_time=datetime.fromisoformat(item["end_time"]),
            organizer_email=item["organizer_email"],
            participants=participants,
            status=MeetingStatus(item["status"]),
            ai_email=item.get("ai_email"),
            transcription_chunks=item.get("transcription_chunks", []),
            full_transcription=item.get("full_transcription", ""),
            summary=summary,
            calendar_event_id=item.get("calendar_event_id", ""),
            join_attempts=item.get("join_attempts", 0),
            last_join_attempt=datetime.fromisoformat(item["last_join_attempt"])
            if item.get("last_join_attempt")
            else None,
            error_message=item.get("error_message", ""),
            created_at=datetime.fromisoformat(item["created_at"]),
            updated_at=datetime.fromisoformat(item["updated_at"]),
            joined_at=datetime.fromisoformat(item["joined_at"]) if item.get("joined_at") else None,
            ended_at=datetime.fromisoformat(item["ended_at"]) if item.get("ended_at") else None,
            audio_enabled=item.get("audio_enabled", True),
            video_enabled=item.get("video_enabled", False),
            recording_enabled=item.get("recording_enabled", False),
            bot_joined=item.get("bot_joined", False),
            transcript=item.get("transcript", False),
            voice_id=item.get("voice_id") or None,
            bot_name=item.get("bot_name") or None,
            context_id=item.get("context_id") or None,
        )


__all__ = ["ServiceMeeting"]


