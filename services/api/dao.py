"""
Data Access Object for DynamoDB operations.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError

from shared.config import get_settings
from shared.schemas import (Action, ActionStatus, RoomInfo, User, Meeting, ApiKey, ApiKeyStatus)

logger = logging.getLogger(__name__)
settings = get_settings()


class DynamoDBDAO:
    """DynamoDB Data Access Object."""

    def __init__(self):
        self.dynamodb = None
        self.actions_table = None
        self.users_table = None
        self.rooms_table = None
        self.meetings_table = None
        self.api_keys_table = None
        self.initialized = False

    async def initialize(self):
        """Initialize DynamoDB client and tables."""
        try:
            if settings.aws_access_key_id:
                self.dynamodb = boto3.resource(
                    "dynamodb",
                    region_name=settings.aws_region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                )
                logger.info("🌐 DynamoDB: Connected to AWS (ONLINE)")
            else:
                # Use local DynamoDB for development
                self.dynamodb = boto3.resource(
                    "dynamodb",
                    region_name=settings.aws_region,
                    endpoint_url="http://localhost:8001",
                )
                logger.info("🏠 DynamoDB: Connected to Local (OFFLINE)")

            # Get table references
            self.actions_table = self.dynamodb.Table(
                f"{settings.dynamodb_table_prefix}{settings.actions_table}"
            )
            self.users_table = self.dynamodb.Table(
                f"{settings.dynamodb_table_prefix}{settings.users_table}"
            )
            self.rooms_table = self.dynamodb.Table(
                f"{settings.dynamodb_table_prefix}rooms"
            )
            self.meetings_table = self.dynamodb.Table(
                f"{settings.dynamodb_table_prefix}meetings"
            )
            self.api_keys_table = self.dynamodb.Table(
                f"{settings.dynamodb_table_prefix}{settings.api_keys_table}"
            )

            self.initialized = True
            logger.info("DynamoDB DAO initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DynamoDB DAO: {e}")
            raise

    # Action operations
    async def get_actions(
        self,
        action_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Action]:
        """Get actions with filtering."""
        try:
            scan_params = {"Limit": limit}
            
            if offset > 0:
                scan_params["ExclusiveStartKey"] = {"id": str(offset)}
            
            response = self.actions_table.scan(**scan_params)

            actions = []
            for item in response.get("Items", []):
                if action_type and item["action_type"] != action_type:
                    continue
                if status and item["status"] != status:
                    continue

                action = Action(
                    id=UUID(item["id"]),
                    turn_id=UUID(item["turn_id"]) if item.get("turn_id") else None,
                    action_type=item["action_type"],
                    status=ActionStatus(item["status"]),
                    parameters=json.loads(item.get("parameters", "{}")),
                    result=json.loads(item["result"]) if item.get("result") else None,
                    error_message=item.get("error_message"),
                    created_at=datetime.fromisoformat(item["created_at"]),
                    completed_at=datetime.fromisoformat(item["completed_at"])
                    if item.get("completed_at")
                    else None,
                )
                actions.append(action)

            return actions

        except Exception as e:
            logger.error(f"Failed to get actions: {e}")
            raise

    async def get_action(self, action_id: str) -> Optional[Action]:
        """Get an action by ID."""
        try:
            response = self.actions_table.get_item(Key={"id": action_id})

            if "Item" not in response:
                return None

            item = response["Item"]
            return Action(
                id=UUID(item["id"]),
                turn_id=UUID(item["turn_id"]) if item.get("turn_id") else None,
                action_type=item["action_type"],
                status=ActionStatus(item["status"]),
                parameters=json.loads(item.get("parameters", "{}")),
                result=json.loads(item["result"]) if item.get("result") else None,
                error_message=item.get("error_message"),
                created_at=datetime.fromisoformat(item["created_at"]),
                completed_at=datetime.fromisoformat(item["completed_at"])
                if item.get("completed_at")
                else None,
            )

        except Exception as e:
            logger.error(f"Failed to get action {action_id}: {e}")
            raise

    async def update_action(self, action: Action) -> Action:
        """Update an action."""
        try:
            item = {
                "id": str(action.id),
                "turn_id": str(action.turn_id) if action.turn_id else None,
                "action_type": action.action_type.value,
                "status": action.status.value,
                "parameters": json.dumps(action.parameters),
                "result": json.dumps(action.result) if action.result else None,
                "error_message": action.error_message,
                "created_at": action.created_at.isoformat(),
                "completed_at": action.completed_at.isoformat()
                if action.completed_at
                else None,
            }

            self.actions_table.put_item(Item=item)
            logger.info(f"Updated action: {action.id}")
            return action

        except Exception as e:
            logger.error(f"Failed to update action: {e}")
            raise

    # Room operations
    async def get_active_rooms(self) -> List[RoomInfo]:
        """Get list of active rooms."""
        try:
            response = self.rooms_table.scan(
                FilterExpression=Attr("is_active").eq(True)
            )

            rooms = []
            for item in response.get("Items", []):
                room = RoomInfo(
                    room_id=item["room_id"],
                    name=item["name"],
                    participant_count=item["participant_count"],
                    is_active=item["is_active"],
                    created_at=datetime.fromisoformat(item["created_at"]),
                )
                rooms.append(room)

            return rooms

        except Exception as e:
            logger.error(f"Failed to get active rooms: {e}")
            return []

    async def get_room(self, room_id: str) -> Optional[RoomInfo]:
        """Get a room by ID."""
        try:
            response = self.rooms_table.get_item(Key={"room_id": room_id})

            if "Item" not in response:
                return None

            item = response["Item"]
            return RoomInfo(
                room_id=item["room_id"],
                name=item["name"],
                participant_count=item["participant_count"],
                is_active=item["is_active"],
                created_at=datetime.fromisoformat(item["created_at"]),
            )

        except Exception as e:
            logger.error(f"Failed to get room {room_id}: {e}")
            raise

    async def create_room(self, room: RoomInfo) -> RoomInfo:
        """Create a new room."""
        try:
            item = {
                "room_id": room.room_id,
                "name": room.name,
                "participant_count": room.participant_count,
                "is_active": room.is_active,
                "created_at": room.created_at.isoformat(),
            }

            self.rooms_table.put_item(Item=item)
            logger.info(f"Created room: {room.room_id}")
            return room

        except Exception as e:
            logger.error(f"Failed to create room: {e}")
            raise

    async def update_room(self, room: RoomInfo) -> RoomInfo:
        """Update a room."""
        try:
            item = {
                "room_id": room.room_id,
                "name": room.name,
                "participant_count": room.participant_count,
                "is_active": room.is_active,
                "created_at": room.created_at.isoformat(),
            }

            self.rooms_table.put_item(Item=item)
            logger.info(f"Updated room: {room.room_id}")
            return room

        except Exception as e:
            logger.error(f"Failed to update room: {e}")
            raise

    async def delete_room(self, room_id: str):
        """Delete a room."""
        try:
            self.rooms_table.delete_item(Key={"room_id": room_id})
            logger.info(f"Deleted room: {room_id}")

        except Exception as e:
            logger.error(f"Failed to delete room: {e}")
            raise

    # Meeting operations
    async def create_meeting(self, meeting: Meeting) -> Meeting:
        """Create a new meeting."""
        try:
            # Convert participants to DynamoDB format
            participants_data = []
            for participant in meeting.participants:
                participants_data.append({
                    "email": participant.email,
                    "name": participant.name,
                    "is_organizer": participant.is_organizer,
                    "response_status": participant.response_status
                })

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
                "summary": meeting.summary or "",
                "calendar_event_id": meeting.calendar_event_id or "",
                "join_attempts": meeting.join_attempts,
                "last_join_attempt": meeting.last_join_attempt.isoformat() if meeting.last_join_attempt else "",
                "error_message": meeting.error_message or "",
                "created_at": meeting.created_at.isoformat(),
                "updated_at": meeting.updated_at.isoformat(),
                "joined_at": meeting.joined_at.isoformat() if meeting.joined_at else "",
                "ended_at": meeting.ended_at.isoformat() if meeting.ended_at else "",
                "audio_enabled": meeting.audio_enabled,
                "video_enabled": meeting.video_enabled,
                "recording_enabled": meeting.recording_enabled,
                "bot_joined": meeting.bot_joined,
            }

            self.meetings_table.put_item(Item=item)
            logger.info(f"Created meeting: {meeting.id}")
            return meeting

        except Exception as e:
            logger.error(f"Failed to create meeting: {e}")
            raise

    async def get_meeting(self, meeting_id: str) -> Optional[Meeting]:
        """Get a meeting by ID."""
        try:
            response = self.meetings_table.get_item(Key={"id": meeting_id})
            
            if "Item" not in response:
                return None

            item = response["Item"]
            return self._item_to_meeting(item)

        except Exception as e:
            logger.error(f"Failed to get meeting: {e}")
            raise

    async def get_meetings(self, limit: int = 10, user_id: Optional[str] = None) -> List[Meeting]:
        """Get meetings with pagination. Optionally filter by user_id."""
        try:
            if user_id:
                # Use GSI to query by user_id
                response = self.meetings_table.query(
                    IndexName="user-id-start-time-index",
                    KeyConditionExpression=Key("user_id").eq(user_id),
                    ScanIndexForward=False,  # Sort by start_time descending
                    Limit=limit
                )
                items = response.get("Items", [])
            else:
                response = self.meetings_table.scan(Limit=limit)
                items = response.get("Items", [])
            
            meetings = []
            for item in items:
                meetings.append(self._item_to_meeting(item))
            
            return meetings

        except Exception as e:
            logger.error(f"Failed to get meetings: {e}")
            raise

    async def get_meeting_by_url(self, meeting_url: str) -> Optional[Meeting]:
        """Get a meeting by URL."""
        try:
            # Scan through meetings to find one matching the URL
            # Note: This is inefficient for large datasets. Consider adding a GSI on meeting_url
            response = self.meetings_table.scan(
                FilterExpression=Attr('meeting_url').eq(meeting_url),
                Limit=1
            )
            
            if "Items" not in response or len(response["Items"]) == 0:
                return None
            
            item = response["Items"][0]
            return self._item_to_meeting(item)

        except Exception as e:
            logger.error(f"Failed to get meeting by URL: {e}")
            raise

    async def delete_meeting(self, meeting_id: str) -> bool:
        """Delete a meeting from DynamoDB."""
        try:
            self.meetings_table.delete_item(Key={"id": meeting_id})
            logger.info(f"Deleted meeting: {meeting_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete meeting {meeting_id}: {e}")
            raise

    async def update_meeting(self, meeting: Meeting) -> Meeting:
        """Update a meeting in DynamoDB."""
        try:
            # Convert participants to dict
            participants_data = [
                {
                    "email": p.email,
                    "name": p.name,
                    "is_organizer": p.is_organizer,
                    "response_status": p.response_status
                }
                for p in (meeting.participants or [])
            ]
            
            # Convert summary to dict if it's a MeetingSummary object
            summary_data = None
            if meeting.summary:
                if hasattr(meeting.summary, 'dict'):
                    # It's a Pydantic model, convert to dict
                    summary_dict = meeting.summary.dict()
                    # Convert UUIDs and datetimes to strings
                    summary_data = {
                        "meeting_id": str(summary_dict["meeting_id"]),
                        "topics_discussed": summary_dict.get("topics_discussed", []),
                        "key_decisions": summary_dict.get("key_decisions", []),
                        "action_items": [
                            {
                                "description": ai.get("description", ""),
                                "assignee": ai.get("assignee"),
                                "priority": ai.get("priority", "medium"),
                                "due_date": ai.get("due_date").isoformat() if ai.get("due_date") else None
                            }
                            for ai in summary_dict.get("action_items", [])
                        ],
                        "summary_text": summary_dict.get("summary_text", ""),
                        "sentiment": summary_dict.get("sentiment", "neutral"),
                        "duration_minutes": summary_dict.get("duration_minutes"),
                        "created_at": summary_dict["created_at"].isoformat() if summary_dict.get("created_at") else None
                    }
                elif isinstance(meeting.summary, dict):
                    # Already a dict
                    summary_data = meeting.summary
            
            # Prepare update data
            item = {
                "id": str(meeting.id),
                "user_id": str(meeting.user_id) if meeting.user_id else "",
                "platform": meeting.platform.value,
                "meeting_url": meeting.meeting_url,
                "meeting_id_external": meeting.meeting_id_external,
                "title": meeting.title,
                "description": meeting.description,
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
                "bot_joined": meeting.bot_joined,
                "created_at": meeting.created_at.isoformat(),
                "updated_at": meeting.updated_at.isoformat(),
                "joined_at": meeting.joined_at.isoformat() if meeting.joined_at else None,
                "ended_at": meeting.ended_at.isoformat() if meeting.ended_at else None,
                "audio_enabled": meeting.audio_enabled,
                "video_enabled": meeting.video_enabled,
                "recording_enabled": meeting.recording_enabled,
            }
            
            # Remove None values
            item = {k: v for k, v in item.items() if v is not None}
            
            # Update in DynamoDB
            self.meetings_table.put_item(Item=item)
            logger.info(f"Updated meeting: {meeting.id}")
            
            return meeting

        except Exception as e:
            logger.error(f"Failed to update meeting: {e}")
            raise

    def _item_to_meeting(self, item: Dict[str, Any]) -> Meeting:
        """Convert DynamoDB item to Meeting object."""
        from shared.schemas import MeetingParticipant, MeetingPlatform, MeetingStatus
        from datetime import datetime
        
        # Convert participants
        participants = []
        for p in item.get("participants", []):
            participants.append(MeetingParticipant(
                email=p["email"],
                name=p["name"],
                is_organizer=p["is_organizer"],
                response_status=p["response_status"]
            ))

        # Handle summary field - convert old dict format to MeetingSummary if needed
        summary_data = item.get("summary")
        summary = None
        if summary_data:
            # Check if it's already a MeetingSummary object or needs conversion
            if isinstance(summary_data, dict):
                # Old format - convert to MeetingSummary with meeting_id
                if "meeting_id" not in summary_data:
                    # Add missing meeting_id for backward compatibility
                    summary_data["meeting_id"] = item["id"]
                try:
                    from shared.schemas import MeetingSummary, ActionItem
                    # Convert action_items dicts to ActionItem objects if needed
                    action_items = []
                    for ai in summary_data.get("action_items", []):
                        if isinstance(ai, dict):
                            action_items.append(ActionItem(**ai))
                        else:
                            action_items.append(ai)
                    
                    summary = MeetingSummary(
                        meeting_id=summary_data.get("meeting_id", item["id"]),
                        topics_discussed=summary_data.get("topics_discussed", summary_data.get("topics", [])),
                        key_decisions=summary_data.get("key_decisions", summary_data.get("decisions", [])),
                        action_items=action_items,
                        summary_text=summary_data.get("summary_text", ""),
                        sentiment=summary_data.get("sentiment", "neutral"),
                        duration_minutes=summary_data.get("duration_minutes"),
                        created_at=datetime.fromisoformat(summary_data["created_at"]) if summary_data.get("created_at") else datetime.utcnow()
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse summary for meeting {item['id']}: {e}")
                    summary = None
            else:
                summary = summary_data

        return Meeting(
            id=item["id"],
            user_id=UUID(item["user_id"]) if item.get("user_id") else None,
            platform=MeetingPlatform(item["platform"]),
            meeting_url=item["meeting_url"],
            meeting_id_external=item["meeting_id_external"],
            title=item["title"],
            description=item.get("description", ""),
            start_time=datetime.fromisoformat(item["start_time"]),
            end_time=datetime.fromisoformat(item["end_time"]),
            organizer_email=item["organizer_email"],
            participants=participants,
            status=MeetingStatus(item["status"]),
            ai_email=item["ai_email"],
            transcription_chunks=item.get("transcription_chunks", []),
            full_transcription=item.get("full_transcription", ""),
            summary=summary,
            calendar_event_id=item.get("calendar_event_id", ""),
            join_attempts=item.get("join_attempts", 0),
            last_join_attempt=datetime.fromisoformat(item["last_join_attempt"]) if item.get("last_join_attempt") else None,
            error_message=item.get("error_message", ""),
            created_at=datetime.fromisoformat(item["created_at"]),
            updated_at=datetime.fromisoformat(item["updated_at"]),
            joined_at=datetime.fromisoformat(item["joined_at"]) if item.get("joined_at") else None,
            ended_at=datetime.fromisoformat(item["ended_at"]) if item.get("ended_at") else None,
            audio_enabled=item.get("audio_enabled", True),
            video_enabled=item.get("video_enabled", False),
            recording_enabled=item.get("recording_enabled", False),
            bot_joined=item.get("bot_joined", False),
        )

    async def add_meeting_participant(self, meeting_id: str, participant: Dict[str, Any]) -> bool:
        """Add a participant to a meeting."""
        try:
            if not self.meetings_table:
                logger.warning("Meetings table not initialized")
                return False
            
            # Get the meeting first
            response = self.meetings_table.get_item(
                Key={'id': meeting_id}
            )
            
            if 'Item' not in response:
                logger.error(f"Meeting {meeting_id} not found")
                return False
            
            meeting = response['Item']
            
            # Initialize participants list if it doesn't exist
            if 'participants' not in meeting:
                meeting['participants'] = []
            
            # Add the new participant
            meeting['participants'].append(participant)
            meeting['participant_count'] = len(meeting['participants'])
            meeting['updated_at'] = datetime.utcnow().isoformat()
            
            # Update the meeting
            self.meetings_table.put_item(Item=meeting)
            
            logger.info(f"Added participant {participant['id']} to meeting {meeting_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding meeting participant: {e}")
            return False

    async def remove_meeting_participant(self, meeting_id: str, participant_id: str) -> bool:
        """Remove a participant from a meeting."""
        try:
            if not self.meetings_table:
                logger.warning("Meetings table not initialized")
                return False
            
            # Get the meeting first
            response = self.meetings_table.get_item(
                Key={'id': meeting_id}
            )
            
            if 'Item' not in response:
                logger.error(f"Meeting {meeting_id} not found")
                return False
            
            meeting = response['Item']
            
            # Remove the participant
            if 'participants' in meeting:
                meeting['participants'] = [
                    p for p in meeting['participants'] 
                    if p['id'] != participant_id
                ]
                meeting['participant_count'] = len(meeting['participants'])
                meeting['updated_at'] = datetime.utcnow().isoformat()
                
                # Update the meeting
                self.meetings_table.put_item(Item=meeting)
                
                logger.info(f"Removed participant {participant_id} from meeting {meeting_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing meeting participant: {e}")
            return False

    async def get_meeting_participants(self, meeting_id: str) -> List[Dict[str, Any]]:
        """Get all participants for a meeting."""
        try:
            if not self.meetings_table:
                logger.warning("Meetings table not initialized")
                return []
            
            response = self.meetings_table.get_item(
                Key={'id': meeting_id}
            )
            
            if 'Item' not in response:
                logger.error(f"Meeting {meeting_id} not found")
                return []
            
            meeting = response['Item']
            return meeting.get('participants', [])
            
        except Exception as e:
            logger.error(f"Error getting meeting participants: {e}")
            return []

    # User operations
    async def create_user(self, user: User) -> User:
        """Create a new user."""
        try:
            if not self.users_table:
                raise RuntimeError("Users table not initialized")
            
            # Check if user with this email already exists
            existing_user = await self.get_user_by_email(user.email)
            if existing_user:
                raise ValueError(f"User with email {user.email} already exists")
            
            item = {
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "phone": user.phone or "",
                "password_hash": user.password_hash or "",
                "auth_provider": user.auth_provider or "",
                "timezone": user.timezone,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "updated_at": user.updated_at.isoformat(),
            }
            
            self.users_table.put_item(Item=item)
            logger.info(f"Created user: {user.id} ({user.email})")
            return user

        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        try:
            if not self.users_table:
                raise RuntimeError("Users table not initialized")
            
            response = self.users_table.get_item(Key={"id": user_id})
            
            if "Item" not in response:
                return None
            
            item = response["Item"]
            return User(
                id=UUID(item["id"]),
                email=item["email"],
                name=item["name"],
                phone=item.get("phone") or None,
                password_hash=item.get("password_hash") or None,
                auth_provider=item.get("auth_provider") or None,
                timezone=item.get("timezone", "UTC"),
                is_active=item.get("is_active", True),
                created_at=datetime.fromisoformat(item["created_at"]),
                updated_at=datetime.fromisoformat(item["updated_at"]),
            )

        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            raise

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email using GSI."""
        try:
            if not self.users_table:
                raise RuntimeError("Users table not initialized")
            
            response = self.users_table.query(
                IndexName="email-index",
                KeyConditionExpression=Key("email").eq(email),
                Limit=1
            )
            
            if not response.get("Items"):
                return None
            
            item = response["Items"][0]
            return User(
                id=UUID(item["id"]),
                email=item["email"],
                name=item["name"],
                phone=item.get("phone") or None,
                password_hash=item["password_hash"],
                timezone=item.get("timezone", "UTC"),
                is_active=item.get("is_active", True),
                created_at=datetime.fromisoformat(item["created_at"]),
                updated_at=datetime.fromisoformat(item["updated_at"]),
            )

        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            raise

    async def update_user(self, user: User) -> User:
        """Update a user."""
        try:
            if not self.users_table:
                raise RuntimeError("Users table not initialized")
            
            # Check if user exists
            existing = await self.get_user_by_id(str(user.id))
            if not existing:
                raise ValueError(f"User {user.id} not found")
            
            item = {
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "phone": user.phone or "",
                "password_hash": user.password_hash or "",
                "auth_provider": user.auth_provider or "",
                "timezone": user.timezone,
                "is_active": user.is_active,
                "created_at": existing.created_at.isoformat(),  # Preserve original
                "updated_at": datetime.utcnow().isoformat(),
            }
            
            self.users_table.put_item(Item=item)
            logger.info(f"Updated user: {user.id}")
            return user

        except Exception as e:
            logger.error(f"Failed to update user: {e}")
            raise

    # API Key operations
    async def create_api_key(self, api_key: ApiKey) -> ApiKey:
        """Create a new API key."""
        try:
            if not self.api_keys_table:
                raise RuntimeError("API keys table not initialized")
            
            item = {
                "id": str(api_key.id),
                "user_id": str(api_key.user_id),
                "name": api_key.name,
                "key_hash": api_key.key_hash,
                "key_prefix": api_key.key_prefix,
                "status": api_key.status.value,
                "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
                "scopes": json.dumps(api_key.scopes),
                "created_at": api_key.created_at.isoformat(),
                "updated_at": api_key.updated_at.isoformat(),
            }
            
            self.api_keys_table.put_item(Item=item)
            logger.info(f"Created API key: {api_key.id} for user {api_key.user_id}")
            return api_key

        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise

    async def get_api_keys_by_user(self, user_id: str) -> List[ApiKey]:
        """Get all API keys for a user."""
        try:
            if not self.api_keys_table:
                raise RuntimeError("API keys table not initialized")
            
            response = self.api_keys_table.query(
                IndexName="user-id-index",
                KeyConditionExpression=Key("user_id").eq(user_id),
                ScanIndexForward=False,  # Sort by created_at descending
            )
            
            api_keys = []
            for item in response.get("Items", []):
                api_key = ApiKey(
                    id=UUID(item["id"]),
                    user_id=UUID(item["user_id"]),
                    name=item["name"],
                    key_hash=item["key_hash"],
                    key_prefix=item["key_prefix"],
                    status=ApiKeyStatus(item["status"]),
                    last_used_at=datetime.fromisoformat(item["last_used_at"]) if item.get("last_used_at") else None,
                    expires_at=datetime.fromisoformat(item["expires_at"]) if item.get("expires_at") else None,
                    scopes=json.loads(item.get("scopes", "[]")),
                    created_at=datetime.fromisoformat(item["created_at"]),
                    updated_at=datetime.fromisoformat(item["updated_at"]),
                )
                api_keys.append(api_key)
            
            return api_keys

        except Exception as e:
            logger.error(f"Failed to get API keys for user {user_id}: {e}")
            raise

    async def get_api_key_by_id(self, api_key_id: str) -> Optional[ApiKey]:
        """Get an API key by ID."""
        try:
            if not self.api_keys_table:
                raise RuntimeError("API keys table not initialized")
            
            response = self.api_keys_table.get_item(Key={"id": api_key_id})
            
            if "Item" not in response:
                return None
            
            item = response["Item"]
            return ApiKey(
                id=UUID(item["id"]),
                user_id=UUID(item["user_id"]),
                name=item["name"],
                key_hash=item["key_hash"],
                key_prefix=item["key_prefix"],
                status=ApiKeyStatus(item["status"]),
                last_used_at=datetime.fromisoformat(item["last_used_at"]) if item.get("last_used_at") else None,
                expires_at=datetime.fromisoformat(item["expires_at"]) if item.get("expires_at") else None,
                scopes=json.loads(item.get("scopes", "[]")),
                created_at=datetime.fromisoformat(item["created_at"]),
                updated_at=datetime.fromisoformat(item["updated_at"]),
            )

        except Exception as e:
            logger.error(f"Failed to get API key {api_key_id}: {e}")
            raise

    async def validate_api_key(self, api_key_token: str) -> Optional[ApiKey]:
        """Validate an API key token and return the ApiKey if valid."""
        try:
            if not self.api_keys_table:
                logger.error("API keys table not initialized")
                raise RuntimeError("API keys table not initialized")
            
            # Extract prefix from token (e.g., "sk_live_12345678...")
            if not api_key_token.startswith("sk_"):
                logger.debug("API key token doesn't start with 'sk_'")
                return None
            
            # Try to extract prefix (first 12 chars: "sk_live_1234")
            prefix = api_key_token[:12]
            logger.info(f"Looking for API key with prefix: {prefix}")
            logger.info(f"Full token length: {len(api_key_token)}")
            logger.info(f"Full token (first 20 chars): {api_key_token[:20]}")
            
            # Scan for keys with matching prefix
            response = self.api_keys_table.scan(
                FilterExpression=Attr("key_prefix").eq(prefix),
            )
            
            items = response.get("Items", [])
            logger.info(f"Found {len(items)} API keys with prefix {prefix}")
            
            # Log all found prefixes for debugging
            if items:
                for item in items:
                    logger.info(f"Found key: id={item.get('id')}, prefix={item.get('key_prefix')}, status={item.get('status')}")
            else:
                # If no items found, let's check what prefixes exist
                logger.warning(f"No keys found with prefix '{prefix}'. Checking all keys...")
                all_keys_response = self.api_keys_table.scan()
                all_items = all_keys_response.get("Items", [])
                logger.warning(f"Total API keys in database: {len(all_items)}")
                for item in all_items[:5]:  # Log first 5
                    logger.warning(f"  Existing key prefix: {item.get('key_prefix')} (id: {item.get('id')})")
            
            if len(items) == 0:
                logger.warning(f"No API keys found with prefix {prefix}")
                return None
            
            # Verify the key hash matches
            from .auth import verify_password
            
            for item in items:
                key_hash = item["key_hash"]
                key_id = item.get("id", "unknown")
                stored_prefix = item.get("key_prefix", "unknown")
                logger.info(f"Verifying API key {key_id} (stored prefix: {stored_prefix})...")
                
                # For API keys, we use bcrypt-like verification
                verification_result = verify_password(api_key_token, key_hash)
                logger.info(f"Password verification result for key {key_id}: {verification_result}")
                
                if verification_result:
                    logger.info(f"Password verification successful for key {key_id}")
                    api_key = ApiKey(
                        id=UUID(item["id"]),
                        user_id=UUID(item["user_id"]),
                        name=item["name"],
                        key_hash=item["key_hash"],
                        key_prefix=item["key_prefix"],
                        status=ApiKeyStatus(item["status"]),
                        last_used_at=datetime.fromisoformat(item["last_used_at"]) if item.get("last_used_at") else None,
                        expires_at=datetime.fromisoformat(item["expires_at"]) if item.get("expires_at") else None,
                        scopes=json.loads(item.get("scopes", "[]")),
                        created_at=datetime.fromisoformat(item["created_at"]),
                        updated_at=datetime.fromisoformat(item["updated_at"]),
                    )
                    
                    # Check if key is active and not expired
                    if api_key.status != ApiKeyStatus.ACTIVE:
                        logger.warning(f"API key {key_id} is not active (status: {api_key.status})")
                        return None
                    
                    if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                        logger.warning(f"API key {key_id} is expired")
                        # Mark as expired
                        api_key.status = ApiKeyStatus.EXPIRED
                        await self.update_api_key(api_key)
                        return None
                    
                    logger.info(f"API key {key_id} validated successfully")
                    # Update last_used_at
                    api_key.last_used_at = datetime.utcnow()
                    await self.update_api_key(api_key)
                    
                    return api_key
                else:
                    logger.warning(f"Password verification failed for key {key_id} (stored prefix: {stored_prefix})")
            
            logger.warning(f"None of the {len(items)} API keys with prefix {prefix} matched the provided token")
            return None

        except Exception as e:
            logger.error(f"Failed to validate API key: {e}", exc_info=True)
            return None

    async def update_api_key(self, api_key: ApiKey) -> ApiKey:
        """Update an API key."""
        try:
            if not self.api_keys_table:
                raise RuntimeError("API keys table not initialized")
            
            item = {
                "id": str(api_key.id),
                "user_id": str(api_key.user_id),
                "name": api_key.name,
                "key_hash": api_key.key_hash,
                "key_prefix": api_key.key_prefix,
                "status": api_key.status.value,
                "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
                "scopes": json.dumps(api_key.scopes),
                "created_at": api_key.created_at.isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            
            self.api_keys_table.put_item(Item=item)
            logger.info(f"Updated API key: {api_key.id}")
            return api_key

        except Exception as e:
            logger.error(f"Failed to update API key: {e}")
            raise

    async def delete_api_key(self, api_key_id: str, user_id: str) -> bool:
        """Delete an API key. Only the owner can delete it."""
        try:
            if not self.api_keys_table:
                raise RuntimeError("API keys table not initialized")
            
            # Verify ownership
            api_key = await self.get_api_key_by_id(api_key_id)
            if not api_key:
                return False
            
            if str(api_key.user_id) != user_id:
                return False
            
            self.api_keys_table.delete_item(Key={"id": api_key_id})
            logger.info(f"Deleted API key: {api_key_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete API key {api_key_id}: {e}")
            raise


# Dependency injection
_dao_instance = None


def set_dao_instance(dao: DynamoDBDAO):
    """Set the global DAO instance."""
    global _dao_instance
    _dao_instance = dao


def get_dao() -> DynamoDBDAO:
    """Get DAO instance for dependency injection."""
    global _dao_instance
    if _dao_instance is None:
        raise RuntimeError("DAO instance not initialized. Call set_dao_instance() first.")
    return _dao_instance
