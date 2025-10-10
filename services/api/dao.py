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
from shared.schemas import (Action, ActionStatus, Conversation,
                            ConversationStatus, RoomInfo, Turn, User, Meeting)

logger = logging.getLogger(__name__)
settings = get_settings()


class DynamoDBDAO:
    """DynamoDB Data Access Object."""

    def __init__(self):
        self.dynamodb = None
        self.conversations_table = None
        self.turns_table = None
        self.actions_table = None
        self.users_table = None
        self.rooms_table = None
        self.meetings_table = None
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
            self.conversations_table = self.dynamodb.Table(
                f"{settings.dynamodb_table_prefix}{settings.conversations_table}"
            )
            self.turns_table = self.dynamodb.Table(
                f"{settings.dynamodb_table_prefix}{settings.turns_table}"
            )
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

            # Create tables if they don't exist (for development)
            # await self._create_tables_if_not_exist()

            self.initialized = True
            logger.info("DynamoDB DAO initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DynamoDB DAO: {e}")
            raise

    async def _create_tables_if_not_exist(self):
        """Create tables if they don't exist (development only)."""
        try:
            # This is a simplified version - in production, use Terraform
            tables_to_create = [
                {
                    "name": f"{settings.dynamodb_table_prefix}{settings.conversations_table}",
                    "key_schema": [{"AttributeName": "id", "KeyType": "HASH"}],
                    "attribute_definitions": [
                        {"AttributeName": "id", "AttributeType": "S"},
                        {"AttributeName": "user_id", "AttributeType": "S"},
                        {"AttributeName": "started_at", "AttributeType": "S"},
                    ],
                    "global_secondary_indexes": [
                        {
                            "IndexName": "user-id-index",
                            "KeySchema": [
                                {"AttributeName": "user_id", "KeyType": "HASH"},
                                {"AttributeName": "started_at", "KeyType": "RANGE"},
                            ],
                            "Projection": {"ProjectionType": "ALL"},
                            "ProvisionedThroughput": {
                                "ReadCapacityUnits": 5,
                                "WriteCapacityUnits": 5,
                            },
                        }
                    ],
                },
                {
                    "name": f"{settings.dynamodb_table_prefix}meetings",
                    "key_schema": [{"AttributeName": "id", "KeyType": "HASH"}],
                    "attribute_definitions": [
                        {"AttributeName": "id", "AttributeType": "S"},
                        {"AttributeName": "organizer_email", "AttributeType": "S"},
                        {"AttributeName": "start_time", "AttributeType": "S"},
                    ],
                    "global_secondary_indexes": [
                        {
                            "IndexName": "organizer-start-index",
                            "KeySchema": [
                                {"AttributeName": "organizer_email", "KeyType": "HASH"},
                                {"AttributeName": "start_time", "KeyType": "RANGE"},
                            ],
                            "Projection": {"ProjectionType": "ALL"},
                            "ProvisionedThroughput": {
                                "ReadCapacityUnits": 5,
                                "WriteCapacityUnits": 5,
                            },
                        }
                    ],
                }
            ]

            for table_config in tables_to_create:
                try:
                    self.dynamodb.create_table(**table_config)
                    logger.info(f"Created table: {table_config['name']}")
                except ClientError as e:
                    if e.response["Error"]["Code"] == "ResourceInUseException":
                        logger.info(f"Table already exists: {table_config['name']}")
                    else:
                        raise

        except Exception as e:
            logger.warning(f"Failed to create tables: {e}")

    # Conversation operations
    async def create_conversation(self, conversation: Conversation) -> Conversation:
        """Create a new conversation."""
        try:
            item = {
                "id": str(conversation.id),
                "user_id": str(conversation.user_id),
                "room_id": conversation.room_id,
                "status": conversation.status.value,
                "started_at": conversation.started_at.isoformat(),
                "ended_at": conversation.ended_at.isoformat()
                if conversation.ended_at
                else None,
                "summary": conversation.summary,
                "metadata": json.dumps(conversation.metadata),
            }

            self.conversations_table.put_item(Item=item)
            logger.info(f"Created conversation: {conversation.id}")
            return conversation

        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        try:
            response = self.conversations_table.get_item(Key={"id": conversation_id})

            if "Item" not in response:
                return None

            item = response["Item"]
            return Conversation(
                id=UUID(item["id"]),
                user_id=UUID(item["user_id"]),
                room_id=item["room_id"],
                status=ConversationStatus(item["status"]),
                started_at=datetime.fromisoformat(item["started_at"]),
                ended_at=datetime.fromisoformat(item["ended_at"])
                if item.get("ended_at")
                else None,
                summary=item.get("summary"),
                metadata=json.loads(item.get("metadata", "{}")),
            )

        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {e}")
            raise

    async def get_conversations(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Conversation]:
        """Get conversations with filtering."""
        try:
            conversations = []

            if user_id:
                # Use GSI for user_id query
                response = self.conversations_table.query(
                    IndexName="user-id-index",
                    KeyConditionExpression=Key("user_id").eq(user_id),
                    ScanIndexForward=False,  # Sort by started_at descending
                    Limit=limit,
                )
            else:
                # Scan all conversations
                scan_params = {"Limit": limit}
                if offset > 0:
                    scan_params["ExclusiveStartKey"] = {"id": str(offset)}
                
                response = self.conversations_table.scan(**scan_params)

            for item in response.get("Items", []):
                if status and item["status"] != status:
                    continue

                conversation = Conversation(
                    id=UUID(item["id"]),
                    user_id=UUID(item["user_id"]),
                    room_id=item["room_id"],
                    status=ConversationStatus(item["status"]),
                    started_at=datetime.fromisoformat(item["started_at"]),
                    ended_at=datetime.fromisoformat(item["ended_at"])
                    if item.get("ended_at")
                    else None,
                    summary=item.get("summary"),
                    metadata=json.loads(item.get("metadata", "{}")),
                )
                conversations.append(conversation)

            return conversations

        except Exception as e:
            logger.error(f"Failed to get conversations: {e}")
            raise

    async def update_conversation(self, conversation: Conversation) -> Conversation:
        """Update a conversation."""
        try:
            item = {
                "id": str(conversation.id),
                "user_id": str(conversation.user_id),
                "room_id": conversation.room_id,
                "status": conversation.status.value,
                "started_at": conversation.started_at.isoformat(),
                "ended_at": conversation.ended_at.isoformat()
                if conversation.ended_at
                else None,
                "summary": conversation.summary,
                "metadata": json.dumps(conversation.metadata),
            }

            self.conversations_table.put_item(Item=item)
            logger.info(f"Updated conversation: {conversation.id}")
            return conversation

        except Exception as e:
            logger.error(f"Failed to update conversation: {e}")
            raise

    async def delete_conversation(self, conversation_id: str):
        """Delete a conversation and related data."""
        try:
            # Delete conversation
            self.conversations_table.delete_item(Key={"id": conversation_id})

            # Delete related turns
            turns = await self.get_conversation_turns(conversation_id, limit=1000)
            for turn in turns:
                self.turns_table.delete_item(Key={"id": str(turn.id)})

            # Delete related actions
            actions = await self.get_actions(
                conversation_id=conversation_id, limit=1000
            )
            for action in actions:
                self.actions_table.delete_item(Key={"id": str(action.id)})

            logger.info(f"Deleted conversation and related data: {conversation_id}")

        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            raise

    # Turn operations
    async def get_conversation_turns(
        self, conversation_id: str, limit: int = 100, offset: int = 0
    ) -> List[Turn]:
        """Get turns for a conversation."""
        try:
            # Use scan with filter since the table schema might not support query on conversation_id
            response = self.turns_table.scan(
                FilterExpression=Attr("conversation_id").eq(conversation_id),
                Limit=limit,
            )

            turns = []
            for item in response.get("Items", []):
                try:
                    turn = Turn(
                        id=UUID(item["id"]),
                        conversation_id=UUID(item["conversation_id"]),
                        turn_number=item["turn_number"],
                        turn_type=item["turn_type"],
                        content=item["content"],
                        audio_url=item.get("audio_url"),
                        confidence_score=item.get("confidence_score"),
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        metadata=json.loads(item.get("metadata", "{}")),
                    )
                    turns.append(turn)
                except Exception as turn_error:
                    logger.warning(f"Failed to parse turn {item.get('id', 'unknown')}: {turn_error}")
                    continue

            # Sort by turn_number since scan doesn't guarantee order
            turns.sort(key=lambda t: t.turn_number)
            return turns

        except Exception as e:
            logger.error(f"Failed to get conversation turns: {e}")
            return []  # Return empty list instead of raising to avoid breaking the dashboard

    async def get_turn_count(self, conversation_id: str) -> int:
        """Get turn count for a conversation."""
        try:
            response = self.turns_table.scan(
                FilterExpression=Attr("conversation_id").eq(conversation_id),
                Select="COUNT",
            )
            return response.get("Count", 0)
        except Exception as e:
            logger.error(f"Failed to get turn count: {e}")
            return 0

    # Action operations
    async def get_actions(
        self,
        conversation_id: Optional[str] = None,
        action_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Action]:
        """Get actions with filtering."""
        try:
            # Use scan with filter since the table schema might not support query on conversation_id
            scan_params = {"Limit": limit}
            if conversation_id:
                scan_params["FilterExpression"] = Attr("conversation_id").eq(conversation_id)
            
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
                    conversation_id=UUID(item["conversation_id"]),
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
                conversation_id=UUID(item["conversation_id"]),
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
                "conversation_id": str(action.conversation_id),
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

    async def get_meetings(self, limit: int = 10) -> List[Meeting]:
        """Get meetings with pagination."""
        try:
            response = self.meetings_table.scan(Limit=limit)
            meetings = []
            
            for item in response.get("Items", []):
                meetings.append(self._item_to_meeting(item))
            
            return meetings

        except Exception as e:
            logger.error(f"Failed to get meetings: {e}")
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
        )


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
