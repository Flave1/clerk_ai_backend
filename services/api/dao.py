"""
Data Access Object for MongoDB operations.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.errors import DuplicateKeyError

from shared.config import get_settings
from shared.schemas import (
    Action,
    ActionStatus,
    RoomInfo,
    User,
    Meeting,
    MeetingStatus,
    MeetingPlatform,
    ApiKey,
    ApiKeyStatus,
    UserIntegration,
    IntegrationStatus,
    NewsletterSubscription,
    Conversation,
    ConversationStatus,
    Turn,
    TurnType,
    MeetingContext,
    MeetingRole,
    TonePersonality,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class MongoDBDAO:
    """MongoDB Data Access Object."""

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.actions_collection = None
        self.users_collection = None
        self.rooms_collection = None
        self.meetings_collection = None
        self.api_keys_collection = None
        self.user_integrations_collection = None
        self.meeting_contexts_collection = None
        self.newsletter_collection = None
        self.conversations_collection = None
        self.turns_collection = None
        self.initialized = False

    async def initialize(self):
        """Initialize MongoDB client and collections."""
        try:
            # Connect to MongoDB
            # mongodb+srv:// automatically uses TLS, and we've added tls=true to the connection string
            self.client = AsyncIOMotorClient(
                settings.mongodb_url,
                serverSelectionTimeoutMS=15000,  # Increased timeout for SSL handshake
                connectTimeoutMS=15000,
                socketTimeoutMS=30000,
            )
            
            # Get database
            self.db = self.client[settings.mongodb_database]
            
            # Test connection
            await self.client.admin.command('ping')
            
            # Get collection references
            self.actions_collection = self.db[
                f"{settings.mongodb_collection_prefix}{settings.actions_collection}"
            ]
            self.users_collection = self.db[
                f"{settings.mongodb_collection_prefix}{settings.users_collection}"
            ]
            self.rooms_collection = self.db[
                f"{settings.mongodb_collection_prefix}{settings.rooms_collection}"
            ]
            self.meetings_collection = self.db[
                f"{settings.mongodb_collection_prefix}{settings.meetings_collection}"
            ]
            self.api_keys_collection = self.db[
                f"{settings.mongodb_collection_prefix}{settings.api_keys_collection}"
            ]
            self.user_integrations_collection = self.db[
                f"{settings.mongodb_collection_prefix}{settings.user_integrations_collection}"
            ]
            self.meeting_contexts_collection = self.db[
                f"{settings.mongodb_collection_prefix}{settings.meeting_contexts_collection}"
            ]
            self.newsletter_collection = self.db[
                f"{settings.mongodb_collection_prefix}{settings.newsletter_collection}"
            ]
            self.conversations_collection = self.db[
                f"{settings.mongodb_collection_prefix}{settings.conversations_collection}"
            ]
            self.turns_collection = self.db[
                f"{settings.mongodb_collection_prefix}{settings.turns_collection}"
            ]

            # Create indexes
            await self._create_indexes()
            
            self.initialized = True
            logger.info(f"✅ MongoDB DAO initialized successfully (Database: {settings.mongodb_database})")

        except Exception as e:
            logger.error(f"❌ Failed to initialize MongoDB DAO: {e}", exc_info=True)
            raise

    async def _create_indexes(self):
        """Create indexes for collections."""
        try:
            # Actions indexes
            await self.actions_collection.create_indexes([
                IndexModel([("id", ASCENDING)], unique=True),
                IndexModel([("status", ASCENDING), ("action_type", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)]),
            ])

            # Users indexes
            await self.users_collection.create_indexes([
                IndexModel([("id", ASCENDING)], unique=True),
                IndexModel([("email", ASCENDING)], unique=True),
            ])

            # Rooms indexes
            await self.rooms_collection.create_indexes([
                IndexModel([("room_id", ASCENDING)], unique=True),
                IndexModel([("is_active", ASCENDING)]),
            ])

            # Meetings indexes
            await self.meetings_collection.create_indexes([
                IndexModel([("id", ASCENDING)], unique=True),
                IndexModel([("user_id", ASCENDING), ("start_time", DESCENDING)]),
                IndexModel([("platform", ASCENDING), ("status", ASCENDING)]),
                IndexModel([("ai_email", ASCENDING), ("start_time", DESCENDING)]),
                IndexModel([("meeting_url", ASCENDING)]),
            ])

            # API Keys indexes
            await self.api_keys_collection.create_indexes([
                IndexModel([("id", ASCENDING)], unique=True),
                IndexModel([("user_id", ASCENDING)]),
                IndexModel([("key_prefix", ASCENDING)]),
                IndexModel([("status", ASCENDING)]),
            ])

            # User Integrations indexes
            await self.user_integrations_collection.create_indexes([
                IndexModel([("id", ASCENDING)], unique=True),
                IndexModel([("user_id", ASCENDING)]),
                IndexModel([("integration_id", ASCENDING)]),
                IndexModel([("user_id", ASCENDING), ("integration_id", ASCENDING)], unique=True),
            ])

            # Newsletter indexes
            await self.newsletter_collection.create_indexes([
                IndexModel([("email", ASCENDING)], unique=True),
            ])

            # Meeting Contexts indexes
            await self.meeting_contexts_collection.create_indexes([
                IndexModel([("id", ASCENDING)], unique=True),
                IndexModel([("user_id", ASCENDING)]),
                IndexModel([("user_id", ASCENDING), ("is_default", ASCENDING)]),
            ])

            # Conversations indexes
            await self.conversations_collection.create_indexes([
                IndexModel([("id", ASCENDING)], unique=True),
                IndexModel([("user_id", ASCENDING), ("started_at", DESCENDING)]),
                IndexModel([("room_id", ASCENDING)]),
                IndexModel([("status", ASCENDING)]),
            ])

            # Turns indexes
            await self.turns_collection.create_indexes([
                IndexModel([("id", ASCENDING)], unique=True),
                IndexModel([("conversation_id", ASCENDING), ("turn_number", ASCENDING)]),
                IndexModel([("conversation_id", ASCENDING), ("timestamp", ASCENDING)]),
            ])

            logger.info("✅ MongoDB indexes created successfully")
        except Exception as e:
            logger.warning(f"⚠️  Failed to create some indexes: {e}")

    # Helper methods for document conversion
    def _action_to_doc(self, action: Action) -> dict:
        """Convert Action to MongoDB document."""
        doc = {
            "id": str(action.id),
            "turn_id": str(action.turn_id) if action.turn_id else None,
            "action_type": action.action_type.value,
            "status": action.status.value,
            "parameters": action.parameters,
            "result": action.result,
            "error_message": action.error_message,
            "created_at": action.created_at,
            "completed_at": action.completed_at,
        }
        return doc

    def _doc_to_action(self, doc: dict) -> Action:
        """Convert MongoDB document to Action."""
        return Action(
            id=UUID(doc["id"]),
            turn_id=UUID(doc["turn_id"]) if doc.get("turn_id") else None,
            action_type=doc["action_type"],
            status=ActionStatus(doc["status"]),
            parameters=doc.get("parameters", {}),
            result=doc.get("result"),
            error_message=doc.get("error_message"),
            created_at=doc["created_at"] if isinstance(doc["created_at"], datetime) else datetime.fromisoformat(doc["created_at"]),
            completed_at=doc["completed_at"] if isinstance(doc["completed_at"], datetime) else (datetime.fromisoformat(doc["completed_at"]) if doc.get("completed_at") else None),
        )

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
            query = {}
            if action_type:
                query["action_type"] = action_type
            if status:
                query["status"] = status
            
            cursor = self.actions_collection.find(query).sort("created_at", DESCENDING).skip(offset).limit(limit)
            actions = []
            async for doc in cursor:
                actions.append(self._doc_to_action(doc))
            return actions
        except Exception as e:
            logger.error(f"Failed to get actions: {e}")
            raise

    async def get_action(self, action_id: str) -> Optional[Action]:
        """Get an action by ID."""
        try:
            doc = await self.actions_collection.find_one({"id": action_id})
            if not doc:
                return None
            return self._doc_to_action(doc)
        except Exception as e:
            logger.error(f"Failed to get action {action_id}: {e}")
            raise

    async def update_action(self, action: Action) -> Action:
        """Update an action."""
        try:
            doc = self._action_to_doc(action)
            await self.actions_collection.update_one(
                {"id": str(action.id)},
                {"$set": doc},
                upsert=True
            )
            logger.info(f"Updated action: {action.id}")
            return action
        except Exception as e:
            logger.error(f"Failed to update action: {e}")
            raise

    # Newsletter operations
    async def get_newsletter_signup(self, email: str) -> Optional[NewsletterSubscription]:
        """Retrieve a newsletter subscription by email."""
        try:
            doc = await self.newsletter_collection.find_one({"email": email.lower()})
            if not doc:
                return None
            return NewsletterSubscription(
                email=doc["email"],
                name=doc.get("name", ""),
                country=doc.get("country", ""),
                signed_up_at=doc["signed_up_at"] if isinstance(doc["signed_up_at"], datetime) else datetime.fromisoformat(doc["signed_up_at"]),
            )
        except Exception as e:
            logger.error(f"Failed to get newsletter signup for {email}: {e}")
            raise

    async def add_newsletter_signup(self, signup: NewsletterSubscription) -> NewsletterSubscription:
        """Create a new newsletter subscription."""
        try:
            doc = {
                "email": signup.email.lower(),
                "name": signup.name,
                "country": signup.country,
                "signed_up_at": signup.signed_up_at,
            }
            await self.newsletter_collection.insert_one(doc)
            return signup
        except DuplicateKeyError:
            logger.info("Newsletter signup already exists for email %s", signup.email)
            raise ValueError("Email already exists in newsletter")
        except Exception as e:
            logger.error(f"Failed to add newsletter signup: {e}")
            raise

    async def list_newsletter_signups(self) -> List[NewsletterSubscription]:
        """List all newsletter subscriptions."""
        try:
            cursor = self.newsletter_collection.find().sort("signed_up_at", DESCENDING)
            signups = []
            async for doc in cursor:
                signups.append(NewsletterSubscription(
                    email=doc["email"],
                    name=doc.get("name", ""),
                    country=doc.get("country", ""),
                    signed_up_at=doc["signed_up_at"] if isinstance(doc["signed_up_at"], datetime) else datetime.fromisoformat(doc["signed_up_at"]),
                ))
            return signups
        except Exception as e:
            logger.error(f"Failed to list newsletter signups: {e}")
            raise

    # Room operations
    async def get_active_rooms(self) -> List[RoomInfo]:
        """Get list of active rooms."""
        try:
            cursor = self.rooms_collection.find({"is_active": True})
            rooms = []
            async for doc in cursor:
                rooms.append(RoomInfo(
                    room_id=doc["room_id"],
                    name=doc["name"],
                    participant_count=doc["participant_count"],
                    is_active=doc["is_active"],
                    created_at=doc["created_at"] if isinstance(doc["created_at"], datetime) else datetime.fromisoformat(doc["created_at"]),
                ))
            return rooms
        except Exception as e:
            logger.error(f"Failed to get active rooms: {e}")
            return []

    async def get_room(self, room_id: str) -> Optional[RoomInfo]:
        """Get a room by ID."""
        try:
            doc = await self.rooms_collection.find_one({"room_id": room_id})
            if not doc:
                return None
            return RoomInfo(
                room_id=doc["room_id"],
                name=doc["name"],
                participant_count=doc["participant_count"],
                is_active=doc["is_active"],
                created_at=doc["created_at"] if isinstance(doc["created_at"], datetime) else datetime.fromisoformat(doc["created_at"]),
            )
        except Exception as e:
            logger.error(f"Failed to get room {room_id}: {e}")
            raise

    async def create_room(self, room: RoomInfo) -> RoomInfo:
        """Create a new room."""
        try:
            doc = {
                "room_id": room.room_id,
                "name": room.name,
                "participant_count": room.participant_count,
                "is_active": room.is_active,
                "created_at": room.created_at,
            }
            await self.rooms_collection.insert_one(doc)
            logger.info(f"Created room: {room.room_id}")
            return room
        except Exception as e:
            logger.error(f"Failed to create room: {e}")
            raise

    async def update_room(self, room: RoomInfo) -> RoomInfo:
        """Update a room."""
        try:
            doc = {
                "room_id": room.room_id,
                "name": room.name,
                "participant_count": room.participant_count,
                "is_active": room.is_active,
                "created_at": room.created_at,
            }
            await self.rooms_collection.update_one(
                {"room_id": room.room_id},
                {"$set": doc},
                upsert=True
            )
            logger.info(f"Updated room: {room.room_id}")
            return room
        except Exception as e:
            logger.error(f"Failed to update room: {e}")
            raise

    async def delete_room(self, room_id: str):
        """Delete a room."""
        try:
            await self.rooms_collection.delete_one({"room_id": room_id})
            logger.info(f"Deleted room: {room_id}")
        except Exception as e:
            logger.error(f"Failed to delete room: {e}")
            raise

    # User operations
    async def create_user(self, user: User) -> User:
        """Create a new user."""
        try:
            # Check if user with this email already exists
            existing_user = await self.get_user_by_email(user.email)
            if existing_user:
                raise ValueError(f"User with email {user.email} already exists")
            
            doc = {
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "phone": user.phone,
                "password_hash": user.password_hash,
                "auth_provider": user.auth_provider,
                "timezone": user.timezone,
                "is_active": user.is_active,
                "created_at": user.created_at,
                "updated_at": user.updated_at,
            }
            await self.users_collection.insert_one(doc)
            logger.info(f"Created user: {user.id} ({user.email})")
            return user
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        try:
            doc = await self.users_collection.find_one({"id": user_id})
            if not doc:
                return None
            return User(
                id=UUID(doc["id"]),
                email=doc["email"],
                name=doc["name"],
                phone=doc.get("phone"),
                password_hash=doc.get("password_hash"),
                auth_provider=doc.get("auth_provider"),
                timezone=doc.get("timezone", "UTC"),
                is_active=doc.get("is_active", True),
                created_at=doc["created_at"] if isinstance(doc["created_at"], datetime) else datetime.fromisoformat(doc["created_at"]),
                updated_at=doc["updated_at"] if isinstance(doc["updated_at"], datetime) else datetime.fromisoformat(doc["updated_at"]),
            )
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            raise

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        try:
            doc = await self.users_collection.find_one({"email": email})
            if not doc:
                return None
            return User(
                id=UUID(doc["id"]),
                email=doc["email"],
                name=doc["name"],
                phone=doc.get("phone"),
                password_hash=doc.get("password_hash"),
                auth_provider=doc.get("auth_provider"),
                timezone=doc.get("timezone", "UTC"),
                is_active=doc.get("is_active", True),
                created_at=doc["created_at"] if isinstance(doc["created_at"], datetime) else datetime.fromisoformat(doc["created_at"]),
                updated_at=doc["updated_at"] if isinstance(doc["updated_at"], datetime) else datetime.fromisoformat(doc["updated_at"]),
            )
        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            raise

    async def update_user(self, user: User) -> User:
        """Update a user."""
        try:
            # Check if user exists
            existing = await self.get_user_by_id(str(user.id))
            if not existing:
                raise ValueError(f"User {user.id} not found")
            
            doc = {
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "phone": user.phone,
                "password_hash": user.password_hash,
                "auth_provider": user.auth_provider,
                "timezone": user.timezone,
                "is_active": user.is_active,
                "created_at": existing.created_at,  # Preserve original
                "updated_at": datetime.now(timezone.utc),
            }
            await self.users_collection.update_one(
                {"id": str(user.id)},
                {"$set": doc},
                upsert=True
            )
            logger.info(f"Updated user: {user.id}")
            return user
        except Exception as e:
            logger.error(f"Failed to update user: {e}")
            raise

    # API Key operations
    async def create_api_key(self, api_key: ApiKey) -> ApiKey:
        """Create a new API key."""
        try:
            doc = {
                "id": str(api_key.id),
                "user_id": str(api_key.user_id),
                "name": api_key.name,
                "key_hash": api_key.key_hash,
                "key_prefix": api_key.key_prefix,
                "status": api_key.status.value,
                "last_used_at": api_key.last_used_at,
                "expires_at": api_key.expires_at,
                "scopes": api_key.scopes,
                "created_at": api_key.created_at,
                "updated_at": api_key.updated_at,
            }
            await self.api_keys_collection.insert_one(doc)
            logger.info(f"Created API key: {api_key.id} for user {api_key.user_id}")
            return api_key
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise

    async def get_api_keys_by_user(self, user_id: str) -> List[ApiKey]:
        """Get all API keys for a user."""
        try:
            cursor = self.api_keys_collection.find({"user_id": user_id}).sort("created_at", DESCENDING)
            api_keys = []
            async for doc in cursor:
                api_keys.append(ApiKey(
                    id=UUID(doc["id"]),
                    user_id=UUID(doc["user_id"]),
                    name=doc["name"],
                    key_hash=doc["key_hash"],
                    key_prefix=doc["key_prefix"],
                    status=ApiKeyStatus(doc["status"]),
                    last_used_at=doc["last_used_at"] if isinstance(doc.get("last_used_at"), datetime) else (datetime.fromisoformat(doc["last_used_at"]) if doc.get("last_used_at") else None),
                    expires_at=doc["expires_at"] if isinstance(doc.get("expires_at"), datetime) else (datetime.fromisoformat(doc["expires_at"]) if doc.get("expires_at") else None),
                    scopes=doc.get("scopes", []),
                    created_at=doc["created_at"] if isinstance(doc["created_at"], datetime) else datetime.fromisoformat(doc["created_at"]),
                    updated_at=doc["updated_at"] if isinstance(doc["updated_at"], datetime) else datetime.fromisoformat(doc["updated_at"]),
                ))
            return api_keys
        except Exception as e:
            logger.error(f"Failed to get API keys for user {user_id}: {e}")
            raise

    async def get_api_key_by_id(self, api_key_id: str) -> Optional[ApiKey]:
        """Get an API key by ID."""
        try:
            doc = await self.api_keys_collection.find_one({"id": api_key_id})
            if not doc:
                return None
            return ApiKey(
                id=UUID(doc["id"]),
                user_id=UUID(doc["user_id"]),
                name=doc["name"],
                key_hash=doc["key_hash"],
                key_prefix=doc["key_prefix"],
                status=ApiKeyStatus(doc["status"]),
                last_used_at=doc["last_used_at"] if isinstance(doc.get("last_used_at"), datetime) else (datetime.fromisoformat(doc["last_used_at"]) if doc.get("last_used_at") else None),
                expires_at=doc["expires_at"] if isinstance(doc.get("expires_at"), datetime) else (datetime.fromisoformat(doc["expires_at"]) if doc.get("expires_at") else None),
                scopes=doc.get("scopes", []),
                created_at=doc["created_at"] if isinstance(doc["created_at"], datetime) else datetime.fromisoformat(doc["created_at"]),
                updated_at=doc["updated_at"] if isinstance(doc["updated_at"], datetime) else datetime.fromisoformat(doc["updated_at"]),
            )
        except Exception as e:
            logger.error(f"Failed to get API key {api_key_id}: {e}")
            raise

    async def validate_api_key(self, api_key_token: str) -> Optional[ApiKey]:
        """Validate an API key token and return the ApiKey if valid."""
        try:
            # Extract prefix from token (e.g., "sk_live_12345678...")
            if not api_key_token.startswith("sk_"):
                logger.debug("API key token doesn't start with 'sk_'")
                return None
            
            # Try to extract prefix (first 12 chars: "sk_live_1234")
            prefix = api_key_token[:12]
            logger.info(f"Looking for API key with prefix: {prefix}")
            
            # Find keys with matching prefix
            cursor = self.api_keys_collection.find({"key_prefix": prefix})
            items = []
            async for doc in cursor:
                items.append(doc)
            
            logger.info(f"Found {len(items)} API keys with prefix {prefix}")
            
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
                        last_used_at=item["last_used_at"] if isinstance(item.get("last_used_at"), datetime) else (datetime.fromisoformat(item["last_used_at"]) if item.get("last_used_at") else None),
                        expires_at=item["expires_at"] if isinstance(item.get("expires_at"), datetime) else (datetime.fromisoformat(item["expires_at"]) if item.get("expires_at") else None),
                        scopes=item.get("scopes", []),
                        created_at=item["created_at"] if isinstance(item["created_at"], datetime) else datetime.fromisoformat(item["created_at"]),
                        updated_at=item["updated_at"] if isinstance(item["updated_at"], datetime) else datetime.fromisoformat(item["updated_at"]),
                    )
                    
                    # Check if key is active and not expired
                    if api_key.status != ApiKeyStatus.ACTIVE:
                        logger.warning(f"API key {key_id} is not active (status: {api_key.status})")
                        return None
                    
                    if api_key.expires_at and api_key.expires_at < datetime.now(timezone.utc):
                        logger.warning(f"API key {key_id} is expired")
                        api_key.status = ApiKeyStatus.EXPIRED
                        await self.update_api_key(api_key)
                        return None
                    
                    logger.info(f"API key {key_id} validated successfully")
                    # Update last_used_at
                    api_key.last_used_at = datetime.now(timezone.utc)
                    await self.update_api_key(api_key)
                    
                    return api_key
                else:
                    logger.warning(f"Password verification failed for key {key_id}")
            
            logger.warning(f"None of the {len(items)} API keys with prefix {prefix} matched the provided token")
            return None

        except Exception as e:
            logger.error(f"Failed to validate API key: {e}", exc_info=True)
            return None

    async def update_api_key(self, api_key: ApiKey) -> ApiKey:
        """Update an API key."""
        try:
            doc = {
                "id": str(api_key.id),
                "user_id": str(api_key.user_id),
                "name": api_key.name,
                "key_hash": api_key.key_hash,
                "key_prefix": api_key.key_prefix,
                "status": api_key.status.value,
                "last_used_at": api_key.last_used_at,
                "expires_at": api_key.expires_at,
                "scopes": api_key.scopes,
                "created_at": api_key.created_at,
                "updated_at": datetime.now(timezone.utc),
            }
            await self.api_keys_collection.update_one(
                {"id": str(api_key.id)},
                {"$set": doc},
                upsert=True
            )
            logger.info(f"Updated API key: {api_key.id}")
            return api_key
        except Exception as e:
            logger.error(f"Failed to update API key: {e}")
            raise

    async def delete_api_key(self, api_key_id: str, user_id: str) -> bool:
        """Delete an API key. Only the owner can delete it."""
        try:
            # Verify ownership
            api_key = await self.get_api_key_by_id(api_key_id)
            if not api_key:
                return False
            
            if str(api_key.user_id) != user_id:
                return False
            
            await self.api_keys_collection.delete_one({"id": api_key_id})
            logger.info(f"Deleted API key: {api_key_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete API key {api_key_id}: {e}")
            raise

    # User Integration operations
    async def create_user_integration(self, user_integration: UserIntegration) -> UserIntegration:
        """Create a new user integration."""
        try:
            doc = {
                "id": str(user_integration.id),
                "user_id": str(user_integration.user_id),
                "integration_id": user_integration.integration_id,
                "status": user_integration.status.value,
                "access_token": user_integration.access_token,
                "refresh_token": user_integration.refresh_token,
                "token_type": user_integration.token_type,
                "expires_at": user_integration.expires_at,
                "scope": user_integration.scope,
                "connected_at": user_integration.connected_at,
                "last_used_at": user_integration.last_used_at,
                "last_refreshed_at": user_integration.last_refreshed_at,
                "error_message": user_integration.error_message,
                "metadata": user_integration.metadata,
                "created_at": user_integration.created_at,
                "updated_at": user_integration.updated_at,
            }
            await self.user_integrations_collection.insert_one(doc)
            logger.info(f"Created user integration: {user_integration.id} ({user_integration.integration_id})")
            return user_integration
        except Exception as e:
            logger.error(f"Failed to create user integration: {e}")
            raise

    async def get_user_integration(self, user_id: str, integration_id: str) -> Optional[UserIntegration]:
        """Get a user integration by user_id and integration_id."""
        try:
            doc = await self.user_integrations_collection.find_one({
                "user_id": user_id,
                "integration_id": integration_id
            })
            if not doc:
                return None
            return UserIntegration(
                id=UUID(doc["id"]),
                user_id=UUID(doc["user_id"]),
                integration_id=doc["integration_id"],
                status=IntegrationStatus(doc["status"]),
                access_token=doc.get("access_token"),
                refresh_token=doc.get("refresh_token"),
                token_type=doc.get("token_type", "Bearer"),
                expires_at=doc["expires_at"] if isinstance(doc.get("expires_at"), datetime) else (datetime.fromisoformat(doc["expires_at"]) if doc.get("expires_at") else None),
                scope=doc.get("scope"),
                connected_at=doc["connected_at"] if isinstance(doc.get("connected_at"), datetime) else (datetime.fromisoformat(doc["connected_at"]) if doc.get("connected_at") else None),
                last_used_at=doc["last_used_at"] if isinstance(doc.get("last_used_at"), datetime) else (datetime.fromisoformat(doc["last_used_at"]) if doc.get("last_used_at") else None),
                last_refreshed_at=doc["last_refreshed_at"] if isinstance(doc.get("last_refreshed_at"), datetime) else (datetime.fromisoformat(doc["last_refreshed_at"]) if doc.get("last_refreshed_at") else None),
                error_message=doc.get("error_message"),
                metadata=doc.get("metadata", {}),
                created_at=doc["created_at"] if isinstance(doc["created_at"], datetime) else datetime.fromisoformat(doc["created_at"]),
                updated_at=doc["updated_at"] if isinstance(doc["updated_at"], datetime) else datetime.fromisoformat(doc["updated_at"]),
            )
        except Exception as e:
            logger.error(f"Failed to get user integration: {e}")
            return None

    async def get_user_integrations_by_user(self, user_id: str) -> List[UserIntegration]:
        """Get all integrations for a user."""
        try:
            cursor = self.user_integrations_collection.find({"user_id": user_id})
            integrations = []
            async for doc in cursor:
                integrations.append(UserIntegration(
                    id=UUID(doc["id"]),
                    user_id=UUID(doc["user_id"]),
                    integration_id=doc["integration_id"],
                    status=IntegrationStatus(doc["status"]),
                    access_token=doc.get("access_token"),
                    refresh_token=doc.get("refresh_token"),
                    token_type=doc.get("token_type", "Bearer"),
                    expires_at=doc["expires_at"] if isinstance(doc.get("expires_at"), datetime) else (datetime.fromisoformat(doc["expires_at"]) if doc.get("expires_at") else None),
                    scope=doc.get("scope"),
                    connected_at=doc["connected_at"] if isinstance(doc.get("connected_at"), datetime) else (datetime.fromisoformat(doc["connected_at"]) if doc.get("connected_at") else None),
                    last_used_at=doc["last_used_at"] if isinstance(doc.get("last_used_at"), datetime) else (datetime.fromisoformat(doc["last_used_at"]) if doc.get("last_used_at") else None),
                    last_refreshed_at=doc["last_refreshed_at"] if isinstance(doc.get("last_refreshed_at"), datetime) else (datetime.fromisoformat(doc["last_refreshed_at"]) if doc.get("last_refreshed_at") else None),
                    error_message=doc.get("error_message"),
                    metadata=doc.get("metadata", {}),
                    created_at=doc["created_at"] if isinstance(doc["created_at"], datetime) else datetime.fromisoformat(doc["created_at"]),
                    updated_at=doc["updated_at"] if isinstance(doc["updated_at"], datetime) else datetime.fromisoformat(doc["updated_at"]),
                ))
            return integrations
        except Exception as e:
            logger.error(f"Failed to get user integrations: {e}")
            return []

    async def update_user_integration(self, user_integration: UserIntegration) -> UserIntegration:
        """Update a user integration."""
        try:
            doc = {
                "id": str(user_integration.id),
                "user_id": str(user_integration.user_id),
                "integration_id": user_integration.integration_id,
                "status": user_integration.status.value,
                "access_token": user_integration.access_token,
                "refresh_token": user_integration.refresh_token,
                "token_type": user_integration.token_type,
                "expires_at": user_integration.expires_at,
                "scope": user_integration.scope,
                "connected_at": user_integration.connected_at,
                "last_used_at": user_integration.last_used_at,
                "last_refreshed_at": user_integration.last_refreshed_at,
                "error_message": user_integration.error_message,
                "metadata": user_integration.metadata,
                "created_at": user_integration.created_at,
                "updated_at": datetime.now(timezone.utc),
            }
            await self.user_integrations_collection.update_one(
                {"id": str(user_integration.id)},
                {"$set": doc},
                upsert=True
            )
            logger.info(f"Updated user integration: {user_integration.id}")
            return user_integration
        except Exception as e:
            logger.error(f"Failed to update user integration: {e}")
            raise

    async def delete_user_integration(self, integration_id: str, user_id: str) -> bool:
        """Delete a user integration."""
        try:
            # First, get the integration to find its ID
            user_integration = await self.get_user_integration(user_id, integration_id)
            if not user_integration:
                return False
            
            await self.user_integrations_collection.delete_one({"id": str(user_integration.id)})
            logger.info(f"Deleted user integration: {user_integration.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete user integration: {e}")
            return False

    # Conversation operations
    async def get_conversations(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Conversation]:
        """Get conversations for a user."""
        try:
            query = {"user_id": str(user_id)}
            if status:
                query["status"] = status
            
            cursor = self.conversations_collection.find(query).sort("started_at", DESCENDING).skip(offset).limit(limit)
            conversations = []
            async for doc in cursor:
                conversations.append(Conversation(
                    id=UUID(doc["id"]),
                    user_id=UUID(doc["user_id"]),
                    room_id=doc["room_id"],
                    status=ConversationStatus(doc["status"]),
                    started_at=doc["started_at"] if isinstance(doc["started_at"], datetime) else datetime.fromisoformat(doc["started_at"]),
                    ended_at=doc["ended_at"] if isinstance(doc.get("ended_at"), datetime) else (datetime.fromisoformat(doc["ended_at"]) if doc.get("ended_at") else None),
                    summary=doc.get("summary"),
                    metadata=doc.get("metadata", {}),
                ))
            return conversations
        except Exception as e:
            logger.error(f"Failed to get conversations: {e}")
            raise

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        try:
            doc = await self.conversations_collection.find_one({"id": conversation_id})
            if not doc:
                return None
            return Conversation(
                id=UUID(doc["id"]),
                user_id=UUID(doc["user_id"]),
                room_id=doc["room_id"],
                status=ConversationStatus(doc["status"]),
                started_at=doc["started_at"] if isinstance(doc["started_at"], datetime) else datetime.fromisoformat(doc["started_at"]),
                ended_at=doc["ended_at"] if isinstance(doc.get("ended_at"), datetime) else (datetime.fromisoformat(doc["ended_at"]) if doc.get("ended_at") else None),
                summary=doc.get("summary"),
                metadata=doc.get("metadata", {}),
            )
        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {e}")
            raise

    async def create_conversation(self, conversation: Conversation) -> Conversation:
        """Create a new conversation."""
        try:
            doc = {
                "id": str(conversation.id),
                "user_id": str(conversation.user_id),
                "room_id": conversation.room_id,
                "status": conversation.status.value,
                "started_at": conversation.started_at,
                "ended_at": conversation.ended_at,
                "summary": conversation.summary,
                "metadata": conversation.metadata,
            }
            await self.conversations_collection.insert_one(doc)
            logger.info(f"Created conversation: {conversation.id}")
            return conversation
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise

    async def update_conversation(self, conversation: Conversation) -> Conversation:
        """Update a conversation."""
        try:
            doc = {
                "id": str(conversation.id),
                "user_id": str(conversation.user_id),
                "room_id": conversation.room_id,
                "status": conversation.status.value,
                "started_at": conversation.started_at,
                "ended_at": conversation.ended_at,
                "summary": conversation.summary,
                "metadata": conversation.metadata,
            }
            await self.conversations_collection.update_one(
                {"id": str(conversation.id)},
                {"$set": doc},
                upsert=True
            )
            logger.info(f"Updated conversation: {conversation.id}")
            return conversation
        except Exception as e:
            logger.error(f"Failed to update conversation: {e}")
            raise

    async def delete_conversation(self, conversation_id: str):
        """Delete a conversation and all related turns."""
        try:
            # Delete conversation
            await self.conversations_collection.delete_one({"id": conversation_id})
            # Delete related turns
            await self.turns_collection.delete_many({"conversation_id": conversation_id})
            logger.info(f"Deleted conversation: {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            raise

    async def get_conversation_turns(
        self,
        conversation_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Turn]:
        """Get turns for a conversation."""
        try:
            cursor = self.turns_collection.find({"conversation_id": conversation_id}).sort("turn_number", ASCENDING).skip(offset).limit(limit)
            turns = []
            async for doc in cursor:
                turns.append(Turn(
                    id=UUID(doc["id"]),
                    conversation_id=UUID(doc["conversation_id"]),
                    turn_number=doc["turn_number"],
                    turn_type=TurnType(doc["turn_type"]),
                    content=doc["content"],
                    audio_url=doc.get("audio_url"),
                    confidence_score=doc.get("confidence_score"),
                    timestamp=doc["timestamp"] if isinstance(doc["timestamp"], datetime) else datetime.fromisoformat(doc["timestamp"]),
                    metadata=doc.get("metadata", {}),
                ))
            return turns
        except Exception as e:
            logger.error(f"Failed to get conversation turns: {e}")
            raise

    async def get_turn_count(self, conversation_id: str) -> int:
        """Get count of turns for a conversation."""
        try:
            count = await self.turns_collection.count_documents({"conversation_id": conversation_id})
            return count
        except Exception as e:
            logger.error(f"Failed to get turn count: {e}")
            return 0

    # Meeting operations
    async def get_meeting(self, meeting_id: str) -> Optional[Meeting]:
        """Get a meeting by ID."""
        try:
            doc = await self.meetings_collection.find_one({"id": meeting_id})
            if not doc:
                return None
            
            # Convert participants
            participants = []
            for p in doc.get("participants", []):
                from shared.schemas import MeetingParticipant
                participants.append(MeetingParticipant(
                    email=p["email"],
                    name=p.get("name"),
                    is_organizer=p.get("is_organizer", False),
                    response_status=p.get("response_status", "accepted"),
                ))
            
            # Convert summary if exists
            summary = None
            if doc.get("summary"):
                from shared.schemas import MeetingSummary, ActionItem
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
        except Exception as e:
            logger.error(f"Failed to get meeting {meeting_id}: {e}")
            raise

    async def get_meetings(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Meeting]:
        """Get meetings with filtering."""
        try:
            query = {}
            if user_id:
                # Ensure user_id is a string for querying
                query["user_id"] = str(user_id)
            if status:
                # Handle both enum and string status
                if hasattr(status, 'value'):
                    query["status"] = status.value
                else:
                    query["status"] = str(status)
            
            # Use sort with fallback for missing start_time field
            # MongoDB will put documents without the field at the end
            cursor = self.meetings_collection.find(query).sort("start_time", DESCENDING).limit(limit)
            meetings = []
            async for doc in cursor:
                try:
                    meeting = await self._doc_to_meeting(doc)
                    if meeting:
                        meetings.append(meeting)
                except Exception as e:
                    logger.warning(f"Failed to convert document to meeting, skipping: {e}")
                    continue
            return meetings
        except Exception as e:
            logger.error(f"Failed to get meetings: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            raise

    async def _doc_to_meeting(self, doc: dict) -> Optional[Meeting]:
        """Convert MongoDB document to Meeting."""
        try:
            # Convert participants
            participants = []
            for p in doc.get("participants", []):
                from shared.schemas import MeetingParticipant
                if not p.get("email"):
                    continue  # Skip invalid participants
                participants.append(MeetingParticipant(
                    email=p["email"],
                    name=p.get("name"),
                    is_organizer=p.get("is_organizer", False),
                    response_status=p.get("response_status", "accepted"),
                ))
            
            # Convert summary if exists
            summary = None
            if doc.get("summary"):
                from shared.schemas import MeetingSummary, ActionItem
                s = doc["summary"]
                if s and isinstance(s, dict) and s.get("id") and s.get("meeting_id"):
                    action_items = []
                    for ai in s.get("action_items", []):
                        if ai.get("id") and ai.get("description"):
                            try:
                                action_items.append(ActionItem(
                                    id=UUID(ai["id"]),
                                    description=ai["description"],
                                    assignee=ai.get("assignee"),
                                    due_date=datetime.fromisoformat(ai["due_date"]) if ai.get("due_date") else None,
                                    priority=ai.get("priority", "medium"),
                                    status=ai.get("status", "pending"),
                                ))
                            except (ValueError, KeyError) as e:
                                logger.warning(f"Failed to convert action item: {e}")
                                continue
                    try:
                        summary = MeetingSummary(
                            id=UUID(s["id"]),
                            meeting_id=UUID(s["meeting_id"]),
                            topics_discussed=s.get("topics_discussed", []),
                            key_decisions=s.get("key_decisions", []),
                            action_items=action_items,
                            summary_text=s.get("summary_text", ""),
                            sentiment=s.get("sentiment"),
                            duration_minutes=s.get("duration_minutes"),
                            created_at=datetime.fromisoformat(s["created_at"]) if isinstance(s.get("created_at"), str) else (s.get("created_at") or datetime.now(timezone.utc)),
                        )
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Failed to convert summary: {e}")
                        summary = None
            
            # Helper to safely convert datetime
            def safe_datetime(value):
                if value is None:
                    return None
                if isinstance(value, datetime):
                    return value
                if isinstance(value, str):
                    try:
                        return datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        return None
                return None
            
            # Validate required fields
            if not doc.get("id"):
                logger.warning("Document missing required field 'id'")
                return None
            
            start_time = safe_datetime(doc.get("start_time")) or datetime.now(timezone.utc)
            end_time = safe_datetime(doc.get("end_time")) or datetime.now(timezone.utc)
            
            return Meeting(
                id=UUID(doc["id"]),
                user_id=UUID(doc["user_id"]) if doc.get("user_id") else None,
                platform=MeetingPlatform(doc["platform"]) if isinstance(doc.get("platform"), str) else doc.get("platform"),
                meeting_url=doc.get("meeting_url", ""),
                meeting_id_external=doc.get("meeting_id_external", str(doc["id"])),
                title=doc.get("title", "Meeting"),
                description=doc.get("description"),
                start_time=start_time,
                end_time=end_time,
                organizer_email=doc.get("organizer_email", ""),
                participants=participants,
                status=MeetingStatus(doc["status"]) if isinstance(doc.get("status"), str) else doc.get("status"),
                ai_email=doc.get("ai_email", ""),
                transcription_chunks=doc.get("transcription_chunks", []),
                full_transcription=doc.get("full_transcription"),
                summary=summary,
                calendar_event_id=doc.get("calendar_event_id"),
                join_attempts=doc.get("join_attempts", 0),
                last_join_attempt=safe_datetime(doc.get("last_join_attempt")),
                error_message=doc.get("error_message"),
                bot_joined=doc.get("bot_joined", False),
                created_at=safe_datetime(doc.get("created_at")) or datetime.now(timezone.utc),
                updated_at=safe_datetime(doc.get("updated_at")) or datetime.now(timezone.utc),
                joined_at=safe_datetime(doc.get("joined_at")),
                ended_at=safe_datetime(doc.get("ended_at")),
                audio_enabled=doc.get("audio_enabled", True),
                video_enabled=doc.get("video_enabled", False),
                recording_enabled=doc.get("recording_enabled", False),
                transcript=doc.get("transcript", False),
                voice_id=doc.get("voice_id"),
                bot_name=doc.get("bot_name"),
                context_id=doc.get("context_id"),
            )
        except Exception as e:
            logger.error(f"Failed to convert document to meeting: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def create_meeting(self, meeting: Meeting) -> Meeting:
        """Create a new meeting."""
        try:
            doc = {
                "id": str(meeting.id),
                "user_id": str(meeting.user_id) if meeting.user_id else None,
                "platform": meeting.platform.value if hasattr(meeting.platform, 'value') else meeting.platform,
                "meeting_url": meeting.meeting_url,
                "meeting_id_external": meeting.meeting_id_external,
                "title": meeting.title,
                "description": meeting.description,
                "start_time": meeting.start_time,
                "end_time": meeting.end_time,
                "organizer_email": meeting.organizer_email,
                "participants": [{
                    "email": p.email,
                    "name": p.name,
                    "is_organizer": p.is_organizer,
                    "response_status": p.response_status,
                } for p in meeting.participants],
                "status": meeting.status.value if hasattr(meeting.status, 'value') else meeting.status,
                "ai_email": meeting.ai_email,
                "transcription_chunks": meeting.transcription_chunks,
                "full_transcription": meeting.full_transcription,
                "summary": {
                    "id": str(meeting.summary.id),
                    "meeting_id": str(meeting.summary.meeting_id),
                    "topics_discussed": meeting.summary.topics_discussed,
                    "key_decisions": meeting.summary.key_decisions,
                    "action_items": [{
                        "id": str(ai.id),
                        "description": ai.description,
                        "assignee": ai.assignee,
                        "due_date": ai.due_date.isoformat() if ai.due_date else None,
                        "priority": ai.priority,
                        "status": ai.status,
                    } for ai in meeting.summary.action_items] if meeting.summary else None,
                    "summary_text": meeting.summary.summary_text if meeting.summary else None,
                    "sentiment": meeting.summary.sentiment if meeting.summary else None,
                    "duration_minutes": meeting.summary.duration_minutes if meeting.summary else None,
                    "created_at": meeting.summary.created_at.isoformat() if meeting.summary else None,
                } if meeting.summary else None,
                "calendar_event_id": meeting.calendar_event_id,
                "join_attempts": meeting.join_attempts,
                "last_join_attempt": meeting.last_join_attempt,
                "error_message": meeting.error_message,
                "bot_joined": meeting.bot_joined,
                "created_at": meeting.created_at,
                "updated_at": meeting.updated_at,
                "joined_at": meeting.joined_at,
                "ended_at": meeting.ended_at,
                "audio_enabled": meeting.audio_enabled,
                "video_enabled": meeting.video_enabled,
                "recording_enabled": meeting.recording_enabled,
                "transcript": meeting.transcript,
                "voice_id": meeting.voice_id,
                "bot_name": meeting.bot_name,
                "context_id": meeting.context_id,
            }
            await self.meetings_collection.insert_one(doc)
            logger.info(f"Created meeting: {meeting.id}")
            return meeting
        except Exception as e:
            logger.error(f"Failed to create meeting: {e}")
            raise

    async def update_meeting(self, meeting: Meeting) -> Meeting:
        """Update a meeting."""
        try:
            doc = {
                "id": str(meeting.id),
                "user_id": str(meeting.user_id) if meeting.user_id else None,
                "platform": meeting.platform.value if hasattr(meeting.platform, 'value') else meeting.platform,
                "meeting_url": meeting.meeting_url,
                "meeting_id_external": meeting.meeting_id_external,
                "title": meeting.title,
                "description": meeting.description,
                "start_time": meeting.start_time,
                "end_time": meeting.end_time,
                "organizer_email": meeting.organizer_email,
                "participants": [{
                    "email": p.email,
                    "name": p.name,
                    "is_organizer": p.is_organizer,
                    "response_status": p.response_status,
                } for p in meeting.participants],
                "status": meeting.status.value if hasattr(meeting.status, 'value') else meeting.status,
                "ai_email": meeting.ai_email,
                "transcription_chunks": meeting.transcription_chunks,
                "full_transcription": meeting.full_transcription,
                "summary": {
                    "id": str(meeting.summary.id),
                    "meeting_id": str(meeting.summary.meeting_id),
                    "topics_discussed": meeting.summary.topics_discussed,
                    "key_decisions": meeting.summary.key_decisions,
                    "action_items": [{
                        "id": str(ai.id),
                        "description": ai.description,
                        "assignee": ai.assignee,
                        "due_date": ai.due_date.isoformat() if ai.due_date else None,
                        "priority": ai.priority,
                        "status": ai.status,
                    } for ai in meeting.summary.action_items] if meeting.summary else None,
                    "summary_text": meeting.summary.summary_text if meeting.summary else None,
                    "sentiment": meeting.summary.sentiment if meeting.summary else None,
                    "duration_minutes": meeting.summary.duration_minutes if meeting.summary else None,
                    "created_at": meeting.summary.created_at.isoformat() if meeting.summary else None,
                } if meeting.summary else None,
                "calendar_event_id": meeting.calendar_event_id,
                "join_attempts": meeting.join_attempts,
                "last_join_attempt": meeting.last_join_attempt,
                "error_message": meeting.error_message,
                "bot_joined": meeting.bot_joined,
                "created_at": meeting.created_at,
                "updated_at": datetime.now(timezone.utc),
                "joined_at": meeting.joined_at,
                "ended_at": meeting.ended_at,
                "audio_enabled": meeting.audio_enabled,
                "video_enabled": meeting.video_enabled,
                "recording_enabled": meeting.recording_enabled,
                "transcript": meeting.transcript,
                "voice_id": meeting.voice_id,
                "bot_name": meeting.bot_name,
                "context_id": meeting.context_id,
            }
            await self.meetings_collection.update_one(
                {"id": str(meeting.id)},
                {"$set": doc},
                upsert=True
            )
            logger.info(f"Updated meeting: {meeting.id}")
            return meeting
        except Exception as e:
            logger.error(f"Failed to update meeting: {e}")
            raise

    # Meeting Context operations
    async def create_meeting_context(self, meeting_context: MeetingContext) -> MeetingContext:
        """Create a new meeting context."""
        try:
            doc = {
                "id": str(meeting_context.id),
                "user_id": str(meeting_context.user_id),
                "name": meeting_context.name,
                "voice_id": meeting_context.voice_id,
                "context_description": meeting_context.context_description,
                "tools_integrations": meeting_context.tools_integrations,
                "meeting_role": meeting_context.meeting_role.value,
                "tone_personality": meeting_context.tone_personality.value,
                "custom_tone": meeting_context.custom_tone,
                "is_default": meeting_context.is_default,
                "created_at": meeting_context.created_at,
                "updated_at": meeting_context.updated_at,
            }
            await self.meeting_contexts_collection.insert_one(doc)
            logger.info(f"Created meeting context: {meeting_context.id}")
            return meeting_context
        except Exception as e:
            logger.error(f"Failed to create meeting context: {e}")
            raise

    async def get_meeting_context(self, context_id: str, user_id: str) -> Optional[MeetingContext]:
        """Get a meeting context by ID for a specific user."""
        try:
            doc = await self.meeting_contexts_collection.find_one({
                "id": context_id,
                "user_id": str(user_id)
            })
            if not doc:
                return None
            return self._doc_to_meeting_context(doc)
        except Exception as e:
            logger.error(f"Failed to get meeting context: {e}")
            return None

    async def get_meeting_context_by_id(self, context_id: str) -> Optional[MeetingContext]:
        """Get a meeting context by ID without user ownership check."""
        try:
            doc = await self.meeting_contexts_collection.find_one({"id": context_id})
            if not doc:
                return None
            return self._doc_to_meeting_context(doc)
        except Exception as e:
            logger.error(f"Failed to get meeting context: {e}")
            return None

    async def get_meeting_contexts_by_user(self, user_id: str) -> List[MeetingContext]:
        """Get all meeting contexts for a user."""
        try:
            cursor = self.meeting_contexts_collection.find({"user_id": str(user_id)}).sort("created_at", DESCENDING)
            contexts = []
            async for doc in cursor:
                contexts.append(self._doc_to_meeting_context(doc))
            # Sort: default first, then by updated_at
            contexts.sort(key=lambda ctx: (0 if ctx.is_default else 1, -((ctx.updated_at or ctx.created_at).timestamp())))
            return contexts
        except Exception as e:
            logger.error(f"Failed to get meeting contexts: {e}")
            return []

    async def update_meeting_context(self, meeting_context: MeetingContext) -> MeetingContext:
        """Update a meeting context."""
        try:
            meeting_context.updated_at = datetime.now(timezone.utc)
            doc = {
                "id": str(meeting_context.id),
                "user_id": str(meeting_context.user_id),
                "name": meeting_context.name,
                "voice_id": meeting_context.voice_id,
                "context_description": meeting_context.context_description,
                "tools_integrations": meeting_context.tools_integrations,
                "meeting_role": meeting_context.meeting_role.value,
                "tone_personality": meeting_context.tone_personality.value,
                "custom_tone": meeting_context.custom_tone,
                "is_default": meeting_context.is_default,
                "created_at": meeting_context.created_at,
                "updated_at": meeting_context.updated_at,
            }
            await self.meeting_contexts_collection.update_one(
                {"id": str(meeting_context.id)},
                {"$set": doc},
                upsert=True
            )
            logger.info(f"Updated meeting context: {meeting_context.id}")
            return meeting_context
        except Exception as e:
            logger.error(f"Failed to update meeting context: {e}")
            raise

    async def delete_meeting_context(self, context_id: str, user_id: str) -> bool:
        """Delete a meeting context."""
        try:
            # Verify ownership
            context = await self.get_meeting_context(context_id, user_id)
            if not context:
                return False
            await self.meeting_contexts_collection.delete_one({"id": context_id})
            logger.info(f"Deleted meeting context: {context_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete meeting context: {e}")
            return False

    async def clear_default_meeting_contexts(self, user_id: str, exclude_context_id: Optional[str] = None) -> None:
        """Clear default flag from all meeting contexts except the excluded one."""
        try:
            query = {
                "user_id": str(user_id),
                "is_default": True
            }
            if exclude_context_id:
                query["id"] = {"$ne": exclude_context_id}
            
            await self.meeting_contexts_collection.update_many(
                query,
                {
                    "$set": {
                        "is_default": False,
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            logger.info(f"Cleared default contexts for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to clear default contexts: {e}")
            raise

    def _doc_to_meeting_context(self, doc: dict) -> MeetingContext:
        """Convert MongoDB document to MeetingContext."""
        return MeetingContext(
            id=UUID(doc["id"]),
            user_id=UUID(doc["user_id"]),
            name=doc["name"],
            voice_id=doc["voice_id"],
            context_description=doc.get("context_description", ""),
            tools_integrations=doc.get("tools_integrations", []),
            meeting_role=MeetingRole(doc.get("meeting_role", MeetingRole.PARTICIPANT.value)),
            tone_personality=TonePersonality(doc.get("tone_personality", TonePersonality.FRIENDLY.value)),
            custom_tone=doc.get("custom_tone"),
            is_default=doc.get("is_default", False),
            created_at=doc["created_at"] if isinstance(doc["created_at"], datetime) else datetime.fromisoformat(doc["created_at"]),
            updated_at=doc["updated_at"] if isinstance(doc["updated_at"], datetime) else datetime.fromisoformat(doc["updated_at"]),
        )


# Dependency injection
_dao_instance = None


def set_dao_instance(dao):
    """Set the global DAO instance."""
    global _dao_instance
    _dao_instance = dao


def get_dao():
    """Get DAO instance for dependency injection."""
    global _dao_instance
    if _dao_instance is None:
        raise RuntimeError("DAO instance not initialized. Call set_dao_instance() first.")
    return _dao_instance
