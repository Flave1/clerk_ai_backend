"""
Data Access Object for DynamoDB operations.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError

from shared.config import get_settings
from shared.schemas import (
    Action,
    ActionStatus,
    RoomInfo,
    User,
    Meeting,
    ApiKey,
    ApiKeyStatus,
    UserIntegration,
    IntegrationStatus,
    NewsletterSubscription,
)

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
        self.user_integrations_table = None
        self.meeting_contexts_table = None
        self.newsletter_table = None
        self.initialized = False

    async def initialize(self):
        """Initialize DynamoDB client and tables."""
        try:
            # Determine if we should use local DynamoDB
            use_local = settings.use_local_dynamodb
            
            # Auto-detect: if USE_LOCAL_DYNAMODB is not explicitly set and AWS credentials are missing,
            # try local DynamoDB first
            if not use_local and not settings.aws_access_key_id:
                logger.info("No AWS credentials found. Attempting to use local DynamoDB...")
                use_local = True
            
            # Try local DynamoDB first if configured
            if use_local:
                try:
                    logger.info(f"Attempting to connect to local DynamoDB at {settings.dynamodb_local_endpoint}")
                    self.dynamodb = boto3.resource(
                        "dynamodb",
                        region_name=settings.aws_region,
                        endpoint_url=settings.dynamodb_local_endpoint,
                        aws_access_key_id="dummy",  # Required for local DynamoDB
                        aws_secret_access_key="dummy",  # Required for local DynamoDB
                    )
                    # Test connection by listing tables
                    list(self.dynamodb.tables.limit(1))
                    logger.info(f"✅ DynamoDB: Connected to LOCAL DynamoDB at {settings.dynamodb_local_endpoint} (ONLINE)")
                except Exception as local_error:
                    if settings.use_local_dynamodb:
                        # If explicitly set to use local, fail
                        logger.error(f"❌ Failed to connect to local DynamoDB: {local_error}")
                        logger.error("Please ensure local DynamoDB is running:")
                        logger.error(f"  1. Start DynamoDB Local: docker-compose up -d dynamodb")
                        logger.error(f"  2. Or set DYNAMODB_LOCAL_ENDPOINT if using a different endpoint")
                        raise RuntimeError(f"Failed to connect to local DynamoDB: {local_error}")
                    else:
                        # If auto-detecting, fall back to AWS
                        logger.warning(f"⚠️  Local DynamoDB not available at {settings.dynamodb_local_endpoint}, falling back to AWS")
                        use_local = False
            
            # Use AWS DynamoDB if local is not being used
            if not use_local:
                if settings.aws_access_key_id:
                    # Use AWS credentials if provided
                    logger.info(f"Using AWS credentials for DynamoDB connection (Region: {settings.aws_region})")
                    try:
                        self.dynamodb = boto3.resource(
                            "dynamodb",
                            region_name=settings.aws_region,
                            aws_access_key_id=settings.aws_access_key_id,
                            aws_secret_access_key=settings.aws_secret_access_key,
                        )
                        # Test connection by listing tables
                        list(self.dynamodb.tables.limit(1))
                        logger.info("🌐 DynamoDB: Connected to AWS using credentials (ONLINE)")
                    except Exception as cred_error:
                        logger.error(f"❌ Failed to connect to AWS DynamoDB using credentials: {cred_error}")
                        logger.error("Please check:")
                        logger.error("  1. AWS credentials are valid")
                        logger.error("  2. AWS region is correct")
                        logger.error("  3. Network connectivity to AWS services")
                        raise RuntimeError(f"Failed to connect to AWS DynamoDB: {cred_error}")
                else:
                    # Try using IAM role (for ECS/Lambda)
                    logger.info(f"Attempting to use IAM role for DynamoDB (Region: {settings.aws_region})")
                    try:
                        self.dynamodb = boto3.resource(
                            "dynamodb",
                            region_name=settings.aws_region,
                        )
                        # Test connection by listing tables
                        list(self.dynamodb.tables.limit(1))
                        logger.info("🌐 DynamoDB: Connected to AWS using IAM role (ONLINE)")
                    except Exception as iam_error:
                        logger.error(f"❌ Failed to connect to AWS DynamoDB using IAM role: {iam_error}")
                        logger.error("Please check:")
                        logger.error("  1. IAM role has DynamoDB permissions")
                        logger.error("  2. AWS credentials are set in Secrets Manager")
                        logger.error("  3. Network connectivity to AWS services")
                        logger.error("  4. Or set USE_LOCAL_DYNAMODB=true to use local DynamoDB")
                        raise RuntimeError(f"Failed to connect to AWS DynamoDB: {iam_error}")

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
            
            self.user_integrations_table = self.dynamodb.Table(
                f"{settings.dynamodb_table_prefix}{settings.user_integrations_table}"
            )
            self.meeting_contexts_table = self.dynamodb.Table(
                f"{settings.dynamodb_table_prefix}{settings.meeting_contexts_table}"
            )
            self.newsletter_table = self.dynamodb.Table(
                f"{settings.dynamodb_table_prefix}{settings.newsletter_table}"
            )

            # Test connection by trying to describe one table
            try:
                self.users_table.load()
                logger.info(f"✅ Successfully connected to table: {self.users_table.table_name}")
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.warning(f"⚠️  Table {self.users_table.table_name} does not exist. Please create it first.")
                else:
                    logger.error(f"❌ Error accessing table {self.users_table.table_name}: {e}")
                    raise

            self.initialized = True
            logger.info("✅ DynamoDB DAO initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize DynamoDB DAO: {e}", exc_info=True)
            logger.error("Please check:")
            logger.error("  1. AWS credentials are set (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
            logger.error("  2. IAM role has DynamoDB permissions (if using ECS/Lambda)")
            logger.error("  3. AWS region is configured (AWS_REGION)")
            logger.error("  4. Network connectivity to AWS services")
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

    # Newsletter operations
    async def get_newsletter_signup(self, email: str) -> Optional[NewsletterSubscription]:
        """Retrieve a newsletter subscription by email."""
        if not self.newsletter_table:
            raise RuntimeError("Newsletter table is not initialized")

        try:
            response = self.newsletter_table.get_item(Key={"email": email.lower()})
            item = response.get("Item")
            if not item:
                return None

            signed_up_at = (
                datetime.fromisoformat(item["signed_up_at"])
                if isinstance(item.get("signed_up_at"), str)
                else datetime.utcnow()
            )

            return NewsletterSubscription(
                email=item["email"],
                name=item.get("name", ""),
                country=item.get("country", ""),
                signed_up_at=signed_up_at,
            )
        except Exception as e:
            logger.error(f"Failed to get newsletter signup for {email}: {e}")
            raise

    async def add_newsletter_signup(self, signup: NewsletterSubscription) -> NewsletterSubscription:
        """Create a new newsletter subscription."""
        if not self.newsletter_table:
            raise RuntimeError("Newsletter table is not initialized")

        item = {
            "email": signup.email.lower(),
            "name": signup.name,
            "country": signup.country,
            "signed_up_at": signup.signed_up_at.isoformat(),
        }

        try:
            self.newsletter_table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(email)",
            )
            return signup
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.info("Newsletter signup already exists for email %s", signup.email)
                raise ValueError("Email already exists in newsletter")
            logger.error(f"Failed to add newsletter signup: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to add newsletter signup: {e}")
            raise

    async def list_newsletter_signups(self) -> List[NewsletterSubscription]:
        """List all newsletter subscriptions."""
        if not self.newsletter_table:
            raise RuntimeError("Newsletter table is not initialized")

        try:
            response = self.newsletter_table.scan()
            items = response.get("Items", [])

            signups: List[NewsletterSubscription] = []
            for item in items:
                signed_up_at = (
                    datetime.fromisoformat(item["signed_up_at"])
                    if isinstance(item.get("signed_up_at"), str)
                    else datetime.utcnow()
                )
                signups.append(
                    NewsletterSubscription(
                        email=item.get("email", ""),
                        name=item.get("name", ""),
                        country=item.get("country", ""),
                        signed_up_at=signed_up_at,
                    )
                )

            # Sort by newest first
            signups.sort(key=lambda s: s.signed_up_at, reverse=True)
            return signups
        except Exception as e:
            logger.error(f"Failed to list newsletter signups: {e}")
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

    # User Integration operations
    async def create_user_integration(self, user_integration: UserIntegration) -> UserIntegration:
        """Create a new user integration."""
        try:
            if not self.user_integrations_table:
                raise RuntimeError("User integrations table not initialized")
            
            item = {
                "id": str(user_integration.id),
                "user_id": str(user_integration.user_id),
                "integration_id": user_integration.integration_id,
                "status": user_integration.status.value,
                "access_token": user_integration.access_token or "",
                "refresh_token": user_integration.refresh_token or "",
                "token_type": user_integration.token_type,
                "expires_at": user_integration.expires_at.isoformat() if user_integration.expires_at else None,
                "scope": user_integration.scope or "",
                "connected_at": user_integration.connected_at.isoformat() if user_integration.connected_at else None,
                "last_used_at": user_integration.last_used_at.isoformat() if user_integration.last_used_at else None,
                "last_refreshed_at": user_integration.last_refreshed_at.isoformat() if user_integration.last_refreshed_at else None,
                "error_message": user_integration.error_message or "",
                "metadata": json.dumps(user_integration.metadata),
                "created_at": user_integration.created_at.isoformat(),
                "updated_at": user_integration.updated_at.isoformat(),
            }
            
            self.user_integrations_table.put_item(Item=item)
            logger.info(f"Created user integration: {user_integration.id} ({user_integration.integration_id})")
            return user_integration

        except Exception as e:
            logger.error(f"Failed to create user integration: {e}")
            raise

    async def get_user_integration(self, user_id: str, integration_id: str) -> Optional[UserIntegration]:
        """Get a user integration by user_id and integration_id."""
        try:
            if not self.user_integrations_table:
                raise RuntimeError("User integrations table not initialized")
            
            # Query by user_id using GSI, then filter by integration_id
            response = self.user_integrations_table.query(
                IndexName="user-id-index",
                KeyConditionExpression=Key("user_id").eq(user_id),
                FilterExpression=Attr("integration_id").eq(integration_id),
            )
            
            items = response.get("Items", [])
            if not items:
                return None
            
            item = items[0]
            return UserIntegration(
                id=UUID(item["id"]),
                user_id=UUID(item["user_id"]),
                integration_id=item["integration_id"],
                status=IntegrationStatus(item["status"]),
                access_token=item.get("access_token") or None,
                refresh_token=item.get("refresh_token") or None,
                token_type=item.get("token_type", "Bearer"),
                expires_at=datetime.fromisoformat(item["expires_at"]) if item.get("expires_at") else None,
                scope=item.get("scope") or None,
                connected_at=datetime.fromisoformat(item["connected_at"]) if item.get("connected_at") else None,
                last_used_at=datetime.fromisoformat(item["last_used_at"]) if item.get("last_used_at") else None,
                last_refreshed_at=datetime.fromisoformat(item["last_refreshed_at"]) if item.get("last_refreshed_at") else None,
                error_message=item.get("error_message") or None,
                metadata=json.loads(item.get("metadata", "{}")),
                created_at=datetime.fromisoformat(item["created_at"]),
                updated_at=datetime.fromisoformat(item["updated_at"]),
            )

        except Exception as e:
            logger.error(f"Failed to get user integration: {e}")
            return None

    async def get_user_integrations_by_user(self, user_id: str) -> List[UserIntegration]:
        """Get all integrations for a user."""
        try:
            if not self.user_integrations_table:
                raise RuntimeError("User integrations table not initialized")
            
            response = self.user_integrations_table.query(
                IndexName="user-id-index",
                KeyConditionExpression=Key("user_id").eq(user_id),
                ScanIndexForward=False,
            )
            
            integrations = []
            for item in response.get("Items", []):
                integrations.append(UserIntegration(
                    id=UUID(item["id"]),
                    user_id=UUID(item["user_id"]),
                    integration_id=item["integration_id"],
                    status=IntegrationStatus(item["status"]),
                    access_token=item.get("access_token") or None,
                    refresh_token=item.get("refresh_token") or None,
                    token_type=item.get("token_type", "Bearer"),
                    expires_at=datetime.fromisoformat(item["expires_at"]) if item.get("expires_at") else None,
                    scope=item.get("scope") or None,
                    connected_at=datetime.fromisoformat(item["connected_at"]) if item.get("connected_at") else None,
                    last_used_at=datetime.fromisoformat(item["last_used_at"]) if item.get("last_used_at") else None,
                    last_refreshed_at=datetime.fromisoformat(item["last_refreshed_at"]) if item.get("last_refreshed_at") else None,
                    error_message=item.get("error_message") or None,
                    metadata=json.loads(item.get("metadata", "{}")),
                    created_at=datetime.fromisoformat(item["created_at"]),
                    updated_at=datetime.fromisoformat(item["updated_at"]),
                ))
            
            return integrations

        except Exception as e:
            logger.error(f"Failed to get user integrations: {e}")
            return []

    async def update_user_integration(self, user_integration: UserIntegration) -> UserIntegration:
        """Update a user integration."""
        try:
            if not self.user_integrations_table:
                raise RuntimeError("User integrations table not initialized")
            
            item = {
                "id": str(user_integration.id),
                "user_id": str(user_integration.user_id),
                "integration_id": user_integration.integration_id,
                "status": user_integration.status.value,
                "access_token": user_integration.access_token or "",
                "refresh_token": user_integration.refresh_token or "",
                "token_type": user_integration.token_type,
                "expires_at": user_integration.expires_at.isoformat() if user_integration.expires_at else None,
                "scope": user_integration.scope or "",
                "connected_at": user_integration.connected_at.isoformat() if user_integration.connected_at else None,
                "last_used_at": user_integration.last_used_at.isoformat() if user_integration.last_used_at else None,
                "last_refreshed_at": user_integration.last_refreshed_at.isoformat() if user_integration.last_refreshed_at else None,
                "error_message": user_integration.error_message or "",
                "metadata": json.dumps(user_integration.metadata),
                "created_at": user_integration.created_at.isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            
            self.user_integrations_table.put_item(Item=item)
            logger.info(f"Updated user integration: {user_integration.id}")
            return user_integration

        except Exception as e:
            logger.error(f"Failed to update user integration: {e}")
            raise

    async def delete_user_integration(self, integration_id: str, user_id: str) -> bool:
        """Delete a user integration."""
        try:
            if not self.user_integrations_table:
                raise RuntimeError("User integrations table not initialized")
            
            # First, get the integration to find its ID
            user_integration = await self.get_user_integration(user_id, integration_id)
            if not user_integration:
                return False
            
            self.user_integrations_table.delete_item(
                Key={"id": str(user_integration.id)}
            )
            logger.info(f"Deleted user integration: {user_integration.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete user integration: {e}")
            return False


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
