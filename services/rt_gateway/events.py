"""
Event publishing system for SNS/SQS integration.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

from shared.config import get_settings
from shared.schemas import Action, Conversation, Turn

logger = logging.getLogger(__name__)
settings = get_settings()


class EventPublisher:
    """Publishes events to SNS/SQS for async processing."""

    def __init__(self):
        self.sns_client = None
        self.sqs_client = None
        self.initialized = False

    async def initialize(self):
        """Initialize AWS clients."""
        try:
            if settings.aws_access_key_id:
                # Initialize SNS
                self.sns_client = boto3.client(
                    "sns",
                    region_name=settings.aws_region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                )

                # Initialize SQS
                self.sqs_client = boto3.client(
                    "sqs",
                    region_name=settings.aws_region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                )

                self.initialized = True
                logger.info("Event publisher initialized successfully")
            else:
                logger.warning(
                    "AWS credentials not provided, event publishing disabled"
                )

        except Exception as e:
            logger.error(f"Failed to initialize event publisher: {e}")
            raise

    async def publish_event(self, event_data: Dict[str, Any]):
        """Publish event to SNS topic."""
        if not self.initialized:
            logger.warning("Event publisher not initialized, skipping event")
            return

        try:
            # Prepare event message
            message = {
                "event_type": event_data.get("type", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
                "data": event_data,
                "source": "rt_gateway",
            }

            # Publish to SNS topic
            # Get AWS account ID first
            try:
                sts_client = boto3.client(
                    "sts",
                    region_name=settings.aws_region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                )
                account_id = sts_client.get_caller_identity()["Account"]
                topic_arn = f"arn:aws:sns:{settings.aws_region}:{account_id}:{settings.sns_topic_prefix}{settings.events_topic}"
            except Exception as e:
                logger.warning(f"Could not get AWS account ID: {e}")
                # Fallback to a simpler topic name
                topic_arn = f"{settings.sns_topic_prefix}{settings.events_topic}"

            # Debug: Check for UUID objects in the message
            try:
                json_message = json.dumps(message)
            except TypeError as e:
                logger.error(f"JSON serialization failed: {e}")
                logger.error(f"Message data: {message}")
                # Try to convert UUIDs to strings
                def convert_uuids(obj):
                    if isinstance(obj, dict):
                        return {k: convert_uuids(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_uuids(item) for item in obj]
                    elif hasattr(obj, '__str__') and 'UUID' in str(type(obj)):
                        return str(obj)
                    else:
                        return obj
                
                message = convert_uuids(message)
                json_message = json.dumps(message)
            
            response = self.sns_client.publish(
                TopicArn=topic_arn,
                Message=json_message,
                Subject=f"Event: {event_data.get('type', 'unknown')}",
            )

            logger.info(
                f"Published event {event_data.get('type')} to SNS: {response['MessageId']}"
            )

        except ClientError as e:
            logger.error(f"Failed to publish event to SNS: {e}")
        except Exception as e:
            logger.error(f"Unexpected error publishing event: {e}")

    async def publish_action(self, action: Action):
        """Publish action to SQS queue for processing."""
        if not self.initialized:
            logger.warning("Event publisher not initialized, skipping action")
            return

        try:
            # Prepare action message
            message = {
                "action_id": str(action.id),
                "action_type": action.action_type.value,
                "conversation_id": str(action.conversation_id),
                "turn_id": str(action.turn_id) if action.turn_id else None,
                "parameters": action.parameters,
                "created_at": action.created_at.isoformat(),
            }

            # Send to SQS queue
            queue_url = self._get_queue_url(settings.actions_queue)

            response = self.sqs_client.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(message),
                MessageAttributes={
                    "ActionType": {
                        "StringValue": action.action_type.value,
                        "DataType": "String",
                    },
                    "ConversationId": {
                        "StringValue": str(action.conversation_id),
                        "DataType": "String",
                    },
                },
            )

            logger.info(f"Published action {action.id} to SQS: {response['MessageId']}")

        except ClientError as e:
            logger.error(f"Failed to publish action to SQS: {e}")
        except Exception as e:
            logger.error(f"Unexpected error publishing action: {e}")

    async def publish_conversation_update(
        self, conversation: Conversation, update_type: str, data: Dict[str, Any]
    ):
        """Publish conversation update."""
        event_data = {
            "type": "conversation_update",
            "conversation_id": str(conversation.id),
            "update_type": update_type,
            "status": conversation.status.value,
            "data": data,
        }

        await self.publish_event(event_data)

    async def publish_turn_update(self, turn: Turn, update_type: str):
        """Publish turn update."""
        event_data = {
            "type": "turn_update",
            "turn_id": str(turn.id),
            "conversation_id": str(turn.conversation_id),
            "turn_type": turn.turn_type.value,
            "update_type": update_type,
            "content": turn.content[:100] + "..."
            if len(turn.content) > 100
            else turn.content,
        }

        await self.publish_event(event_data)

    async def publish_action_update(self, action: Action, update_type: str):
        """Publish action update."""
        event_data = {
            "type": "action_update",
            "action_id": str(action.id),
            "action_type": action.action_type.value,
            "status": action.status.value,
            "update_type": update_type,
            "conversation_id": str(action.conversation_id),
        }

        await self.publish_event(event_data)

    def _get_queue_url(self, queue_name: str) -> str:
        """Get SQS queue URL."""
        try:
            response = self.sqs_client.get_queue_url(
                QueueName=f"{settings.sqs_queue_prefix}{queue_name}"
            )
            return response["QueueUrl"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "AWS.SimpleQueueService.NonExistentQueue":
                logger.warning(f"Queue {queue_name} does not exist, creating it")
                return self._create_queue(queue_name)
            else:
                raise

    def _create_queue(self, queue_name: str) -> str:
        """Create SQS queue if it doesn't exist."""
        try:
            response = self.sqs_client.create_queue(
                QueueName=f"{settings.sqs_queue_prefix}{queue_name}",
                Attributes={
                    "VisibilityTimeout": "300",  # 5 minutes
                    "MessageRetentionPeriod": "1209600",  # 14 days
                    "ReceiveMessageWaitTimeSeconds": "20",  # Long polling
                },
            )
            return response["QueueUrl"]
        except Exception as e:
            logger.error(f"Failed to create queue {queue_name}: {e}")
            raise

    async def publish_health_check(self):
        """Publish health check event."""
        event_data = {
            "type": "health_check",
            "service": "rt_gateway",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
        }

        await self.publish_event(event_data)

    async def publish_error(
        self, error_type: str, error_message: str, context: Dict[str, Any]
    ):
        """Publish error event."""
        event_data = {
            "type": "error",
            "error_type": error_type,
            "error_message": error_message,
            "context": context,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.publish_event(event_data)
