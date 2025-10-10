"""
Background worker for processing actions from SQS.
"""
import asyncio
import json
import logging
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

from shared.config import get_settings
from shared.schemas import Action, ActionStatus, ActionType

from .tools.calendar_google import GoogleCalendarTool
from .tools.crm_mock import MockCRMTool
from .tools.email import EmailTool
from .tools.rag import RAGTool
from .tools.slack import SlackTool

logger = logging.getLogger(__name__)
settings = get_settings()


class ActionWorker:
    """Worker for processing actions from SQS."""

    def __init__(self):
        self.sqs_client = None
        self.tools = {}
        self.running = False
        self._initialize_services()

    def _initialize_services(self):
        """Initialize AWS SQS and tools."""
        try:
            if settings.aws_access_key_id:
                self.sqs_client = boto3.client(
                    "sqs",
                    region_name=settings.aws_region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                )

            # Initialize tools
            self.tools = {
                ActionType.CALENDAR_CREATE: GoogleCalendarTool(),
                ActionType.CALENDAR_UPDATE: GoogleCalendarTool(),
                ActionType.CALENDAR_DELETE: GoogleCalendarTool(),
                ActionType.EMAIL_SEND: EmailTool(),
                ActionType.SLACK_MESSAGE: SlackTool(),
                ActionType.CRM_UPDATE: MockCRMTool(),
                ActionType.KNOWLEDGE_SEARCH: RAGTool(),
            }

            logger.info("Action worker initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize action worker: {e}")
            raise

    async def start(self):
        """Start the worker."""
        self.running = True
        logger.info("Starting action worker...")

        while self.running:
            try:
                await self._process_messages()
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def stop(self):
        """Stop the worker."""
        self.running = False
        logger.info("Stopping action worker...")

    async def _process_messages(self):
        """Process messages from SQS queue."""
        try:
            queue_url = self._get_queue_url(settings.actions_queue)

            # Receive messages
            response = self.sqs_client.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=20,  # Long polling
                MessageAttributeNames=["All"],
            )

            messages = response.get("Messages", [])

            if not messages:
                return

            # Process messages
            for message in messages:
                try:
                    await self._process_message(message, queue_url)
                except Exception as e:
                    logger.error(
                        f"Failed to process message {message['MessageId']}: {e}"
                    )
                    # Message will be retried or sent to DLQ

        except ClientError as e:
            logger.error(f"SQS error: {e}")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected error processing messages: {e}")
            await asyncio.sleep(5)

    async def _process_message(self, message: Dict[str, Any], queue_url: str):
        """Process a single message."""
        try:
            # Parse message body
            message_body = json.loads(message["Body"])
            action_data = message_body

            # Create Action object
            action = Action(
                id=action_data["action_id"],
                conversation_id=action_data["conversation_id"],
                turn_id=action_data.get("turn_id"),
                action_type=ActionType(action_data["action_type"]),
                parameters=action_data["parameters"],
                status=ActionStatus.PENDING,
            )

            logger.info(f"Processing action: {action.action_type.value}")

            # Update status to in progress
            action.status = ActionStatus.IN_PROGRESS
            await self._update_action_status(action)

            # Execute action
            result = await self._execute_action(action)

            # Update action with result
            action.status = (
                ActionStatus.COMPLETED if result["success"] else ActionStatus.FAILED
            )
            action.result = result.get("result")
            action.error_message = result.get("error")

            await self._update_action_status(action)

            # Delete message from queue
            self.sqs_client.delete_message(
                QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"]
            )

            logger.info(f"Completed action: {action.action_type.value}")

        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            # Delete message to prevent infinite retries
            try:
                self.sqs_client.delete_message(
                    QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"]
                )
            except:
                pass

    async def _execute_action(self, action: Action) -> Dict[str, Any]:
        """Execute an action using the appropriate tool."""
        try:
            tool = self.tools.get(action.action_type)

            if not tool:
                return {
                    "success": False,
                    "error": f"No tool available for action type: {action.action_type.value}",
                }

            # Execute the action
            if action.action_type == ActionType.CALENDAR_CREATE:
                result = await tool.create_event(action.parameters)
            elif action.action_type == ActionType.CALENDAR_UPDATE:
                result = await tool.update_event(action.parameters)
            elif action.action_type == ActionType.CALENDAR_DELETE:
                result = await tool.delete_event(action.parameters)
            elif action.action_type == ActionType.EMAIL_SEND:
                result = await tool.send_email(action.parameters)
            elif action.action_type == ActionType.SLACK_MESSAGE:
                result = await tool.send_message(action.parameters)
            elif action.action_type == ActionType.CRM_UPDATE:
                result = await tool.update_contact(action.parameters)
            elif action.action_type == ActionType.KNOWLEDGE_SEARCH:
                result = await tool.search(action.parameters)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown action type: {action.action_type.value}",
                }

            return result

        except Exception as e:
            logger.error(f"Error executing action {action.action_type.value}: {e}")
            return {"success": False, "error": str(e)}

    async def _update_action_status(self, action: Action):
        """Update action status in database."""
        try:
            # This would update the action in DynamoDB
            # For now, just log
            logger.info(f"Action {action.id} status: {action.status.value}")

        except Exception as e:
            logger.error(f"Failed to update action status: {e}")

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


async def main():
    """Main worker entry point."""
    worker = ActionWorker()

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
