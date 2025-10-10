"""
Slack integration tool.
"""
import logging
from typing import Any, Dict

import slack_sdk
from slack_sdk.errors import SlackApiError

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SlackTool:
    """Slack integration tool."""

    def __init__(self):
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Slack client."""
        try:
            if not settings.slack_bot_token:
                logger.warning("Slack bot token not configured")
                return

            self.client = slack_sdk.WebClient(token=settings.slack_bot_token)
            logger.info("Slack client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Slack client: {e}")
            self.client = None

    async def send_message(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to Slack."""
        try:
            if not self.client:
                logger.warning("Slack client not initialized")
                return {
                    "success": False,
                    "error": "Slack bot token not configured. Please set SLACK_BOT_TOKEN with proper scopes."
                }

            channel = parameters.get("channel")
            message = parameters.get("message", "")

            if not channel or not message:
                return {"success": False, "error": "channel and message are required"}

            # Send message
            response = self.client.chat_postMessage(channel=channel, text=message)

            logger.info(f"Sent Slack message to {channel}: {response['ts']}")

            return {
                "success": True,
                "result": {
                    "message_ts": response["ts"],
                    "channel": channel,
                    "message": message,
                },
            }

        except SlackApiError as e:
            logger.error(f"Slack API error: {e}")
            return {
                "success": False,
                "error": f'Slack API error: {e.response["error"]}',
            }
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return {"success": False, "error": str(e)}

    async def send_direct_message(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send a direct message to a user."""
        try:
            if not self.client:
                return {"success": False, "error": "Slack client not initialized"}

            user_id = parameters.get("user_id")
            message = parameters.get("message", "")

            if not user_id or not message:
                return {"success": False, "error": "user_id and message are required"}

            # Open DM channel
            dm_response = self.client.conversations_open(users=user_id)
            channel_id = dm_response["channel"]["id"]

            # Send message
            response = self.client.chat_postMessage(channel=channel_id, text=message)

            logger.info(f"Sent DM to {user_id}: {response['ts']}")

            return {
                "success": True,
                "result": {
                    "message_ts": response["ts"],
                    "user_id": user_id,
                    "channel_id": channel_id,
                    "message": message,
                },
            }

        except SlackApiError as e:
            logger.error(f"Slack API error: {e}")
            return {
                "success": False,
                "error": f'Slack API error: {e.response["error"]}',
            }
        except Exception as e:
            logger.error(f"Failed to send DM: {e}")
            return {"success": False, "error": str(e)}

    async def get_user_info(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get user information."""
        try:
            if not self.client:
                return {"success": False, "error": "Slack client not initialized"}

            user_id = parameters.get("user_id")
            if not user_id:
                return {"success": False, "error": "user_id is required"}

            # Get user info
            response = self.client.users_info(user=user_id)
            user = response["user"]

            return {
                "success": True,
                "result": {
                    "user_id": user["id"],
                    "name": user.get("real_name", user.get("name")),
                    "email": user.get("profile", {}).get("email"),
                    "timezone": user.get("tz"),
                    "is_bot": user.get("is_bot", False),
                },
            }

        except SlackApiError as e:
            logger.error(f"Slack API error: {e}")
            return {
                "success": False,
                "error": f'Slack API error: {e.response["error"]}',
            }
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return {"success": False, "error": str(e)}

    async def list_channels(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """List Slack channels."""
        try:
            if not self.client:
                return {"success": False, "error": "Slack client not initialized"}

            # List channels
            response = self.client.conversations_list(
                types="public_channel,private_channel", limit=100
            )

            channels = []
            for channel in response["channels"]:
                channels.append(
                    {
                        "id": channel["id"],
                        "name": channel["name"],
                        "is_private": channel.get("is_private", False),
                        "num_members": channel.get("num_members", 0),
                    }
                )

            return {
                "success": True,
                "result": {"channels": channels, "count": len(channels)},
            }

        except SlackApiError as e:
            logger.error(f"Slack API error: {e}")
            return {
                "success": False,
                "error": f'Slack API error: {e.response["error"]}',
            }
        except Exception as e:
            logger.error(f"Failed to list channels: {e}")
            return {"success": False, "error": str(e)}
