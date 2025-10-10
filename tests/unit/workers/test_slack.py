"""Unit tests for Slack tool."""

import pytest
from unittest.mock import Mock, patch
from services.workers.tools.slack import SlackTool


class TestSlackTool:
    """Test Slack integration."""

    def test_send_message(self):
        """Test sending a Slack message."""
        with patch('services.workers.tools.slack.WebClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            mock_instance.chat_postMessage.return_value = {"ok": True, "ts": "1234567890.123456"}
            
            tool = SlackTool()
            result = tool.send_message("test-channel", "Hello, world!")
            
            assert result["ok"] is True
            assert "ts" in result
            mock_instance.chat_postMessage.assert_called_once_with(
                channel="test-channel",
                text="Hello, world!"
            )

    def test_get_user_info(self):
        """Test getting user information."""
        with patch('services.workers.tools.slack.WebClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            mock_instance.users_info.return_value = {"user": {"name": "test_user"}}
            
            tool = SlackTool()
            result = tool.get_user_info("U1234567890")
            
            assert result["user"]["name"] == "test_user"
            mock_instance.users_info.assert_called_once_with(user="U1234567890")

    def test_get_channel_info(self):
        """Test getting channel information."""
        with patch('services.workers.tools.slack.WebClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            mock_instance.conversations_info.return_value = {"channel": {"name": "test_channel"}}
            
            tool = SlackTool()
            result = tool.get_channel_info("C1234567890")
            
            assert result["channel"]["name"] == "test_channel"
            mock_instance.conversations_info.assert_called_once_with(channel="C1234567890")