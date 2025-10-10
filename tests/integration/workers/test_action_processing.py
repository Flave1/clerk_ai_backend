"""Integration tests for worker action processing."""

import pytest
from unittest.mock import patch, Mock
from services.workers.worker import ActionWorker


class TestActionProcessing:
    """Test action processing workflow."""

    @patch('services.workers.worker.boto3.client')
    def test_process_calendar_action(self, mock_boto_client):
        """Test processing a calendar action."""
        # Mock SQS client
        mock_sqs = Mock()
        mock_boto_client.return_value = mock_sqs
        
        # Mock SQS message
        mock_message = {
            "Messages": [{
                "Body": '{"id": "action-123", "action_type": "calendar_create", "parameters": {"title": "Test Event"}}',
                "ReceiptHandle": "test-receipt"
            }]
        }
        mock_sqs.receive_message.return_value = mock_message
        mock_sqs.delete_message.return_value = {}

        # Mock Google Calendar tool
        with patch('services.workers.tools.calendar_google.GoogleCalendarTool') as mock_calendar:
            mock_calendar_instance = Mock()
            mock_calendar.return_value = mock_calendar_instance
            mock_calendar_instance.create_event.return_value = {"id": "test-event-id"}

            processor = ActionWorker()
            processor._process_messages()

            # Verify calendar tool was called
            mock_calendar_instance.create_event.assert_called_once_with({"title": "Test Event"})

    @patch('services.workers.worker.boto3.client')
    def test_process_slack_action(self, mock_boto_client):
        """Test processing a Slack action."""
        # Mock SQS client
        mock_sqs = Mock()
        mock_boto_client.return_value = mock_sqs
        
        # Mock SQS message
        mock_message = {
            "Messages": [{
                "Body": '{"id": "action-456", "action_type": "slack_message", "parameters": {"channel": "test-channel", "text": "Hello"}}',
                "ReceiptHandle": "test-receipt"
            }]
        }
        mock_sqs.receive_message.return_value = mock_message
        mock_sqs.delete_message.return_value = {}

        # Mock Slack tool
        with patch('services.workers.tools.slack.SlackTool') as mock_slack:
            mock_slack_instance = Mock()
            mock_slack.return_value = mock_slack_instance
            mock_slack_instance.send_message.return_value = {"ok": True}

            processor = ActionWorker()
            processor._process_messages()

            # Verify Slack tool was called
            mock_slack_instance.send_message.assert_called_once_with("test-channel", "Hello")