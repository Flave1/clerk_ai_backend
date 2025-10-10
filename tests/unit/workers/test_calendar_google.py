"""Unit tests for Google Calendar tool."""

import pytest
from unittest.mock import Mock, patch
from services.workers.tools.calendar_google import GoogleCalendarTool


class TestGoogleCalendarTool:
    """Test Google Calendar integration."""

    @patch('services.workers.tools.calendar_google.build')
    @patch('services.workers.tools.calendar_google.service_account.Credentials.from_service_account_file')
    def test_initialize_service(self, mock_creds, mock_build):
        """Test service initialization."""
        mock_creds.return_value = Mock()
        mock_build.return_value = Mock()
        
        tool = GoogleCalendarTool()
        assert tool.service is not None
        mock_creds.assert_called_once()
        mock_build.assert_called_once_with("calendar", "v3", credentials=mock_creds.return_value)

    def test_create_event(self):
        """Test creating a calendar event."""
        with patch('services.workers.tools.calendar_google.build') as mock_build:
            mock_service = Mock()
            mock_events = Mock()
            mock_insert = Mock()
            mock_execute = Mock()
            
            mock_build.return_value = mock_service
            mock_service.events.return_value = mock_events
            mock_events.insert.return_value = mock_insert
            mock_insert.execute.return_value = {"id": "test-event-id"}
            
            tool = GoogleCalendarTool()
            tool.service = mock_service
            
            result = tool.create_event({
                "summary": "Test Event",
                "start": {"dateTime": "2024-01-01T10:00:00Z"},
                "end": {"dateTime": "2024-01-01T11:00:00Z"}
            })
            
            assert result["id"] == "test-event-id"
            mock_events.insert.assert_called_once()

    def test_list_events(self):
        """Test listing calendar events."""
        with patch('services.workers.tools.calendar_google.build') as mock_build:
            mock_service = Mock()
            mock_events = Mock()
            mock_list = Mock()
            mock_execute = Mock()
            
            mock_build.return_value = mock_service
            mock_service.events.return_value = mock_events
            mock_events.list.return_value = mock_list
            mock_list.execute.return_value = {"items": []}
            
            tool = GoogleCalendarTool()
            tool.service = mock_service
            
            result = tool.list_events()
            
            assert result == []
            mock_events.list.assert_called_once()