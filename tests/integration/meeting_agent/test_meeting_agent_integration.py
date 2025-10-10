"""
Integration tests for Meeting Agent service.

This module contains integration tests for the meeting agent functionality
including platform clients, transcription, and summarization services.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from services.shared.schemas import Meeting, MeetingPlatform, MeetingStatus, TranscriptionChunk
from services.meeting_agent.models import MeetingJoinRequest, MeetingJoinResponse, MeetingConfig
from services.meeting_agent.google_meet_client import GoogleMeetClient
from services.meeting_agent.zoom_client import ZoomClientWrapper
from services.meeting_agent.teams_client import MicrosoftTeamsClient
from services.meeting_agent.transcription_service import TranscriptionService
from services.meeting_agent.summarization_service import SummarizationService
from services.meeting_agent.notifier import NotificationService
from services.meeting_agent.scheduler import MeetingScheduler


@pytest.fixture
def sample_meeting():
    """Create a sample meeting for testing."""
    return Meeting(
        platform=MeetingPlatform.GOOGLE_MEET,
        meeting_url="https://meet.google.com/test-meeting-id",
        meeting_id_external="test-meeting-id",
        title="Test Meeting",
        description="A test meeting for integration testing",
        start_time=datetime.utcnow() + timedelta(minutes=5),
        end_time=datetime.utcnow() + timedelta(minutes=35),
        organizer_email="organizer@example.com",
        ai_email="favourremmanuel433@gmail.com"
    )


@pytest.fixture
def sample_transcription_chunks():
    """Create sample transcription chunks for testing."""
    from services.meeting_agent.models import TranscriptionChunk
    
    return [
        TranscriptionChunk(
            meeting_id="test-meeting-id",
            text="Hello everyone, welcome to our meeting.",
            confidence=0.95,
            timestamp=datetime.utcnow(),
            is_final=True
        ),
        TranscriptionChunk(
            meeting_id="test-meeting-id",
            text="Today we'll discuss the project updates.",
            confidence=0.92,
            timestamp=datetime.utcnow(),
            is_final=True
        ),
        TranscriptionChunk(
            meeting_id="test-meeting-id",
            text="The budget has been approved for Q1.",
            confidence=0.88,
            timestamp=datetime.utcnow(),
            is_final=True
        )
    ]


class TestGoogleMeetClient:
    """Test Google Meet client functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test Google Meet client initialization."""
        with patch('selenium.webdriver.Chrome') as mock_chrome:
            client = GoogleMeetClient()
            await client.initialize()
            
            assert client.driver is not None
            mock_chrome.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_meeting_id(self):
        """Test meeting ID extraction from URL."""
        client = GoogleMeetClient()
        
        # Test valid URLs
        assert client.extract_meeting_id("https://meet.google.com/abc-def-ghi") == "abc-def-ghi"
        assert client.extract_meeting_id("https://meet.google.com/abc-def-ghi/jkl-mno-pqr") == "jkl-mno-pqr"
        
        # Test invalid URL
        assert client.extract_meeting_id("https://invalid-url.com") is None
    
    @pytest.mark.asyncio
    async def test_join_meeting_success(self, sample_meeting):
        """Test successful meeting join."""
        with patch('selenium.webdriver.Chrome') as mock_chrome:
            client = GoogleMeetClient()
            await client.initialize()
            
            # Mock successful join
            mock_driver = mock_chrome.return_value
            mock_driver.current_url = "https://meet.google.com/test-meeting"
            mock_driver.page_source = "Join now"
            
            with patch.object(client, '_join_meeting_room') as mock_join:
                with patch.object(client, '_start_audio_capture') as mock_audio:
                    result = await client.join_meeting(sample_meeting)
                    
                    assert result.success is True
                    assert result.meeting_id == sample_meeting.id
                    mock_join.assert_called_once()
                    mock_audio.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_join_meeting_failure(self, sample_meeting):
        """Test meeting join failure."""
        client = GoogleMeetClient()
        
        # Test with invalid URL
        sample_meeting.meeting_url = "invalid-url"
        result = await client.join_meeting(sample_meeting)
        
        assert result.success is False
        assert "Could not extract meeting ID" in result.error_message


class TestZoomClient:
    """Test Zoom client functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test Zoom client initialization."""
        with patch('zoomus.ZoomClient') as mock_zoom:
            client = ZoomClientWrapper()
            await client.initialize()
            
            assert client.client is not None
            mock_zoom.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_meeting_id(self):
        """Test meeting ID extraction from URL."""
        client = ZoomClientWrapper()
        
        # Test valid URLs
        assert client.extract_meeting_id("https://zoom.us/j/123456789") == "123456789"
        assert client.extract_meeting_id("https://zoom.us/my/test-meeting") == "test-meeting"
        
        # Test invalid URL
        assert client.extract_meeting_id("https://invalid-url.com") is None
    
    @pytest.mark.asyncio
    async def test_authenticate_success(self):
        """Test successful authentication."""
        client = ZoomClientWrapper()
        await client.initialize()
        
        with patch.object(client, '_exchange_code_for_token') as mock_exchange:
            mock_exchange.return_value = {'access_token': 'test-token'}
            
            result = await client.authenticate('test-code')
            assert result is True
            assert client.client.access_token == 'test-token'


class TestMicrosoftTeamsClient:
    """Test Microsoft Teams client functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test Microsoft Teams client initialization."""
        with patch('msal.ConfidentialClientApplication') as mock_msal:
            client = MicrosoftTeamsClient()
            await client.initialize()
            
            assert client.app is not None
            mock_msal.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_meeting_id(self):
        """Test meeting ID extraction from URL."""
        client = MicrosoftTeamsClient()
        
        # Test valid URLs
        assert client.extract_meeting_id("https://teams.microsoft.com/l/meetup-join/test-meeting-id") == "test-meeting-id"
        assert client.extract_meeting_id("https://teams.live.com/meet/test-meeting") == "test-meeting"
        
        # Test invalid URL
        assert client.extract_meeting_id("https://invalid-url.com") is None
    
    @pytest.mark.asyncio
    async def test_authenticate_success(self):
        """Test successful authentication."""
        client = MicrosoftTeamsClient()
        await client.initialize()
        
        with patch.object(client.app, 'acquire_token_for_client') as mock_acquire:
            mock_acquire.return_value = {'access_token': 'test-token'}
            
            result = await client.authenticate()
            assert result is True
            assert client.access_token == 'test-token'


class TestTranscriptionService:
    """Test transcription service functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test transcription service initialization."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            service = TranscriptionService()
            await service.initialize()
            
            assert service.client is not None
            mock_openai.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_convert_to_wav(self):
        """Test audio conversion to WAV format."""
        service = TranscriptionService()
        
        # Test with sample audio data
        audio_data = b'\x00\x01\x02\x03' * 1000  # Simulate audio data
        wav_data = service._convert_to_wav(audio_data)
        
        assert isinstance(wav_data, bytes)
        assert len(wav_data) > len(audio_data)  # WAV header adds size
    
    @pytest.mark.asyncio
    async def test_transcribe_audio(self):
        """Test audio transcription."""
        service = TranscriptionService()
        await service.initialize()
        
        with patch.object(service.client.audio.transcriptions, 'create') as mock_transcribe:
            mock_transcribe.return_value = "This is a test transcription"
            
            # Create mock WAV data
            wav_data = b'\x00\x01\x02\x03' * 1000
            
            result = await service._transcribe_audio(wav_data)
            
            assert result == "This is a test transcription"
            mock_transcribe.assert_called_once()


class TestSummarizationService:
    """Test summarization service functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test summarization service initialization."""
        with patch('langchain.chat_models.ChatOpenAI') as mock_chat:
            service = SummarizationService()
            await service.initialize()
            
            assert service.llm is not None
            assert service.is_initialized is True
            mock_chat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_combine_transcription_chunks(self, sample_transcription_chunks):
        """Test combining transcription chunks."""
        service = SummarizationService()
        
        combined = service._combine_transcription_chunks(sample_transcription_chunks)
        
        assert isinstance(combined, str)
        assert "Hello everyone" in combined
        assert "project updates" in combined
        assert "budget has been approved" in combined
    
    @pytest.mark.asyncio
    async def test_summarize_meeting(self, sample_meeting, sample_transcription_chunks):
        """Test meeting summarization."""
        service = SummarizationService()
        await service.initialize()
        
        with patch.object(service, '_analyze_transcription') as mock_analyze:
            from services.meeting_agent.models import MeetingAnalysis
            
            mock_analysis = MeetingAnalysis(
                topics_discussed=["Project updates", "Budget review"],
                key_decisions=["Approved Q1 budget"],
                action_items=[{"description": "Send report", "assignee": "john@example.com"}],
                summary_text="Meeting covered project updates and budget review.",
                sentiment="positive",
                duration_minutes=30
            )
            mock_analyze.return_value = mock_analysis
            
            summary = await service.summarize_meeting(sample_meeting, sample_transcription_chunks)
            
            assert summary.meeting_id == sample_meeting.id
            assert len(summary.topics_discussed) == 2
            assert len(summary.key_decisions) == 1
            assert len(summary.action_items) == 1
            assert summary.sentiment == "positive"


class TestNotificationService:
    """Test notification service functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test notification service initialization."""
        with patch('slack_sdk.web.async_client.AsyncWebClient') as mock_slack:
            with patch('smtplib.SMTP') as mock_smtp:
                service = NotificationService()
                await service.initialize()
                
                assert service.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_prepare_summary_notification(self, sample_meeting):
        """Test preparing summary notification."""
        service = NotificationService()
        
        from services.meeting_agent.models import MeetingSummary, ActionItem
        
        summary = MeetingSummary(
            meeting_id=sample_meeting.id,
            topics_discussed=["Project updates"],
            key_decisions=["Approved budget"],
            action_items=[ActionItem(description="Send report", assignee="john@example.com")],
            summary_text="Test summary",
            sentiment="positive",
            duration_minutes=30
        )
        
        notification = await service._prepare_summary_notification(sample_meeting, summary)
        
        assert notification.meeting_id == sample_meeting.id
        assert notification.notification_type == "summary"
        assert "Test Meeting" in notification.subject
        assert "Test summary" in notification.content


class TestMeetingScheduler:
    """Test meeting scheduler functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test scheduler initialization."""
        scheduler = MeetingScheduler()
        
        with patch.object(scheduler, '_initialize_calendar_service') as mock_calendar:
            with patch.object(scheduler.meeting_clients, 'values') as mock_clients:
                mock_clients.return_value = [AsyncMock(), AsyncMock(), AsyncMock()]
                
                await scheduler.initialize()
                
                assert scheduler.scheduler is not None
                mock_calendar.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_is_ai_attendee(self):
        """Test AI attendee detection."""
        scheduler = MeetingScheduler()
        
        from services.meeting_agent.models import CalendarEvent, MeetingParticipant
        
        event = CalendarEvent(
            event_id="test-event",
            title="Test Event",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(hours=1),
            organizer_email="organizer@example.com",
            attendees=[
                MeetingParticipant(email="ai@example.com", name="AI Assistant"),
                MeetingParticipant(email="user@example.com", name="User")
            ],
            calendar_id="primary"
        )
        
        # Mock settings
        with patch('services.meeting_agent.scheduler.settings') as mock_settings:
            mock_settings.ai_email = "ai@example.com"
            
            result = scheduler._is_ai_attendee(event)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_should_join_meeting(self):
        """Test meeting join timing logic."""
        scheduler = MeetingScheduler()
        
        from services.meeting_agent.models import CalendarEvent
        
        # Meeting starting in 3 minutes (within buffer)
        event = CalendarEvent(
            event_id="test-event",
            title="Test Event",
            start_time=datetime.utcnow() + timedelta(minutes=3),
            end_time=datetime.utcnow() + timedelta(hours=1),
            organizer_email="organizer@example.com",
            attendees=[],
            calendar_id="primary"
        )
        
        result = scheduler._should_join_meeting(event)
        assert result is True
        
        # Meeting starting in 10 minutes (outside buffer)
        event.start_time = datetime.utcnow() + timedelta(minutes=10)
        result = scheduler._should_join_meeting(event)
        assert result is False


@pytest.mark.integration
class TestMeetingAgentIntegration:
    """Integration tests for the complete meeting agent workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_meeting_workflow(self, sample_meeting, sample_transcription_chunks):
        """Test the complete meeting workflow."""
        from services.meeting_agent.main import MeetingAgentService
        
        service = MeetingAgentService()
        
        with patch.object(service.scheduler, 'initialize') as mock_scheduler_init:
            with patch.object(service.transcription_service, 'initialize') as mock_transcription_init:
                with patch.object(service.summarization_service, 'initialize') as mock_summarization_init:
                    with patch.object(service.notification_service, 'initialize') as mock_notification_init:
                        
                        await service.initialize()
                        
                        mock_scheduler_init.assert_called_once()
                        mock_transcription_init.assert_called_once()
                        mock_summarization_init.assert_called_once()
                        mock_notification_init.assert_called_once()
                        
                        assert service.is_running is False
    
    @pytest.mark.asyncio
    async def test_meeting_status_endpoints(self):
        """Test meeting status API endpoints."""
        from services.meeting_agent.main import meeting_agent_service
        
        status = await meeting_agent_service.get_status()
        
        assert 'service_status' in status
        assert 'active_meetings_count' in status
        assert 'services' in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
