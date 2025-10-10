"""
Unit tests for Microsoft Teams meeting tool functionality.

Tests the create_teams_meeting tool, bot joining, and audio capture.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from uuid import uuid4

from services.meeting_agent.teams_client import MicrosoftTeamsClient, create_teams_client
from shared.schemas import (
    Meeting, MeetingPlatform, MeetingStatus, MeetingParticipant
)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch('services.meeting_agent.teams_client.settings') as mock:
        mock.ms_client_id = "test-client-id"
        mock.ms_client_secret = "test-client-secret"
        mock.ms_tenant_id = "test-tenant-id"
        mock.api_base_url = "https://test-api.example.com"
        mock.ai_email = "test@example.com"
        yield mock


@pytest.fixture
def sample_meeting():
    """Create a sample Teams meeting for testing."""
    return Meeting(
        id=str(uuid4()),
        platform=MeetingPlatform.MICROSOFT_TEAMS,
        meeting_url="https://teams.microsoft.com/l/meetup-join/test-meeting-id-123",
        meeting_id_external="test-meeting-id-123",
        title="Test Teams Meeting",
        description="A test meeting for unit tests",
        start_time=datetime.utcnow() + timedelta(hours=1),
        end_time=datetime.utcnow() + timedelta(hours=2),
        organizer_email="organizer@example.com",
        participants=[
            MeetingParticipant(
                email="user1@example.com",
                name="User One",
                is_organizer=False,
                response_status="accepted"
            ),
            MeetingParticipant(
                email="user2@example.com",
                name="User Two",
                is_organizer=False,
                response_status="accepted"
            )
        ],
        status=MeetingStatus.SCHEDULED,
        ai_email="test@example.com",
        audio_enabled=True,
        video_enabled=True,
        recording_enabled=False
    )


@pytest.fixture
def mock_graph_api_response():
    """Mock Graph API response for meeting creation."""
    return {
        'id': 'test-graph-meeting-id',
        'joinWebUrl': 'https://teams.microsoft.com/l/meetup-join/test-meeting-id-123',
        'subject': 'Test Meeting',
        'startDateTime': '2025-10-10T14:00:00.000Z',
        'endDateTime': '2025-10-10T15:00:00.000Z',
        'joinMeetingIdSettings': {
            'joinMeetingId': '123456789'
        }
    }


class TestTeamsClientInitialization:
    """Test Teams client initialization and basic functionality."""
    
    @pytest.mark.asyncio
    async def test_client_factory(self, mock_settings):
        """Test the create_teams_client factory function."""
        client = create_teams_client()
        
        assert isinstance(client, MicrosoftTeamsClient)
        assert client.app is None  # Not initialized yet
        assert client.access_token is None
        assert not client.is_joined
    
    @pytest.mark.asyncio
    async def test_initialize(self, mock_settings):
        """Test client initialization with MSAL."""
        with patch('services.meeting_agent.teams_client.ConfidentialClientApplication') as mock_msal:
            with patch('aiohttp.ClientSession') as mock_session:
                client = MicrosoftTeamsClient()
                await client.initialize()
                
                # Verify MSAL was initialized
                mock_msal.assert_called_once_with(
                    client_id="test-client-id",
                    client_credential="test-client-secret",
                    authority="https://login.microsoftonline.com/test-tenant-id"
                )
                
                # Verify session was created
                mock_session.assert_called_once()
                assert client.app is not None
                assert client.session is not None
    
    @pytest.mark.asyncio
    async def test_authentication_with_client_credentials(self, mock_settings):
        """Test authentication using client credentials flow."""
        client = MicrosoftTeamsClient()
        
        # Mock MSAL app
        mock_app = MagicMock()
        mock_app.acquire_token_for_client.return_value = {
            'access_token': 'test-access-token-123'
        }
        client.app = mock_app
        
        result = await client.authenticate()
        
        assert result is True
        assert client.access_token == 'test-access-token-123'
        mock_app.acquire_token_for_client.assert_called_once()
    
    def test_extract_meeting_id(self, mock_settings):
        """Test extracting meeting ID from various URL formats."""
        client = MicrosoftTeamsClient()
        
        # Test valid URLs
        url1 = "https://teams.microsoft.com/l/meetup-join/abc123xyz"
        assert client.extract_meeting_id(url1) == "abc123xyz"
        
        url2 = "https://teams.live.com/meet/test-meeting-123"
        assert client.extract_meeting_id(url2) == "test-meeting-123"
        
        # Test invalid URL
        invalid_url = "https://example.com/not-a-teams-url"
        assert client.extract_meeting_id(invalid_url) is None


class TestMeetingCreation:
    """Test Teams meeting creation functionality."""
    
    @pytest.mark.asyncio
    async def test_create_meeting_success(self, mock_settings, mock_graph_api_response):
        """Test successful meeting creation via Graph API."""
        client = MicrosoftTeamsClient()
        client.access_token = "test-token"
        
        # Mock session and Graph API response
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value=mock_graph_api_response)
        
        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        # Mock DAO
        with patch('services.meeting_agent.teams_client.DynamoDBDAO') as mock_dao:
            mock_dao_instance = AsyncMock()
            mock_dao_instance.initialize = AsyncMock()
            mock_dao_instance.create_meeting = AsyncMock()
            mock_dao.return_value = mock_dao_instance
            
            # Create meeting
            result = await client.create_meeting(
                title="Test Meeting",
                start_time="2025-10-10T14:00:00Z",
                end_time="2025-10-10T15:00:00Z",
                attendees=["user1@example.com", "user2@example.com"],
                description="Test meeting description"
            )
            
            # Verify result
            assert result["success"] is True
            assert "meeting" in result
            assert result["meeting"]["title"] == "Test Meeting"
            assert result["meeting"]["join_url"] == mock_graph_api_response["joinWebUrl"]
            assert len(result["meeting"]["attendees"]) == 2
            
            # Verify DAO was called
            mock_dao_instance.create_meeting.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_meeting_invalid_datetime(self, mock_settings):
        """Test meeting creation with invalid datetime format."""
        client = MicrosoftTeamsClient()
        client.access_token = "test-token"
        
        result = await client.create_meeting(
            title="Test Meeting",
            start_time="invalid-datetime",
            end_time="2025-10-10T15:00:00Z",
            attendees=["user@example.com"]
        )
        
        assert result["success"] is False
        assert "Invalid date/time format" in result["error"]
    
    @pytest.mark.asyncio
    async def test_create_meeting_no_access_token(self, mock_settings):
        """Test meeting creation without access token."""
        client = MicrosoftTeamsClient()
        
        # Mock failed authentication
        with patch.object(client, 'authenticate', return_value=False):
            result = await client.create_meeting(
                title="Test Meeting",
                start_time="2025-10-10T14:00:00Z",
                end_time="2025-10-10T15:00:00Z"
            )
            
            assert result["success"] is False
            assert "authentication" in result["error"].lower()


class TestBotJoining:
    """Test bot joining functionality."""
    
    @pytest.mark.asyncio
    async def test_join_meeting_success(self, mock_settings, sample_meeting):
        """Test successful bot joining."""
        client = MicrosoftTeamsClient()
        client.access_token = "test-token"
        
        # Mock app and session
        client.app = MagicMock()
        client.session = AsyncMock()
        
        # Mock _get_meeting_details_from_url
        with patch.object(client, '_get_meeting_details_from_url', return_value={
            'id': 'test-meeting-id',
            'joinWebUrl': sample_meeting.meeting_url,
            'subject': sample_meeting.title
        }):
            # Mock _join_meeting_as_bot
            with patch.object(client, '_join_meeting_as_bot', return_value=True):
                # Mock _start_audio_capture
                with patch.object(client, '_start_audio_capture', return_value=None):
                    result = await client.join_meeting(sample_meeting)
                    
                    assert result.success is True
                    assert result.meeting_id == sample_meeting.id
                    assert client.is_joined is True
                    assert client.current_meeting == sample_meeting
    
    @pytest.mark.asyncio
    async def test_join_meeting_no_authentication(self, mock_settings, sample_meeting):
        """Test joining without authentication."""
        client = MicrosoftTeamsClient()
        
        # Mock failed authentication
        with patch.object(client, 'authenticate', return_value=False):
            result = await client.join_meeting(sample_meeting)
            
            assert result.success is False
            assert "Authentication failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_join_meeting_no_meeting_id(self, mock_settings, sample_meeting):
        """Test joining with invalid meeting URL."""
        client = MicrosoftTeamsClient()
        client.app = MagicMock()
        client.access_token = "test-token"
        
        # Create meeting with invalid URL
        invalid_meeting = sample_meeting
        invalid_meeting.meeting_url = "https://invalid-url.com"
        invalid_meeting.meeting_id_external = None
        
        result = await client.join_meeting(invalid_meeting)
        
        assert result.success is False
        assert "Could not extract meeting ID" in result.error_message
    
    @pytest.mark.asyncio
    async def test_join_meeting_as_bot_communications_api(self, mock_settings, sample_meeting):
        """Test joining using Communications API."""
        client = MicrosoftTeamsClient()
        client.access_token = "test-token"
        
        # Mock successful API response
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={
            'id': 'call-id-123',
            'state': 'establishing'
        })
        
        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        meeting_details = {
            'id': sample_meeting.meeting_id_external,
            'joinWebUrl': sample_meeting.meeting_url,
            'chatInfo': {'threadId': 'thread-123'},
            'participants': {
                'organizer': {
                    'identity': {
                        'user': {'id': 'user-123'}
                    }
                }
            }
        }
        
        result = await client._join_meeting_as_bot(sample_meeting, meeting_details)
        
        assert result is True


class TestAudioCapture:
    """Test audio capture functionality."""
    
    @pytest.mark.asyncio
    async def test_start_audio_capture(self, mock_settings, sample_meeting):
        """Test starting audio capture."""
        client = MicrosoftTeamsClient()
        
        await client._start_audio_capture(sample_meeting)
        
        assert client.current_meeting == sample_meeting
        assert client.audio_stream is not None
    
    @pytest.mark.asyncio
    async def test_audio_stream_generation(self, mock_settings, sample_meeting):
        """Test audio stream chunk generation."""
        client = MicrosoftTeamsClient()
        client.is_joined = True
        
        # Get audio stream generator
        stream = client._capture_audio_stream()
        
        # Get a few chunks
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
            if len(chunks) >= 3:
                break
        
        # Verify chunks
        assert len(chunks) == 3
        for chunk in chunks:
            assert isinstance(chunk, bytes)
            assert len(chunk) == 3200  # 16kHz * 0.1s * 2 bytes
    
    @pytest.mark.asyncio
    async def test_get_audio_stream_not_joined(self, mock_settings):
        """Test getting audio stream when not joined."""
        client = MicrosoftTeamsClient()
        client.is_joined = False
        
        with pytest.raises(Exception) as exc_info:
            async for _ in client.get_audio_stream():
                break
        
        assert "Not joined" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_audio_stream_success(self, mock_settings, sample_meeting):
        """Test getting audio stream when joined."""
        client = MicrosoftTeamsClient()
        client.is_joined = True
        client.current_meeting = sample_meeting
        
        await client._start_audio_capture(sample_meeting)
        
        # Get a chunk
        chunk_count = 0
        async for chunk in client.get_audio_stream():
            assert isinstance(chunk, bytes)
            chunk_count += 1
            if chunk_count >= 2:
                break
        
        assert chunk_count == 2


class TestLeaveMeeting:
    """Test leaving meeting functionality."""
    
    @pytest.mark.asyncio
    async def test_leave_meeting(self, mock_settings, sample_meeting):
        """Test leaving a meeting."""
        client = MicrosoftTeamsClient()
        client.is_joined = True
        client.meeting_id = "test-meeting-123"
        client.current_meeting = sample_meeting
        client.access_token = "test-token"
        client.audio_stream = MagicMock()
        
        await client.leave_meeting()
        
        assert client.is_joined is False
        assert client.audio_stream is None
        assert client.current_meeting is None
        assert client.meeting_id is None


class TestParticipants:
    """Test participant management."""
    
    @pytest.mark.asyncio
    async def test_get_participants_from_stored_meeting(self, mock_settings, sample_meeting):
        """Test getting participants from stored meeting object."""
        client = MicrosoftTeamsClient()
        client.is_joined = True
        client.meeting_id = "test-meeting-123"
        client.current_meeting = sample_meeting
        
        participants = await client.get_participants()
        
        assert len(participants) == 2
        assert participants[0]['email'] == "user1@example.com"
        assert participants[1]['email'] == "user2@example.com"
    
    @pytest.mark.asyncio
    async def test_get_participants_not_joined(self, mock_settings):
        """Test getting participants when not joined."""
        client = MicrosoftTeamsClient()
        client.is_joined = False
        
        participants = await client.get_participants()
        
        assert len(participants) == 0


class TestCleanup:
    """Test resource cleanup."""
    
    @pytest.mark.asyncio
    async def test_cleanup(self, mock_settings, sample_meeting):
        """Test client cleanup."""
        client = MicrosoftTeamsClient()
        client.is_joined = True
        client.meeting_id = "test-meeting-123"
        client.current_meeting = sample_meeting
        
        # Mock session
        mock_session = AsyncMock()
        mock_session.close = AsyncMock()
        client.session = mock_session
        
        await client.cleanup()
        
        assert client.session is None
        mock_session.close.assert_called_once()


class TestToolIntegration:
    """Test integration with LangChain tool."""
    
    def test_create_teams_meeting_tool_simulation(self, mock_settings):
        """Test the create_teams_meeting tool (simulated)."""
        from services.rt_gateway.llm import ConversationManager
        
        # This would test the actual tool, but requires full LangChain setup
        # For now, we verify the tool is properly registered
        
        with patch('services.rt_gateway.llm.ChatAnthropic'):
            with patch('services.rt_gateway.llm.DynamoDBDAO'):
                manager = ConversationManager(
                    conversation_id="test-conv",
                    user_id="test-user"
                )
                
                # Verify create_teams_meeting tool exists
                tool_names = [tool.name for tool in manager.tools]
                assert 'create_teams_meeting' in tool_names


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

