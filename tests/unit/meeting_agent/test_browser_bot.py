"""
Unit tests for browser bot functionality in meeting agent.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from uuid import uuid4

from services.meeting_agent.scheduler import MeetingScheduler
from services.meeting_agent.main import MeetingAgentService
from shared.schemas import Meeting, MeetingPlatform, MeetingStatus


class TestBrowserBotScheduler:
    """Test browser bot functionality in MeetingScheduler."""
    
    @pytest.fixture
    def scheduler(self):
        """Create a MeetingScheduler instance for testing."""
        scheduler = MeetingScheduler()
        scheduler.browser_bot_enabled = True
        scheduler.bot_image = 'test-bot-image'
        scheduler.bot_container_cpu = '1024'
        scheduler.bot_container_memory = '2048'
        scheduler.max_concurrent_bots = 3
        return scheduler
    
    @pytest.fixture
    def mock_meeting(self):
        """Create a mock meeting for testing."""
        return Meeting(
            id=uuid4(),
            platform=MeetingPlatform.GOOGLE_MEET,
            meeting_url='https://meet.google.com/test-meeting',
            meeting_id_external='test-meeting-id',
            title='Test Meeting',
            description='Test meeting description',
            start_time=datetime.utcnow() + timedelta(minutes=5),
            end_time=datetime.utcnow() + timedelta(minutes=35),
            organizer_email='organizer@example.com',
            participants=[],
            ai_email='ai@example.com'
        )
    
    @pytest.mark.asyncio
    async def test_initialize_browser_bot_orchestration(self, scheduler):
        """Test initialization of browser bot orchestration."""
        with patch('docker.from_env') as mock_docker, \
             patch('boto3.client') as mock_boto3:
            
            mock_docker_client = Mock()
            mock_docker.return_value = mock_docker_client
            
            mock_ecs_client = Mock()
            mock_sqs_client = Mock()
            mock_boto3.side_effect = [mock_ecs_client, mock_sqs_client]
            
            await scheduler._initialize_browser_bot_orchestration()
            
            assert scheduler.docker_client == mock_docker_client
            assert scheduler.ecs_client == mock_ecs_client
            assert scheduler.sqs_client == mock_sqs_client
    
    @pytest.mark.asyncio
    async def test_launch_browser_bot_disabled(self, scheduler):
        """Test launching browser bot when disabled."""
        scheduler.browser_bot_enabled = False
        
        result = await scheduler.launch_browser_bot(
            meeting_id='test-meeting',
            platform='google_meet',
            meeting_url='https://meet.google.com/test',
            bot_name='Test Bot'
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_launch_browser_bot_at_capacity(self, scheduler):
        """Test launching browser bot when at capacity."""
        # Fill up to capacity
        for i in range(scheduler.max_concurrent_bots):
            scheduler.active_bot_containers[f'meeting-{i}'] = {
                'container_id': f'container-{i}',
                'session_id': f'session-{i}',
                'started_at': datetime.utcnow(),
                'platform': 'google_meet'
            }
        
        result = await scheduler.launch_browser_bot(
            meeting_id='test-meeting',
            platform='google_meet',
            meeting_url='https://meet.google.com/test',
            bot_name='Test Bot'
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_launch_docker_bot_success(self, scheduler):
        """Test successful Docker bot launch."""
        mock_container = Mock()
        mock_container.id = 'test-container-id'
        
        with patch.object(scheduler, 'docker_client') as mock_docker_client:
            mock_docker_client.containers.run.return_value = mock_container
            
            result = await scheduler._launch_docker_bot(
                meeting_id='test-meeting',
                session_id='test-session',
                env_vars={'MEETING_URL': 'https://meet.google.com/test', 'PLATFORM': 'google_meet'}
            )
            
            assert result is True
            assert 'test-meeting' in scheduler.active_bot_containers
            assert scheduler.active_bot_containers['test-meeting']['container_id'] == 'test-container-id'
    
    @pytest.mark.asyncio
    async def test_launch_docker_bot_failure(self, scheduler):
        """Test Docker bot launch failure."""
        with patch.object(scheduler, 'docker_client') as mock_docker_client:
            mock_docker_client.containers.run.side_effect = Exception('Docker error')
            
            result = await scheduler._launch_docker_bot(
                meeting_id='test-meeting',
                session_id='test-session',
                env_vars={'MEETING_URL': 'https://meet.google.com/test', 'PLATFORM': 'google_meet'}
            )
            
            assert result is False
            assert 'test-meeting' not in scheduler.active_bot_containers
    
    @pytest.mark.asyncio
    async def test_launch_ecs_bot_success(self, scheduler):
        """Test successful ECS bot launch."""
        mock_response = {
            'tasks': [{'taskArn': 'arn:aws:ecs:us-east-1:123456789012:task/test-task'}]
        }
        
        with patch.object(scheduler, 'ecs_client') as mock_ecs_client:
            mock_ecs_client.run_task.return_value = mock_response
            
            result = await scheduler._launch_ecs_bot(
                meeting_id='test-meeting',
                session_id='test-session',
                env_vars={'MEETING_URL': 'https://meet.google.com/test', 'PLATFORM': 'google_meet'}
            )
            
            assert result is True
            assert 'test-meeting' in scheduler.active_bot_containers
            assert scheduler.active_bot_containers['test-meeting']['task_arn'] == 'arn:aws:ecs:us-east-1:123456789012:task/test-task'
    
    @pytest.mark.asyncio
    async def test_launch_ecs_bot_no_tasks(self, scheduler):
        """Test ECS bot launch with no tasks returned."""
        mock_response = {'tasks': []}
        
        with patch.object(scheduler, 'ecs_client') as mock_ecs_client:
            mock_ecs_client.run_task.return_value = mock_response
            
            result = await scheduler._launch_ecs_bot(
                meeting_id='test-meeting',
                session_id='test-session',
                env_vars={'MEETING_URL': 'https://meet.google.com/test', 'PLATFORM': 'google_meet'}
            )
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_stop_browser_bot_docker(self, scheduler):
        """Test stopping Docker browser bot."""
        # Add a bot container
        scheduler.active_bot_containers['test-meeting'] = {
            'container_id': 'test-container-id',
            'container_name': 'test-container',
            'session_id': 'test-session',
            'started_at': datetime.utcnow(),
            'platform': 'google_meet'
        }
        
        mock_container = Mock()
        
        with patch.object(scheduler, 'docker_client') as mock_docker_client:
            mock_docker_client.containers.get.return_value = mock_container
            
            result = await scheduler.stop_browser_bot('test-meeting')
            
            assert result is True
            assert 'test-meeting' not in scheduler.active_bot_containers
            mock_container.stop.assert_called_once_with(timeout=10)
    
    @pytest.mark.asyncio
    async def test_stop_browser_bot_ecs(self, scheduler):
        """Test stopping ECS browser bot."""
        # Add a bot container
        scheduler.active_bot_containers['test-meeting'] = {
            'task_arn': 'arn:aws:ecs:us-east-1:123456789012:task/test-task',
            'session_id': 'test-session',
            'started_at': datetime.utcnow(),
            'platform': 'google_meet'
        }
        
        with patch.object(scheduler, 'ecs_client') as mock_ecs_client:
            result = await scheduler.stop_browser_bot('test-meeting')
            
            assert result is True
            assert 'test-meeting' not in scheduler.active_bot_containers
            mock_ecs_client.stop_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_browser_bot_not_found(self, scheduler):
        """Test stopping non-existent browser bot."""
        result = await scheduler.stop_browser_bot('non-existent-meeting')
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_active_bot_containers(self, scheduler):
        """Test getting active bot containers."""
        # Add some bot containers
        scheduler.active_bot_containers['meeting-1'] = {'container_id': 'container-1'}
        scheduler.active_bot_containers['meeting-2'] = {'container_id': 'container-2'}
        
        result = await scheduler.get_active_bot_containers()
        
        assert len(result) == 2
        assert 'meeting-1' in result
        assert 'meeting-2' in result
    
    @pytest.mark.asyncio
    async def test_cleanup_bot_containers(self, scheduler):
        """Test cleanup of all bot containers."""
        # Add some bot containers
        scheduler.active_bot_containers['meeting-1'] = {'container_id': 'container-1'}
        scheduler.active_bot_containers['meeting-2'] = {'container_id': 'container-2'}
        
        with patch.object(scheduler, 'stop_browser_bot', return_value=True) as mock_stop:
            await scheduler.cleanup_bot_containers()
            
            assert mock_stop.call_count == 2
            # The containers should be removed by the actual stop_browser_bot method
            # Since we're mocking it, we need to manually verify the calls
            assert mock_stop.call_args_list[0][0][0] == 'meeting-1'
            assert mock_stop.call_args_list[1][0][0] == 'meeting-2'


class TestBrowserBotMainService:
    """Test browser bot functionality in MeetingAgentService."""
    
    @pytest.fixture
    def service(self):
        """Create a MeetingAgentService instance for testing."""
        service = MeetingAgentService()
        service.scheduler = Mock()
        service.scheduler.launch_browser_bot = AsyncMock(return_value=True)
        service.scheduler.stop_browser_bot = AsyncMock(return_value=True)
        service.scheduler.get_active_bot_containers = AsyncMock(return_value={})
        service.scheduler.browser_bot_enabled = True
        service.scheduler.max_concurrent_bots = 5
        return service
    
    @pytest.mark.asyncio
    async def test_launch_browser_bot_success(self, service):
        """Test successful browser bot launch."""
        result = await service.launch_browser_bot(
            meeting_id='test-meeting',
            platform='google_meet',
            meeting_url='https://meet.google.com/test',
            bot_name='Test Bot'
        )
        
        assert result is True
        service.scheduler.launch_browser_bot.assert_called_once_with(
            meeting_id='test-meeting',
            platform='google_meet',
            meeting_url='https://meet.google.com/test',
            bot_name='Test Bot'
        )
    
    @pytest.mark.asyncio
    async def test_launch_browser_bot_no_scheduler_method(self, service):
        """Test browser bot launch when scheduler doesn't have the method."""
        delattr(service.scheduler, 'launch_browser_bot')
        
        result = await service.launch_browser_bot(
            meeting_id='test-meeting',
            platform='google_meet',
            meeting_url='https://meet.google.com/test'
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_stop_browser_bot_success(self, service):
        """Test successful browser bot stop."""
        result = await service.stop_browser_bot('test-meeting')
        
        assert result is True
        service.scheduler.stop_browser_bot.assert_called_once_with('test-meeting')
    
    @pytest.mark.asyncio
    async def test_get_browser_bot_status(self, service):
        """Test getting browser bot status."""
        service.scheduler.get_active_bot_containers.return_value = {
            'meeting-1': {'container_id': 'container-1'},
            'meeting-2': {'container_id': 'container-2'}
        }
        
        result = await service.get_browser_bot_status()
        
        assert result['active_bots_count'] == 2
        assert result['browser_bot_enabled'] is True
        assert result['max_concurrent_bots'] == 5
        assert len(result['active_bots']) == 2
    
    @pytest.mark.asyncio
    async def test_get_browser_bot_status_no_scheduler_method(self, service):
        """Test getting browser bot status when scheduler doesn't have the method."""
        delattr(service.scheduler, 'get_active_bot_containers')
        
        result = await service.get_browser_bot_status()
        
        assert result['active_bots_count'] == 0
        assert result['browser_bot_enabled'] is False
        assert result['max_concurrent_bots'] == 0


class TestBrowserBotIntegration:
    """Integration tests for browser bot functionality."""
    
    @pytest.mark.asyncio
    async def test_meeting_join_with_browser_bot(self):
        """Test joining a meeting with browser bot integration."""
        scheduler = MeetingScheduler()
        scheduler.browser_bot_enabled = True
        
        # Mock the browser bot launch
        with patch.object(scheduler, 'launch_browser_bot', return_value=True) as mock_launch:
            meeting = Meeting(
                id=uuid4(),
                platform=MeetingPlatform.GOOGLE_MEET,
                meeting_url='https://meet.google.com/test-meeting',
                meeting_id_external='test-meeting-id',
                title='Test Meeting',
                description='Test meeting description',
                start_time=datetime.utcnow() + timedelta(minutes=5),
                end_time=datetime.utcnow() + timedelta(minutes=35),
                organizer_email='organizer@example.com',
                participants=[],
                ai_email='ai@example.com'
            )
            
            await scheduler._join_meeting(meeting)
            
            mock_launch.assert_called_once()
            assert meeting.status == MeetingStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_meeting_leave_with_browser_bot(self):
        """Test leaving a meeting with browser bot cleanup."""
        scheduler = MeetingScheduler()
        scheduler.browser_bot_enabled = True
        
        # Add a bot container
        meeting_id = str(uuid4())
        scheduler.active_bot_containers[meeting_id] = {
            'container_id': 'test-container-id',
            'session_id': 'test-session',
            'started_at': datetime.utcnow(),
            'platform': 'google_meet'
        }
        
        meeting = Meeting(
            id=meeting_id,  # Use the same ID as the bot container
            platform=MeetingPlatform.GOOGLE_MEET,
            meeting_url='https://meet.google.com/test-meeting',
            meeting_id_external='test-meeting-id',
            title='Test Meeting',
            description='Test meeting description',
            start_time=datetime.utcnow() - timedelta(minutes=30),
            end_time=datetime.utcnow() + timedelta(minutes=5),
            organizer_email='organizer@example.com',
            participants=[],
            ai_email='ai@example.com'
        )
        
        # Mock the browser bot stop
        with patch.object(scheduler, 'stop_browser_bot', return_value=True) as mock_stop:
            await scheduler._leave_meeting(meeting)
            
            mock_stop.assert_called_once()
            assert meeting.status == MeetingStatus.ENDED
