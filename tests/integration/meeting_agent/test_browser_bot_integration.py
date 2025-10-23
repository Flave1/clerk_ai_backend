"""
Integration tests for browser bot functionality.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from uuid import uuid4

from services.meeting_agent.scheduler import MeetingScheduler
from services.meeting_agent.main import MeetingAgentService
from shared.schemas import Meeting, MeetingPlatform, MeetingStatus


class TestBrowserBotIntegration:
    """Integration tests for browser bot functionality."""
    
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
    async def test_full_meeting_workflow_with_browser_bot(self, scheduler):
        """Test complete meeting workflow with browser bot."""
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
        
        # Mock Docker client
        mock_container = Mock()
        mock_container.id = 'test-container-id'
        
        with patch('docker.from_env') as mock_docker:
            mock_docker_client = Mock()
            mock_docker_client.containers.run.return_value = mock_container
            mock_docker.return_value = mock_docker_client
            
            scheduler.docker_client = mock_docker_client
            
            # Test meeting join with bot launch
            await scheduler._join_meeting(meeting)
            
            # Verify bot was launched
            assert meeting.status == MeetingStatus.ACTIVE
            assert str(meeting.id) in scheduler.active_bot_containers
            
            # Test meeting leave with bot cleanup
            await scheduler._leave_meeting(meeting)
            
            # Verify bot was stopped
            assert meeting.status == MeetingStatus.ENDED
            assert str(meeting.id) not in scheduler.active_bot_containers
    
    @pytest.mark.asyncio
    async def test_browser_bot_capacity_management(self, scheduler):
        """Test browser bot capacity management."""
        # Fill up to capacity
        for i in range(scheduler.max_concurrent_bots):
            scheduler.active_bot_containers[f'meeting-{i}'] = {
                'container_id': f'container-{i}',
                'session_id': f'session-{i}',
                'started_at': datetime.utcnow(),
                'platform': 'google_meet'
            }
        
        # Try to launch another bot
        result = await scheduler.launch_browser_bot(
            meeting_id='overflow-meeting',
            platform='google_meet',
            meeting_url='https://meet.google.com/overflow',
            bot_name='Overflow Bot'
        )
        
        # Should fail due to capacity
        assert result is False
        assert 'overflow-meeting' not in scheduler.active_bot_containers
    
    @pytest.mark.asyncio
    async def test_browser_bot_error_handling(self, scheduler):
        """Test browser bot error handling."""
        # Mock Docker client to raise an error
        with patch('docker.from_env') as mock_docker:
            mock_docker_client = Mock()
            mock_docker_client.containers.run.side_effect = Exception('Docker error')
            mock_docker.return_value = mock_docker_client
            
            scheduler.docker_client = mock_docker_client
            
            # Try to launch bot
            result = await scheduler.launch_browser_bot(
                meeting_id='error-meeting',
                platform='google_meet',
                meeting_url='https://meet.google.com/error',
                bot_name='Error Bot'
            )
            
            # Should handle error gracefully
            assert result is False
            assert 'error-meeting' not in scheduler.active_bot_containers
    
    @pytest.mark.asyncio
    async def test_browser_bot_cleanup_on_service_stop(self, scheduler):
        """Test browser bot cleanup when service stops."""
        # Add some bot containers
        scheduler.active_bot_containers['meeting-1'] = {
            'container_id': 'container-1',
            'session_id': 'session-1',
            'started_at': datetime.utcnow(),
            'platform': 'google_meet'
        }
        scheduler.active_bot_containers['meeting-2'] = {
            'container_id': 'container-2',
            'session_id': 'session-2',
            'started_at': datetime.utcnow(),
            'platform': 'zoom'
        }
        
        # Mock Docker client
        mock_container = Mock()
        
        with patch('docker.from_env') as mock_docker:
            mock_docker_client = Mock()
            mock_docker_client.containers.get.return_value = mock_container
            mock_docker.return_value = mock_docker_client
            
            scheduler.docker_client = mock_docker_client
            
            # Cleanup all bot containers
            await scheduler.cleanup_bot_containers()
            
            # Verify all containers were stopped
            assert len(scheduler.active_bot_containers) == 0
            assert mock_docker_client.containers.get.call_count == 2
            assert mock_container.stop.call_count == 2
    
    @pytest.mark.asyncio
    async def test_browser_bot_status_monitoring(self, service):
        """Test browser bot status monitoring."""
        # Mock active containers
        service.scheduler.get_active_bot_containers.return_value = {
            'meeting-1': {
                'container_id': 'container-1',
                'session_id': 'session-1',
                'started_at': datetime.utcnow(),
                'platform': 'google_meet'
            },
            'meeting-2': {
                'container_id': 'container-2',
                'session_id': 'session-2',
                'started_at': datetime.utcnow(),
                'platform': 'zoom'
            }
        }
        
        # Get status
        status = await service.get_browser_bot_status()
        
        # Verify status information
        assert status['active_bots_count'] == 2
        assert status['browser_bot_enabled'] is True
        assert status['max_concurrent_bots'] == 5
        assert len(status['active_bots']) == 2
        assert 'meeting-1' in status['active_bots']
        assert 'meeting-2' in status['active_bots']
    
    @pytest.mark.asyncio
    async def test_browser_bot_platform_specific_handling(self, scheduler):
        """Test browser bot handling for different platforms."""
        platforms = ['google_meet', 'zoom', 'teams']
        
        for platform in platforms:
            meeting_id = f'meeting-{platform}'
            
            # Mock Docker client
            mock_container = Mock()
            mock_container.id = f'container-{platform}'
            
            with patch('docker.from_env') as mock_docker:
                mock_docker_client = Mock()
                mock_docker_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_docker_client
                
                scheduler.docker_client = mock_docker_client
                
                # Launch bot for this platform
                result = await scheduler.launch_browser_bot(
                    meeting_id=meeting_id,
                    platform=platform,
                    meeting_url=f'https://{platform}.com/test',
                    bot_name=f'{platform.title()} Bot'
                )
                
                # Verify bot was launched
                assert result is True
                assert meeting_id in scheduler.active_bot_containers
                assert scheduler.active_bot_containers[meeting_id]['platform'] == platform
    
    @pytest.mark.asyncio
    async def test_browser_bot_environment_variables(self, scheduler):
        """Test browser bot environment variable configuration."""
        # Mock Docker client
        mock_container = Mock()
        mock_container.id = 'test-container-id'
        
        with patch('docker.from_env') as mock_docker:
            mock_docker_client = Mock()
            mock_docker_client.containers.run.return_value = mock_container
            mock_docker.return_value = mock_docker_client
            
            scheduler.docker_client = mock_docker_client
            
            # Launch bot
            await scheduler.launch_browser_bot(
                meeting_id='env-test-meeting',
                platform='google_meet',
                meeting_url='https://meet.google.com/env-test',
                bot_name='Env Test Bot'
            )
            
            # Verify environment variables were passed correctly
            call_args = mock_docker_client.containers.run.call_args
            env_vars = call_args[1]['environment']
            
            assert env_vars['MEETING_URL'] == 'https://meet.google.com/env-test'
            assert env_vars['BOT_NAME'] == 'Env Test Bot'
            assert env_vars['PLATFORM'] == 'google_meet'
            assert env_vars['MEETING_ID'] == 'env-test-meeting'
            assert 'SESSION_ID' in env_vars
            assert env_vars['AUDIO_SAMPLE_RATE'] == '16000'
            assert env_vars['AUDIO_CHANNELS'] == '1'
    
    @pytest.mark.asyncio
    async def test_browser_bot_concurrent_launches(self, scheduler):
        """Test concurrent browser bot launches."""
        # Mock Docker client
        mock_container = Mock()
        mock_container.id = 'test-container-id'
        
        with patch('docker.from_env') as mock_docker:
            mock_docker_client = Mock()
            mock_docker_client.containers.run.return_value = mock_container
            mock_docker.return_value = mock_docker_client
            
            scheduler.docker_client = mock_docker_client
            
            # Launch multiple bots concurrently
            tasks = []
            for i in range(3):
                task = scheduler.launch_browser_bot(
                    meeting_id=f'concurrent-meeting-{i}',
                    platform='google_meet',
                    meeting_url=f'https://meet.google.com/concurrent-{i}',
                    bot_name=f'Concurrent Bot {i}'
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert all(results)
            assert len(scheduler.active_bot_containers) == 3
            
            # Verify all containers are tracked
            for i in range(3):
                assert f'concurrent-meeting-{i}' in scheduler.active_bot_containers
