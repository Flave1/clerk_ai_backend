"""
Scheduler service for detecting and managing meeting participation.

This module handles monitoring calendar events and triggering meeting joins
based on scheduled meetings.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import boto3
import docker
from uuid import uuid4

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from shared.config import get_settings
from shared.schemas import Meeting, MeetingPlatform, MeetingStatus, CalendarEvent
from .models import MeetingConfig
from .google_meet_client import create_google_meet_client
from .zoom_client import create_zoom_client
from .teams_client import create_teams_client

logger = logging.getLogger(__name__)
settings = get_settings()


class MeetingScheduler:
    """Scheduler for managing meeting participation."""
    
    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.calendar_service = None
        self.meeting_clients = {}
        self.active_meetings: Dict[str, Meeting] = {}
        self.config = MeetingConfig()
        self.is_running = False
        
        # Browser bot orchestration
        self.browser_bot_enabled = getattr(settings, 'browser_bot_enabled', True)
        self.bot_image = getattr(settings, 'bot_image', 'clerk-browser-bot')
        self.bot_container_cpu = getattr(settings, 'bot_container_cpu', '1024')
        self.bot_container_memory = getattr(settings, 'bot_container_memory', '2048')
        self.bot_join_timeout_sec = getattr(settings, 'bot_join_timeout_sec', 60)
        self.max_concurrent_bots = getattr(settings, 'max_concurrent_bots', 5)
        
        # Initialize container orchestration clients
        self.docker_client = None
        self.ecs_client = None
        self.sqs_client = None
        self.active_bot_containers: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self) -> None:
        """Initialize the meeting scheduler."""
        logger.info("Initializing meeting scheduler...")
        
        try:
            # Initialize scheduler
            self.scheduler = AsyncIOScheduler()
            
            # Initialize calendar service
            await self._initialize_calendar_service()
            
            # Initialize meeting clients
            self.meeting_clients = {
                MeetingPlatform.GOOGLE_MEET: create_google_meet_client(),
                MeetingPlatform.ZOOM: create_zoom_client(),
                MeetingPlatform.MICROSOFT_TEAMS: create_teams_client()
            }
            
            # Initialize clients
            for client in self.meeting_clients.values():
                await client.initialize()
            
            # Initialize browser bot orchestration
            if self.browser_bot_enabled:
                await self._initialize_browser_bot_orchestration()
            
            logger.info("Meeting scheduler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize meeting scheduler: {e}")
            raise
    
    async def _initialize_calendar_service(self) -> None:
        """Initialize Google Calendar service."""
        try:
            # Load credentials
            credentials_path = settings.google_oauth_client_config
            if not credentials_path:
                logger.warning("Google OAuth credentials not configured")
                return
            
            # Initialize calendar service
            # This is a simplified implementation
            # In production, implement proper OAuth2 flow
            logger.info("Calendar service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize calendar service: {e}")
            raise
    
    async def start(self) -> None:
        """Start the meeting scheduler."""
        logger.info("Starting meeting scheduler...")
        
        try:
            if not self.scheduler:
                await self.initialize()
            
            # Add job to check for meetings every minute
            self.scheduler.add_job(
                self._check_upcoming_meetings,
                trigger=IntervalTrigger(minutes=1),
                id='check_meetings',
                name='Check for upcoming meetings'
            )
            
            # Add job to monitor active meetings every 30 seconds
            self.scheduler.add_job(
                self._monitor_active_meetings,
                trigger=IntervalTrigger(seconds=30),
                id='monitor_meetings',
                name='Monitor active meetings'
            )
            
            # Start scheduler
            self.scheduler.start()
            self.is_running = True
            
            logger.info("Meeting scheduler started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start meeting scheduler: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the meeting scheduler."""
        logger.info("Stopping meeting scheduler...")
        
        try:
            if self.scheduler:
                self.scheduler.shutdown()
                self.scheduler = None
            
            # Leave all active meetings
            for meeting in self.active_meetings.values():
                await self._leave_meeting(meeting)
            
            # Cleanup clients
            for client in self.meeting_clients.values():
                await client.cleanup()
            
            self.is_running = False
            logger.info("Meeting scheduler stopped")
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    
    async def _check_upcoming_meetings(self) -> None:
        """Check for upcoming meetings and trigger joins."""
        try:
            logger.debug("Checking for upcoming meetings...")
            
            # Get calendar events
            calendar_events = await self._get_calendar_events()
            
            for event in calendar_events:
                # Check if AI email is in attendees
                if not self._is_ai_attendee(event):
                    continue
                
                # Check if meeting is starting soon
                if not self._should_join_meeting(event):
                    continue
                
                # Create meeting object
                meeting = await self._create_meeting_from_event(event)
                
                # Join meeting
                await self._join_meeting(meeting)
                
        except Exception as e:
            logger.error(f"Error checking upcoming meetings: {e}")
    
    async def _get_calendar_events(self) -> List[CalendarEvent]:
        """Get calendar events from Google Calendar."""
        try:
            # This is a simplified implementation
            # In production, use Google Calendar API
            
            # Mock calendar events for testing
            mock_events = [
                CalendarEvent(
                    event_id="test_event_1",
                    title="Test Google Meet",
                    description="Test meeting",
                    start_time=datetime.utcnow() + timedelta(minutes=2),
                    end_time=datetime.utcnow() + timedelta(minutes=32),
                    organizer_email="organizer@example.com",
                    attendees=[
                        {"email": settings.ai_email, "name": "AI Assistant", "is_organizer": False}
                    ],
                    meeting_url="https://meet.google.com/test-meeting-id",
                    platform=MeetingPlatform.GOOGLE_MEET,
                    calendar_id="primary"
                )
            ]
            
            return mock_events
            
        except Exception as e:
            logger.error(f"Failed to get calendar events: {e}")
            return []
    
    def _is_ai_attendee(self, event: CalendarEvent) -> bool:
        """Check if AI email is in the event attendees."""
        ai_email = settings.ai_email
        if not ai_email:
            return False
        
        for attendee in event.attendees:
            if attendee.email.lower() == ai_email.lower():
                return True
        
        return False
    
    def _should_join_meeting(self, event: CalendarEvent) -> bool:
        """Check if meeting should be joined based on timing."""
        now = datetime.utcnow()
        start_time = event.start_time
        
        # Join if meeting starts within the buffer time
        time_until_start = (start_time - now).total_seconds() / 60
        
        return 0 <= time_until_start <= self.config.join_buffer_minutes
    
    async def _create_meeting_from_event(self, event: CalendarEvent) -> Meeting:
        """Create a Meeting object from a CalendarEvent."""
        meeting = Meeting(
            platform=event.platform or MeetingPlatform.GOOGLE_MEET,
            meeting_url=event.meeting_url or "",
            meeting_id_external=self._extract_meeting_id(event.meeting_url or ""),
            title=event.title,
            description=event.description,
            start_time=event.start_time,
            end_time=event.end_time,
            organizer_email=event.organizer_email,
            participants=event.attendees,
            ai_email=settings.ai_email,
            calendar_event_id=event.event_id
        )
        
        return meeting
    
    def _extract_meeting_id(self, meeting_url: str) -> str:
        """Extract meeting ID from URL."""
        if not meeting_url:
            return ""
        
        # Simple extraction - in production, use proper URL parsing
        if "meet.google.com" in meeting_url:
            return meeting_url.split("/")[-1]
        elif "zoom.us" in meeting_url:
            return meeting_url.split("/")[-1]
        elif "teams.microsoft.com" in meeting_url:
            return meeting_url.split("/")[-1]
        
        return ""
    
    async def _join_meeting(self, meeting: Meeting) -> None:
        """Join a meeting using the appropriate client."""
        try:
            logger.info(f"Joining meeting: {meeting.title}")
            
            # Update meeting status
            meeting.status = MeetingStatus.JOINING
            meeting.join_attempts += 1
            meeting.last_join_attempt = datetime.utcnow()
            
            # Launch browser bot if enabled
            bot_launched = False
            if self.browser_bot_enabled and meeting.meeting_url:
                try:
                    bot_launched = await self.launch_browser_bot(
                        meeting_id=str(meeting.id),
                        platform=meeting.platform.value,
                        meeting_url=meeting.meeting_url,
                        bot_name=getattr(settings, 'ai_email', 'Clerk AI Bot')
                    )
                    if bot_launched:
                        logger.info(f"Browser bot launched for meeting: {meeting.title}")
                    else:
                        logger.warning(f"Failed to launch browser bot for meeting: {meeting.title}")
                except Exception as e:
                    logger.error(f"Error launching browser bot: {e}")
            
            # Get appropriate client for traditional joining (if needed)
            client = self.meeting_clients.get(meeting.platform)
            if client:
                # Join meeting using traditional client
                join_response = await client.join_meeting(meeting)
                
                if join_response.success:
                    meeting.status = MeetingStatus.ACTIVE
                    meeting.joined_at = datetime.utcnow()
                    self.active_meetings[meeting.id] = meeting
                    logger.info(f"Successfully joined meeting: {meeting.title}")
                else:
                    meeting.status = MeetingStatus.FAILED
                    meeting.error_message = join_response.error_message
                    logger.error(f"Failed to join meeting: {meeting.title} - {join_response.error_message}")
            elif bot_launched:
                # If only browser bot is used and it launched successfully
                meeting.status = MeetingStatus.ACTIVE
                meeting.joined_at = datetime.utcnow()
                self.active_meetings[meeting.id] = meeting
                logger.info(f"Successfully joined meeting with browser bot: {meeting.title}")
            else:
                meeting.status = MeetingStatus.FAILED
                meeting.error_message = "No client available and browser bot failed to launch"
                logger.error(f"No client available for platform: {meeting.platform}")
            
        except Exception as e:
            logger.error(f"Error joining meeting: {e}")
            meeting.status = MeetingStatus.FAILED
            meeting.error_message = str(e)
    
    async def _monitor_active_meetings(self) -> None:
        """Monitor active meetings and handle completion."""
        try:
            for meeting_id, meeting in list(self.active_meetings.items()):
                # Check if meeting should end
                if datetime.utcnow() >= meeting.end_time:
                    await self._leave_meeting(meeting)
                    del self.active_meetings[meeting_id]
                
        except Exception as e:
            logger.error(f"Error monitoring active meetings: {e}")
    
    async def _leave_meeting(self, meeting: Meeting) -> None:
        """Leave a meeting."""
        try:
            logger.info(f"Leaving meeting: {meeting.title}")
            
            # Stop browser bot if active
            if str(meeting.id) in self.active_bot_containers:
                try:
                    await self.stop_browser_bot(str(meeting.id))
                    logger.info(f"Stopped browser bot for meeting: {meeting.title}")
                except Exception as e:
                    logger.error(f"Error stopping browser bot: {e}")
            
            # Get appropriate client
            client = self.meeting_clients.get(meeting.platform)
            if client:
                await client.leave_meeting()
            
            # Update meeting status
            meeting.status = MeetingStatus.ENDED
            meeting.ended_at = datetime.utcnow()
            
            logger.info(f"Successfully left meeting: {meeting.title}")
            
        except Exception as e:
            logger.error(f"Error leaving meeting: {e}")
    
    async def get_active_meetings(self) -> List[Meeting]:
        """Get list of active meetings."""
        return list(self.active_meetings.values())
    
    async def get_meeting_status(self, meeting_id: str) -> Optional[MeetingStatus]:
        """Get status of a specific meeting."""
        meeting = self.active_meetings.get(meeting_id)
        return meeting.status if meeting else None
    
    async def force_join_meeting(self, meeting: Meeting) -> bool:
        """Force join a meeting (for manual triggers)."""
        try:
            await self._join_meeting(meeting)
            return meeting.status == MeetingStatus.ACTIVE
        except Exception as e:
            logger.error(f"Error force joining meeting: {e}")
            return False
    
    async def _initialize_browser_bot_orchestration(self) -> None:
        """Initialize browser bot orchestration clients."""
        try:
            # Initialize Docker client for local development
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Docker client: {e}")
            
            # Initialize AWS clients for production
            try:
                self.ecs_client = boto3.client('ecs')
                self.sqs_client = boto3.client('sqs')
                logger.info("AWS clients initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AWS clients: {e}")
            
            logger.info("Browser bot orchestration initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize browser bot orchestration: {e}")
            raise
    
    async def launch_browser_bot(self, meeting_id: str, platform: str, meeting_url: str, bot_name: str = None) -> bool:
        """Launch a browser bot container for a meeting."""
        try:
            logger.info(f"Launching browser bot for meeting: {meeting_id}")
            
            if not self.browser_bot_enabled:
                logger.warning("Browser bot is disabled")
                return False
            
            # Check if we're at capacity
            if len(self.active_bot_containers) >= self.max_concurrent_bots:
                logger.warning(f"Maximum concurrent bots ({self.max_concurrent_bots}) reached")
                return False
            
            # Generate unique session ID
            session_id = str(uuid4())
            
            # Prepare environment variables
            env_vars = {
                'MEETING_URL': meeting_url,
                'BOT_NAME': bot_name or 'Clerk AI Bot',
                'PLATFORM': platform,
                'MEETING_ID': meeting_id,
                'SESSION_ID': session_id,
                'RT_GATEWAY_URL': getattr(settings, 'rt_gateway_url', 'ws://localhost:8001'),
                'API_BASE_URL': getattr(settings, 'api_base_url', 'http://localhost:8000'),
                'JOIN_TIMEOUT_SEC': str(self.bot_join_timeout_sec),
                'AUDIO_SAMPLE_RATE': '16000',
                'AUDIO_CHANNELS': '1',
                'LOG_LEVEL': 'info'
            }
            
            # Try Docker first (local development)
            if self.docker_client:
                success = await self._launch_docker_bot(meeting_id, session_id, env_vars)
                if success:
                    return True
            
            # Fall back to AWS ECS (production)
            if self.ecs_client:
                success = await self._launch_ecs_bot(meeting_id, session_id, env_vars)
                if success:
                    return True
            
            logger.error("Failed to launch browser bot with any method")
            return False
            
        except Exception as e:
            logger.error(f"Error launching browser bot: {e}")
            return False
    
    async def _launch_docker_bot(self, meeting_id: str, session_id: str, env_vars: Dict[str, str]) -> bool:
        """Launch browser bot using Docker."""
        try:
            logger.info(f"Launching Docker bot for meeting: {meeting_id}")
            
            # Create container name
            container_name = f"clerk-bot-{meeting_id}-{session_id[:8]}"
            
            # Run container
            container = self.docker_client.containers.run(
                image=self.bot_image,
                name=container_name,
                environment=env_vars,
                detach=True,
                remove=True,
                network_mode='host',
                shm_size='2g',
                mem_limit=f'{self.bot_container_memory}m',
                cpu_quota=int(self.bot_container_cpu),
                security_opt=['seccomp:unconfined'],
                cap_add=['SYS_ADMIN'],
                devices=['/dev/snd:/dev/snd'],
                volumes={
                    '/dev/shm': {'bind': '/dev/shm', 'mode': 'rw'},
                    '/tmp/.X11-unix': {'bind': '/tmp/.X11-unix', 'mode': 'rw'}
                }
            )
            
            # Store container info
            self.active_bot_containers[meeting_id] = {
                'container_id': container.id,
                'container_name': container_name,
                'session_id': session_id,
                'started_at': datetime.utcnow(),
                'platform': env_vars['PLATFORM']
            }
            
            logger.info(f"Docker bot launched successfully: {container_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error launching Docker bot: {e}")
            return False
    
    async def _launch_ecs_bot(self, meeting_id: str, session_id: str, env_vars: Dict[str, str]) -> bool:
        """Launch browser bot using AWS ECS."""
        try:
            logger.info(f"Launching ECS bot for meeting: {meeting_id}")
            
            # Prepare task definition overrides
            overrides = {
                'containerOverrides': [
                    {
                        'name': 'browser-bot',
                        'environment': [
                            {'name': k, 'value': v} for k, v in env_vars.items()
                        ]
                    }
                ]
            }
            
            # Run ECS task
            response = self.ecs_client.run_task(
                cluster=getattr(settings, 'ecs_cluster_name', 'clerk-cluster'),
                taskDefinition=getattr(settings, 'ecs_task_definition', 'clerk-browser-bot'),
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': getattr(settings, 'ecs_subnet_ids', []),
                        'securityGroups': getattr(settings, 'ecs_security_group_ids', []),
                        'assignPublicIp': 'DISABLED'
                    }
                },
                overrides=overrides,
                tags=[
                    {'key': 'MeetingId', 'value': meeting_id},
                    {'key': 'SessionId', 'value': session_id},
                    {'key': 'Platform', 'value': env_vars['PLATFORM']}
                ]
            )
            
            if response['tasks']:
                task_arn = response['tasks'][0]['taskArn']
                
                # Store task info
                self.active_bot_containers[meeting_id] = {
                    'task_arn': task_arn,
                    'session_id': session_id,
                    'started_at': datetime.utcnow(),
                    'platform': env_vars['PLATFORM']
                }
                
                logger.info(f"ECS bot launched successfully: {task_arn}")
                return True
            else:
                logger.error("No tasks created in ECS response")
                return False
                
        except Exception as e:
            logger.error(f"Error launching ECS bot: {e}")
            return False
    
    async def stop_browser_bot(self, meeting_id: str) -> bool:
        """Stop a browser bot container."""
        try:
            logger.info(f"Stopping browser bot for meeting: {meeting_id}")
            
            if meeting_id not in self.active_bot_containers:
                logger.warning(f"No active bot found for meeting: {meeting_id}")
                return False
            
            bot_info = self.active_bot_containers[meeting_id]
            success = False
            
            # Try Docker first
            if 'container_id' in bot_info and self.docker_client:
                try:
                    container = self.docker_client.containers.get(bot_info['container_id'])
                    container.stop(timeout=10)
                    success = True
                    logger.info(f"Docker bot stopped: {bot_info['container_name']}")
                except Exception as e:
                    logger.error(f"Error stopping Docker bot: {e}")
            
            # Try ECS
            if 'task_arn' in bot_info and self.ecs_client:
                try:
                    self.ecs_client.stop_task(
                        cluster=getattr(settings, 'ecs_cluster_name', 'clerk-cluster'),
                        task=bot_info['task_arn'],
                        reason='Meeting ended'
                    )
                    success = True
                    logger.info(f"ECS bot stopped: {bot_info['task_arn']}")
                except Exception as e:
                    logger.error(f"Error stopping ECS bot: {e}")
            
            # Remove from active containers
            del self.active_bot_containers[meeting_id]
            
            return success
            
        except Exception as e:
            logger.error(f"Error stopping browser bot: {e}")
            return False
    
    async def get_active_bot_containers(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active bot containers."""
        return self.active_bot_containers.copy()
    
    async def cleanup_bot_containers(self) -> None:
        """Cleanup all active bot containers."""
        logger.info("Cleaning up bot containers...")
        
        try:
            for meeting_id in list(self.active_bot_containers.keys()):
                await self.stop_browser_bot(meeting_id)
            
            logger.info("Bot containers cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during bot containers cleanup: {e}")

    async def cleanup(self) -> None:
        """Cleanup scheduler resources."""
        logger.info("Cleaning up meeting scheduler...")
        
        try:
            await self.stop()
            
            # Cleanup bot containers
            if self.browser_bot_enabled:
                await self.cleanup_bot_containers()
            
            logger.info("Meeting scheduler cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Factory function for creating meeting scheduler
def create_meeting_scheduler() -> MeetingScheduler:
    """Create a new meeting scheduler instance."""
    return MeetingScheduler()
