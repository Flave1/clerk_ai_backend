"""
Main Meeting Agent service.

This module provides the main service that orchestrates all meeting agent
functionality including scheduling, joining, transcription, and summarization.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID

from shared.config import get_settings
from shared.schemas import Meeting, MeetingSummary, TranscriptionChunk
from .scheduler import create_meeting_scheduler
from .transcription_service import create_transcription_service
from .summarization_service import create_summarization_service
from .notifier import create_notification_service

logger = logging.getLogger(__name__)
settings = get_settings()


class MeetingAgentService:
    """Main service for meeting agent functionality."""
    
    def __init__(self):
        self.scheduler = create_meeting_scheduler()
        self.transcription_service = create_transcription_service()
        self.summarization_service = create_summarization_service()
        self.notification_service = create_notification_service()
        self.is_running = False
        self.active_meetings: Dict[UUID, Dict[str, Any]] = {}
        
    async def initialize(self) -> None:
        """Initialize the meeting agent service."""
        logger.info("Initializing Meeting Agent Service...")
        
        try:
            # Initialize all services
            await self.scheduler.initialize()
            await self.transcription_service.initialize()
            await self.summarization_service.initialize()
            await self.notification_service.initialize()
            
            logger.info("Meeting Agent Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Meeting Agent Service: {e}")
            raise
    
    async def start(self) -> None:
        """Start the meeting agent service."""
        logger.info("Starting Meeting Agent Service...")
        
        try:
            if not self.is_running:
                # Start scheduler
                await self.scheduler.start()
                
                self.is_running = True
                logger.info("Meeting Agent Service started successfully")
            else:
                logger.info("Meeting Agent Service is already running")
                
        except Exception as e:
            logger.error(f"Failed to start Meeting Agent Service: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the meeting agent service."""
        logger.info("Stopping Meeting Agent Service...")
        
        try:
            if self.is_running:
                # Stop scheduler
                await self.scheduler.stop()
                
                # Stop all active meeting processes
                await self._stop_all_active_meetings()
                
                self.is_running = False
                logger.info("Meeting Agent Service stopped successfully")
            else:
                logger.info("Meeting Agent Service is not running")
                
        except Exception as e:
            logger.error(f"Error stopping Meeting Agent Service: {e}")
    
    async def _stop_all_active_meetings(self) -> None:
        """Stop all active meeting processes."""
        try:
            for meeting_id, meeting_data in self.active_meetings.items():
                await self._stop_meeting_process(meeting_id, meeting_data)
            
            self.active_meetings.clear()
            
        except Exception as e:
            logger.error(f"Error stopping active meetings: {e}")
    
    async def _stop_meeting_process(self, meeting_id: UUID, meeting_data: Dict[str, Any]) -> None:
        """Stop a specific meeting process."""
        try:
            # Stop transcription
            if 'transcription_task' in meeting_data:
                meeting_data['transcription_task'].cancel()
            
            # Stop summarization
            if 'summarization_task' in meeting_data:
                meeting_data['summarization_task'].cancel()
            
            logger.info(f"Stopped meeting process for: {meeting_id}")
            
        except Exception as e:
            logger.error(f"Error stopping meeting process {meeting_id}: {e}")
    
    async def process_meeting(self, meeting: Meeting) -> None:
        """Process a meeting (join, transcribe, summarize, notify)."""
        logger.info(f"Processing meeting: {meeting.title}")
        
        try:
            # Get appropriate client from scheduler
            client = self.scheduler.meeting_clients.get(meeting.platform)
            if not client:
                logger.error(f"No client available for platform: {meeting.platform}")
                return
            
            # Join meeting
            join_response = await client.join_meeting(meeting)
            if not join_response.success:
                logger.error(f"Failed to join meeting: {join_response.error_message}")
                return
            
            # Start meeting process
            await self._start_meeting_process(meeting, client)
            
        except Exception as e:
            logger.error(f"Error processing meeting {meeting.title}: {e}")
    
    async def _start_meeting_process(self, meeting: Meeting, client: Any) -> None:
        """Start the complete meeting process."""
        try:
            # Initialize meeting data
            meeting_data = {
                'meeting': meeting,
                'client': client,
                'transcription_chunks': [],
                'start_time': datetime.utcnow()
            }
            
            self.active_meetings[meeting.id] = meeting_data
            
            # Start transcription
            if settings.meeting_transcription_enabled:
                transcription_task = asyncio.create_task(
                    self._transcribe_meeting(meeting, client)
                )
                meeting_data['transcription_task'] = transcription_task
            
            # Start summarization (periodic)
            if settings.meeting_summarization_enabled:
                summarization_task = asyncio.create_task(
                    self._summarize_meeting_periodic(meeting)
                )
                meeting_data['summarization_task'] = summarization_task
            
            logger.info(f"Started meeting process for: {meeting.title}")
            
        except Exception as e:
            logger.error(f"Error starting meeting process: {e}")
    
    async def _transcribe_meeting(self, meeting: Meeting, client: Any) -> None:
        """Transcribe meeting audio."""
        try:
            logger.info(f"Starting transcription for: {meeting.title}")
            
            # Get audio stream from client
            audio_stream = await client.get_audio_stream()
            
            # Start transcription
            async for transcription_chunk in self.transcription_service.start_transcription(meeting, audio_stream):
                # Store transcription chunk
                if meeting.id in self.active_meetings:
                    self.active_meetings[meeting.id]['transcription_chunks'].append(transcription_chunk)
                    
                    # Update meeting with transcription
                    meeting.transcription_chunks.append(transcription_chunk.text)
                    
                    logger.debug(f"Transcribed chunk: {transcription_chunk.text[:50]}...")
            
            # Store full transcription
            if meeting.id in self.active_meetings:
                chunks = self.active_meetings[meeting.id]['transcription_chunks']
                meeting.full_transcription = " ".join(chunk.text for chunk in chunks)
            
            logger.info(f"Completed transcription for: {meeting.title}")
            
        except Exception as e:
            logger.error(f"Error transcribing meeting {meeting.title}: {e}")
    
    async def _summarize_meeting_periodic(self, meeting: Meeting) -> None:
        """Generate periodic summaries during the meeting."""
        try:
            logger.info(f"Starting periodic summarization for: {meeting.title}")
            
            while meeting.id in self.active_meetings:
                # Wait for summarization interval
                await asyncio.sleep(settings.meeting_summary_frequency_minutes * 60)
                
                # Generate summary if we have transcription chunks
                if meeting.id in self.active_meetings:
                    chunks = self.active_meetings[meeting.id]['transcription_chunks']
                    if chunks:
                        summary = await self.summarization_service.summarize_meeting(meeting, chunks)
                        
                        # Store summary
                        meeting.summary = summary
                        
                        # Send notification if enabled
                        if settings.meeting_email_notifications_enabled or settings.meeting_slack_notifications_enabled:
                            await self.notification_service.send_meeting_summary(meeting, summary)
                        
                        logger.info(f"Generated periodic summary for: {meeting.title}")
            
        except asyncio.CancelledError:
            logger.info(f"Periodic summarization cancelled for: {meeting.title}")
        except Exception as e:
            logger.error(f"Error in periodic summarization for {meeting.title}: {e}")
    
    async def finalize_meeting(self, meeting: Meeting) -> None:
        """Finalize meeting processing and send final summary."""
        try:
            logger.info(f"Finalizing meeting: {meeting.title}")
            
            # Stop meeting process
            if meeting.id in self.active_meetings:
                await self._stop_meeting_process(meeting.id, self.active_meetings[meeting.id])
                del self.active_meetings[meeting.id]
            
            # Generate final summary
            if meeting.id in self.active_meetings:
                chunks = self.active_meetings[meeting.id]['transcription_chunks']
                if chunks:
                    summary = await self.summarization_service.summarize_meeting(meeting, chunks)
                    meeting.summary = summary
                    
                    # Send final notification
                    if settings.meeting_email_notifications_enabled or settings.meeting_slack_notifications_enabled:
                        await self.notification_service.send_meeting_summary(meeting, summary)
                        
                        # Send action items separately
                        if summary.action_items:
                            await self.notification_service.send_action_items(meeting, summary.action_items)
                    
                    logger.info(f"Finalized meeting with summary: {meeting.title}")
            
        except Exception as e:
            logger.error(f"Error finalizing meeting {meeting.title}: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get meeting agent service status."""
        try:
            active_meetings = await self.scheduler.get_active_meetings()
            
            return {
                'service_status': 'running' if self.is_running else 'stopped',
                'active_meetings_count': len(active_meetings),
                'processing_meetings_count': len(self.active_meetings),
                'active_meetings': [
                    {
                        'id': str(m.id),
                        'title': m.title,
                        'platform': m.platform.value,
                        'status': m.status.value
                    }
                    for m in active_meetings
                ],
                'services': {
                    'scheduler': 'running' if self.scheduler.is_running else 'stopped',
                    'transcription': 'initialized' if self.transcription_service.is_processing else 'idle',
                    'summarization': 'initialized' if self.summarization_service.is_initialized else 'not_initialized',
                    'notification': 'initialized' if self.notification_service.is_initialized else 'not_initialized'
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {'error': str(e)}
    
    async def cleanup(self) -> None:
        """Cleanup meeting agent service resources."""
        logger.info("Cleaning up Meeting Agent Service...")
        
        try:
            await self.stop()
            
            # Cleanup all services
            await self.scheduler.cleanup()
            await self.transcription_service.cleanup()
            await self.summarization_service.cleanup()
            await self.notification_service.cleanup()
            
            logger.info("Meeting Agent Service cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Global service instance
meeting_agent_service = MeetingAgentService()


# Factory function for creating meeting agent service
def create_meeting_agent_service() -> MeetingAgentService:
    """Create a new meeting agent service instance."""
    return MeetingAgentService()
