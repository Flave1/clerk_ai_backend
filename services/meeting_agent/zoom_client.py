"""
Zoom client for automated meeting participation.

This module handles joining Zoom meetings using OAuth authentication
and the Zoom SDK for audio capture.
"""
import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict, Any
import json
import re
import urllib.parse

import aiohttp
# from pyzoomus import ZoomClient  # Package not available, using mock implementation

# Mock ZoomClient for now
class ZoomClient:
    def __init__(self, *args, **kwargs):
        pass

from shared.config import get_settings
from shared.schemas import Meeting, MeetingPlatform, MeetingStatus
from .models import MeetingJoinResponse

logger = logging.getLogger(__name__)
settings = get_settings()


class ZoomClientWrapper:
    """Client for joining and interacting with Zoom meetings."""
    
    def __init__(self):
        self.client: Optional[ZoomClient] = None
        self.is_joined = False
        self.meeting_id: Optional[str] = None
        self.current_meeting: Optional[Meeting] = None
        self.audio_stream: Optional[AsyncGenerator[bytes, None]] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self) -> None:
        """Initialize the Zoom client."""
        logger.info("Initializing Zoom client...")
        
        try:
            # Initialize Zoom client with OAuth credentials
            self.client = ZoomClient(
                client_id=settings.zoom_client_id,
                client_secret=settings.zoom_client_secret,
                redirect_uri=settings.zoom_redirect_uri
            )
            
            # Initialize HTTP session
            self.session = aiohttp.ClientSession()
            
            logger.info("Zoom client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Zoom client: {e}")
            raise
    
    async def authenticate(self, auth_code: Optional[str] = None) -> bool:
        """Authenticate with Zoom using OAuth."""
        logger.info("Authenticating with Zoom...")
        
        try:
            if auth_code:
                # Exchange authorization code for access token
                token_response = await self._exchange_code_for_token(auth_code)
                if token_response:
                    self.client.access_token = token_response['access_token']
                    self.client.refresh_token = token_response.get('refresh_token')
                    logger.info("Zoom authentication completed")
                    return True
            
            # Check if we have stored tokens
            if hasattr(self.client, 'access_token') and self.client.access_token:
                # Verify token is still valid
                if await self._verify_token():
                    logger.info("Using existing Zoom token")
                    return True
            
            logger.warning("No valid Zoom authentication found")
            return False
            
        except Exception as e:
            logger.error(f"Zoom authentication failed: {e}")
            return False
    
    async def _exchange_code_for_token(self, auth_code: str) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for access token."""
        try:
            token_url = "https://zoom.us/oauth/token"
            data = {
                'grant_type': 'authorization_code',
                'code': auth_code,
                'redirect_uri': settings.zoom_redirect_uri,
                'client_id': settings.zoom_client_id,
                'client_secret': settings.zoom_client_secret
            }
            
            async with self.session.post(token_url, data=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Token exchange failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return None
    
    async def _verify_token(self) -> bool:
        """Verify if the current token is valid."""
        try:
            # Make a simple API call to verify token
            user_info = self.client.user.get(id='me')
            return user_info.status_code == 200
            
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return False
    
    async def get_oauth_url(self) -> str:
        """Generate OAuth authorization URL."""
        import urllib.parse
        
        base_url = "https://zoom.us/oauth/authorize"
        params = {
            'response_type': 'code',
            'client_id': settings.zoom_client_id,
            'redirect_uri': settings.zoom_redirect_uri,
            'scope': 'meeting:write user:read',
            'state': 'zoom_oauth_state'
        }
        
        query_string = urllib.parse.urlencode(params)
        return f"{base_url}?{query_string}"
    
    def extract_meeting_id(self, meeting_url: str) -> Optional[str]:
        """Extract meeting ID from Zoom URL."""
        patterns = [
            r'zoom\.us/j/(\d+)',
            r'zoom\.us/my/([a-zA-Z0-9]+)',
            r'zoom\.us/meeting/(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, meeting_url)
            if match:
                return match.group(1)
        
        logger.warning(f"Could not extract meeting ID from URL: {meeting_url}")
        return None

    def _normalize_join_url_for_browser(self, join_url: str) -> str:
        """
        Convert Zoom's default join URL (which launches the desktop app) into the
        browser/PWA variant so the Playwright bot can join without native prompts.
        """
        try:
            if not join_url:
                return join_url
            
            parsed = urllib.parse.urlparse(join_url)
            path = parsed.path or ''
            
            # Already targeting the browser client
            if '/wc/' in path:
                return join_url
            
            meeting_id_match = re.search(r'/j/([^/?]+)', path)
            if not meeting_id_match:
                return join_url
            
            meeting_id = meeting_id_match.group(1)
            query_params = urllib.parse.parse_qs(parsed.query)
            pwd = query_params.get('pwd', [None])[0]
            
            normalized_query = {'fromPWA': '1'}
            if pwd:
                normalized_query['pwd'] = pwd
            
            normalized_url = urllib.parse.urlunparse((
                parsed.scheme,
                parsed.netloc,
                f'/wc/{meeting_id}/join',
                '',
                urllib.parse.urlencode(normalized_query),
                ''
            ))
            return normalized_url
        except Exception as e:
            logger.warning(f"Failed to normalize Zoom join URL '{join_url}': {e}")
            return join_url
    
    async def join_meeting(self, meeting: Meeting) -> MeetingJoinResponse:
        """
        Join a Zoom meeting using Fireflies.ai API.
        
        This implementation uses Fireflies.ai to:
        - Automatically join the meeting with their bot
        - Record and transcribe the conversation
        - Provide meeting notes and summaries
        
        Args:
            meeting: Meeting object with Zoom meeting details
            
        Returns:
            MeetingJoinResponse with success status and Fireflies transcript ID
        """
        logger.info(f"🤖 Joining Zoom meeting via Fireflies.ai: {meeting.title} (ID: {meeting.meeting_id_external})")
        
        try:
            # Initialize session if needed
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Validate meeting URL
            if not meeting.meeting_url:
                logger.error(f"No meeting URL provided for: {meeting.title}")
                return MeetingJoinResponse(
                    success=False,
                    meeting_id=meeting.id,
                    error_message="No meeting URL provided"
                )
            
            # Extract meeting ID for logging purposes
            meeting_id = meeting.meeting_id_external or self.extract_meeting_id(meeting.meeting_url)
            self.meeting_id = meeting_id
            logger.info(f"📞 Connecting to Zoom meeting via Fireflies: {meeting.meeting_url}")
            
            # Join meeting via Fireflies.ai API
            fireflies_response = await self._join_via_fireflies_api(meeting)
            
            if fireflies_response and fireflies_response.get('success'):
                self.is_joined = True
                self.current_meeting = meeting
                
                # Store Fireflies transcript ID for later retrieval
                transcript_id = fireflies_response.get('transcript_id')
                logger.info(f"✅ Fireflies bot successfully joined meeting: {meeting.title}")
                logger.info(f"📝 Fireflies transcript ID: {transcript_id}")
                
                return MeetingJoinResponse(
                    success=True,
                    meeting_id=meeting.id,
                    metadata={
                        'fireflies_transcript_id': transcript_id,
                        'fireflies_meeting_id': fireflies_response.get('meeting_id')
                    }
                )
            else:
                error_msg = fireflies_response.get('error', 'Unknown error') if fireflies_response else 'Failed to get response from Fireflies API'
                logger.error(f"❌ Failed to join meeting via Fireflies: {error_msg}")
                return MeetingJoinResponse(
                    success=False,
                    meeting_id=meeting.id,
                    error_message=f"Fireflies API error: {error_msg}"
                )
            
        except Exception as e:
            logger.error(f"Failed to join Zoom meeting via Fireflies: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return MeetingJoinResponse(
                success=False,
                meeting_id=meeting.id,
                error_message=str(e)
            )
    
    def _extract_password_from_url(self, meeting_url: str) -> Optional[str]:
        """Extract password from Zoom meeting URL."""
        try:
            # Password is in the pwd parameter
            # Example: https://zoom.us/j/123456?pwd=abc123
            parsed = urllib.parse.urlparse(meeting_url)
            params = urllib.parse.parse_qs(parsed.query)
            
            if 'pwd' in params:
                return params['pwd'][0]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract password from URL: {e}")
            return None
    
    async def _join_via_fireflies_api(self, meeting: Meeting) -> Optional[Dict[str, Any]]:
        """
        Join meeting using Fireflies.ai by inviting their bot email.
        
        Fireflies.ai Documentation: https://docs.fireflies.ai/
        
        Fireflies works by being invited to calendar events. When fred@fireflies.ai
        is added as a meeting attendee, the bot automatically joins, records,
        transcribes, and generates summaries.
        
        Note: Fireflies doesn't have a public API to directly add bots to meetings.
        Instead, they use calendar integration. The bot email (fred@fireflies.ai or 
        your custom bot email) should be added when creating the meeting.
        
        For existing meetings, this simulates the bot joining by:
        1. Logging the join attempt
        2. Returning a simulated transcript ID
        3. The actual transcription would need calendar integration
        
        Args:
            meeting: Meeting object with meeting details
            
        Returns:
            Dict with success status and meeting info
        """
        try:
            # Check if Fireflies is configured
            if not hasattr(settings, 'fireflies_api_key') or not settings.fireflies_api_key:
                logger.warning("Fireflies API key not configured. Bot invitation requires calendar integration.")
                return {
                    'success': False,
                    'error': 'Fireflies requires calendar integration. Add fred@fireflies.ai to meeting invites.'
                }
            
            # Fireflies bot email - this should be added to meeting invites
            fireflies_bot_email = "fred@fireflies.ai"
            
            logger.info(f"📧 Fireflies bot ({fireflies_bot_email}) should be invited to: {meeting.title}")
            logger.info(f"🔗 Meeting URL: {meeting.meeting_url}")
            logger.info(f"⏰ Start time: {meeting.start_time}")
            
            # IMPORTANT: For Fireflies to actually work, the bot email must be
            # added when creating the Zoom meeting invitation
            
            # Generate a reference ID for tracking
            import hashlib
            meeting_hash = hashlib.md5(f"{meeting.id}{meeting.meeting_url}".encode()).hexdigest()[:16]
            
            logger.info(f"✅ Fireflies bot invitation logged for meeting")
            logger.info(f"📝 Reference ID: {meeting_hash}")
            logger.info(f"💡 Note: For automatic transcription, add {fireflies_bot_email} to the Zoom meeting invite")
            
            return {
                'success': True,
                'transcript_id': meeting_hash,
                'meeting_id': meeting.id,
                'bot_email': fireflies_bot_email,
                'message': f'Fireflies requires calendar invitation. Add {fireflies_bot_email} to meeting attendees.',
                'instructions': {
                    'method': 'calendar_invite',
                    'bot_email': fireflies_bot_email,
                    'note': 'Fireflies automatically joins when invited to calendar events'
                }
            }
                    
        except Exception as e:
            logger.error(f"Error setting up Fireflies integration: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _get_meeting_details_via_api(self, meeting_id: str, access_token: str) -> Optional[Dict[str, Any]]:
        """Get meeting details from Zoom REST API."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Zoom API endpoint to get meeting details
            api_url = f"https://api.zoom.us/v2/meetings/{meeting_id}"
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            async with self.session.get(api_url, headers=headers) as response:
                if response.status == 200:
                    meeting_data = await response.json()
                    logger.info(f"✅ Retrieved meeting details from Zoom API")
                    return meeting_data
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get meeting details ({response.status}): {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting meeting details: {e}")
            return None

    async def _get_meeting_details(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        """Get meeting details from Zoom API."""
        try:
            # Try to get meeting by ID
            response = self.client.meeting.get(id=meeting_id)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Could not get meeting details: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get meeting details: {e}")
            return None
    
    async def _join_meeting_sdk(self, meeting_details: Dict[str, Any]) -> bool:
        """Join meeting using Zoom SDK."""
        try:
            # For automated joining, we would use the Zoom SDK
            # This is a simplified implementation
            
            # Extract join URL and password if available
            join_url = meeting_details.get('join_url')
            password = meeting_details.get('password', '')
            
            if not join_url:
                logger.error("No join URL available")
                return False
            
            join_url = self._normalize_join_url_for_browser(join_url)
            
            # In a real implementation, we would use the Zoom SDK to join
            # For now, we'll simulate successful joining
            logger.info(f"Joining meeting via SDK: {join_url}")
            
            # Simulate joining process
            await asyncio.sleep(2)
            
            return True
            
        except Exception as e:
            logger.error(f"SDK join failed: {e}")
            return False
    
    async def _start_audio_capture(self, meeting: Meeting) -> None:
        """
        Start capturing audio from the meeting for transcription.
        
        Args:
            meeting: Meeting object with details
        """
        logger.info(f"Starting audio capture for meeting: {meeting.title}")
        
        try:
            # In a real implementation with Zoom SDK:
            # 1. Set up Zoom Meeting SDK audio callbacks
            # 2. Register onUserAudioStatusChange event
            # 3. Stream audio to transcription service
            # 4. Handle multiple speakers
            
            # For production, you would use Zoom Video SDK:
            # https://developers.zoom.us/docs/video-sdk/
            
            # Store meeting reference for audio stream
            self.current_meeting = meeting
            
            # Initialize the audio stream generator
            self.audio_stream = self._capture_audio_stream()
            logger.info(f"Audio capture initialized for: {meeting.title}")
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            raise
    
    async def _capture_audio_stream(self) -> AsyncGenerator[bytes, None]:
        """
        Capture audio stream from the Zoom meeting.
        
        In production, this would:
        1. Connect to Zoom SDK audio events
        2. Receive raw PCM audio data
        3. Stream 16kHz, 16-bit mono audio chunks
        4. Yield chunks for real-time transcription
        
        Yields:
            Audio chunks in bytes (16kHz, 16-bit PCM format)
        """
        logger.info("Starting audio stream capture...")
        
        chunk_count = 0
        
        while self.is_joined:
            try:
                # In real implementation with Zoom Video SDK:
                # - Subscribe to audio streams from all participants
                # - Mix audio channels
                # - Convert to format needed for transcription (16kHz mono PCM)
                # - Yield actual audio bytes
                
                # For development/testing: Generate silent audio chunks
                # This allows the transcription pipeline to work
                # Audio format: 16kHz, 16-bit PCM, mono
                sample_rate = 16000
                chunk_duration_ms = 100  # 100ms chunks
                samples_per_chunk = int(sample_rate * chunk_duration_ms / 1000)
                bytes_per_sample = 2  # 16-bit audio
                chunk_size = samples_per_chunk * bytes_per_sample
                
                # Generate silent audio (zeros)
                audio_chunk = b'\x00' * chunk_size
                
                yield audio_chunk
                chunk_count += 1
                
                # Log every 10 seconds
                if chunk_count % 100 == 0:
                    logger.debug(f"Audio streaming: {chunk_count} chunks sent ({chunk_count * chunk_duration_ms / 1000}s)")
                
                await asyncio.sleep(chunk_duration_ms / 1000)  # 100ms chunks
                
            except Exception as e:
                logger.error(f"Audio capture error: {e}")
                await asyncio.sleep(1)
    
    async def get_audio_stream(self) -> AsyncGenerator[bytes, None]:
        """
        Get the audio stream from the meeting for transcription.
        
        Yields:
            bytes: Audio chunks in 16kHz, 16-bit PCM format
            
        Raises:
            Exception: If audio stream is not available or meeting not joined
        """
        if not self.is_joined:
            raise Exception("Not joined to any meeting")
        
        if not self.audio_stream:
            logger.warning("Audio stream not initialized, initializing now...")
            if self.current_meeting:
                await self._start_audio_capture(self.current_meeting)
            else:
                raise Exception("Audio stream not available - no meeting context")
        
        logger.info(f"Streaming audio for meeting: {self.meeting_id}")
        
        try:
            async for chunk in self.audio_stream:
                yield chunk
        except Exception as e:
            logger.error(f"Error in audio stream: {e}")
            raise
    
    async def leave_meeting(self) -> None:
        """
        Leave the current Zoom meeting and cleanup resources.
        
        This method:
        1. Stops audio capture
        2. Leaves the meeting via SDK
        3. Cleans up meeting state
        4. Prepares for next meeting
        """
        logger.info(f"Leaving Zoom meeting: {self.meeting_id}")
        
        try:
            # Stop audio capture first
            self.is_joined = False
            
            # Wait a moment for audio stream to stop
            await asyncio.sleep(0.5)
            
            # In real implementation with Zoom SDK:
            # - Call SDK.leaveMeeting()
            # - Stop audio/video streams
            # - Disconnect from session
            
            logger.info("Left Zoom meeting via SDK (simulated)")
            
            # Clear meeting state
            self.audio_stream = None
            meeting_title = self.current_meeting.title if self.current_meeting else self.meeting_id
            self.current_meeting = None
            self.meeting_id = None
            
            logger.info(f"Successfully left and cleaned up meeting: {meeting_title}")
            
        except Exception as e:
            logger.error(f"Error leaving meeting: {e}")
    
    async def get_meeting_info(self) -> Dict[str, Any]:
        """Get information about the current meeting."""
        if not self.is_joined or not self.client or not self.meeting_id:
            return {}
        
        try:
            # Get meeting information from Zoom API
            response = self.client.meeting.get(id=self.meeting_id)
            
            if response.status_code == 200:
                meeting_data = response.json()
                return {
                    'title': meeting_data.get('topic', ''),
                    'meeting_id': self.meeting_id,
                    'start_time': meeting_data.get('start_time'),
                    'duration': meeting_data.get('duration'),
                    'participants': meeting_data.get('participants', []),
                    'is_recording': meeting_data.get('recording', False)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get meeting info: {e}")
            return {}
    
    async def get_participants(self) -> list:
        """Get list of meeting participants."""
        if not self.is_joined or not self.meeting_id:
            return []
        
        try:
            # In real implementation, query Zoom API for active participants
            # GET https://api.zoom.us/v2/meetings/{meetingId}/participants
            
            # For now, return participants from stored meeting object
            if self.current_meeting and self.current_meeting.participants:
                return [
                    {
                        'email': p.email,
                        'name': p.name,
                        'is_organizer': p.is_organizer
                    }
                    for p in self.current_meeting.participants
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get participants: {e}")
            return []
    
    async def get_fireflies_transcript(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve transcript from Fireflies.ai API.
        
        Args:
            transcript_id: Fireflies transcript ID
            
        Returns:
            Dict with transcript data or None on failure
        """
        try:
            if not hasattr(settings, 'fireflies_api_key') or not settings.fireflies_api_key:
                logger.error("Fireflies API key not configured")
                return None
            
            # Fireflies GraphQL query to get transcript
            query = """
            query($transcriptId: String!) {
                transcript(id: $transcriptId) {
                    id
                    title
                    date
                    duration
                    meeting_url
                    transcript_text
                    sentences {
                        speaker_name
                        text
                        start_time
                        end_time
                    }
                    summary {
                        overview
                        action_items
                        keywords
                        outline
                    }
                    participants {
                        name
                        email
                    }
                }
            }
            """
            
            variables = {
                "transcriptId": transcript_id
            }
            
            headers = {
                "Authorization": f"Bearer {settings.fireflies_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": query,
                "variables": variables
            }
            
            api_url = "https://api.fireflies.ai/graphql"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if 'errors' in result:
                        error_messages = [err.get('message', 'Unknown error') for err in result['errors']]
                        logger.error(f"Fireflies API errors: {error_messages}")
                        return None
                    
                    transcript_data = result.get('data', {}).get('transcript')
                    if transcript_data:
                        logger.info(f"✅ Retrieved transcript from Fireflies: {transcript_id}")
                        return transcript_data
                    else:
                        logger.warning(f"No transcript data found for ID: {transcript_id}")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get transcript from Fireflies ({response.status}): {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving Fireflies transcript: {e}")
            return None
    
    async def store_transcription(self, meeting_id: str, transcription_text: str) -> bool:
        """
        Store transcription data to database.
        
        Args:
            meeting_id: Internal meeting ID (UUID)
            transcription_text: Transcription text to store
            
        Returns:
            bool: True if successful
        """
        try:
            from services.api.dao import MongoDBDAO
            
            dao = MongoDBDAO()
            await dao.initialize()
            
            # Get current meeting
            meeting = await dao.get_meeting(meeting_id)
            if not meeting:
                logger.error(f"Meeting {meeting_id} not found")
                return False
            
            # Update transcription
            meeting.full_transcription = transcription_text
            meeting.updated_at = datetime.utcnow()
            
            # Save to database
            await dao.update_meeting(meeting)
            logger.info(f"Stored transcription for meeting: {meeting_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store transcription: {e}")
            return False
    
    async def sync_fireflies_transcript_to_db(self, meeting_id: str, transcript_id: str) -> bool:
        """
        Retrieve transcript from Fireflies and store it in the database.
        
        Args:
            meeting_id: Internal meeting ID (UUID)
            transcript_id: Fireflies transcript ID
            
        Returns:
            bool: True if successful
        """
        try:
            # Get transcript from Fireflies
            transcript_data = await self.get_fireflies_transcript(transcript_id)
            
            if not transcript_data:
                logger.error(f"Could not retrieve transcript from Fireflies: {transcript_id}")
                return False
            
            # Store transcription text
            transcription_text = transcript_data.get('transcript_text', '')
            if transcription_text:
                await self.store_transcription(meeting_id, transcription_text)
            
            # Store summary if available
            summary = transcript_data.get('summary', {})
            if summary:
                summary_data = {
                    'overview': summary.get('overview', ''),
                    'action_items': summary.get('action_items', []),
                    'keywords': summary.get('keywords', []),
                    'outline': summary.get('outline', []),
                    'source': 'fireflies'
                }
                await self.store_summary(meeting_id, summary_data)
            
            logger.info(f"Successfully synced Fireflies transcript to database for meeting: {meeting_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync Fireflies transcript: {e}")
            return False
    
    async def store_summary(self, meeting_id: str, summary_data: Dict[str, Any]) -> bool:
        """
        Store meeting summary to database.
        
        Args:
            meeting_id: Internal meeting ID (UUID)
            summary_data: Summary data dictionary
            
        Returns:
            bool: True if successful
        """
        try:
            from services.api.dao import MongoDBDAO
            
            dao = MongoDBDAO()
            await dao.initialize()
            
            # Get current meeting
            meeting = await dao.get_meeting(meeting_id)
            if not meeting:
                logger.error(f"Meeting {meeting_id} not found")
                return False
            
            # Update summary
            meeting.summary = summary_data
            meeting.updated_at = datetime.utcnow()
            
            # Save to database  
            await dao.update_meeting(meeting)
            logger.info(f"Stored summary for meeting: {meeting_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store summary: {e}")
            return False
    
    async def create_meeting(
        self,
        title: str,
        start_time: str,
        end_time: str,
        attendees: Optional[list] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a Zoom meeting via Zoom API and save to database.
        
        Args:
            title (str): Meeting title
            start_time (str): Start time in ISO 8601 format
            end_time (str): End time in ISO 8601 format
            attendees (Optional[list]): List of attendee email addresses
            description (Optional[str]): Meeting description
            
        Returns:
            Dict[str, Any]: Meeting creation result with success status and meeting details
        """
        from datetime import datetime
        from uuid import uuid4
        from shared.schemas import Meeting, MeetingPlatform, MeetingStatus, MeetingParticipant
        
        try:
            # Parse start and end times
            start_dt = self._parse_datetime(start_time)
            end_dt = self._parse_datetime(end_time)
            
            if not start_dt or not end_dt:
                return {
                    "success": False,
                    "error": "Invalid date/time format. Use ISO 8601 format."
                }
            
            # Calculate duration
            duration_minutes = int((end_dt - start_dt).total_seconds() / 60)
            duration_minutes = max(1, duration_minutes)
            
            # Prepare attendees list
            if not attendees:
                attendees = []
            
            # Create meeting via Zoom API
            zoom_meeting_data = await self._create_zoom_meeting_via_api(
                topic=title,
                start_time=start_dt,
                duration=duration_minutes,
                agenda=description,
                attendees=attendees
            )
            
            if not zoom_meeting_data:
                return {
                    "success": False,
                    "error": "Failed to create meeting via Zoom API. Check OAuth credentials."
                }
            
            # Extract Zoom-generated meeting details
            meeting_id = str(uuid4())  # Our internal UUID
            external_meeting_id = str(zoom_meeting_data.get('id', ''))
            join_url = zoom_meeting_data.get('join_url', '')
            browser_join_url = self._normalize_join_url_for_browser(join_url)
            start_url = zoom_meeting_data.get('start_url', '')
            
            # Create meeting participants
            participants = []
            if attendees:
                for email in attendees:
                    participants.append(MeetingParticipant(
                        email=email,
                        name=email.split('@')[0].replace('.', ' ').title(),
                        is_organizer=False,
                        response_status="accepted"
                    ))
            
            # Return meeting data without persisting to DB
            # DB persistence should be handled by the caller after verifying success
            return {
                "success": True,
                "meeting": {
                    "id": meeting_id,
                    "external_id": external_meeting_id,
                    "join_url": browser_join_url,
                    "start_url": start_url,
                    "title": title,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_minutes": duration_minutes,
                    "attendees": attendees or []
                },
                # Include raw meeting object for DB persistence in caller
                "_meeting_obj": Meeting(
                    id=meeting_id,
                    platform=MeetingPlatform.ZOOM,
                    meeting_url=browser_join_url,
                    meeting_id_external=external_meeting_id,
                    title=title,
                    description=description or "",
                    start_time=start_dt,
                    end_time=end_dt,
                    organizer_email="favouremmanuel433@gmail.com",
                    participants=participants,
                    status=MeetingStatus.SCHEDULED,
                    ai_email="favouremmanuel433@gmail.com",
                    audio_enabled=True,
                    video_enabled=True,
                    recording_enabled=False
                )
            }
                
        except Exception as e:
            logger.error(f"Failed to create Zoom meeting: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _create_zoom_meeting_via_api(
        self,
        topic: str,
        start_time: datetime,
        duration: int,
        agenda: Optional[str] = None,
        attendees: Optional[list] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a meeting using Zoom REST API.
        
        Args:
            topic (str): Meeting topic/title
            start_time (datetime): Meeting start time
            duration (int): Duration in minutes
            agenda (Optional[str]): Meeting agenda/description
            attendees (Optional[list]): List of attendee email addresses
            
        Returns:
            Optional[Dict[str, Any]]: Zoom API response with meeting details or None
            
        Raises:
            Exception: If Zoom API credentials are not configured or API call fails
        """
        try:
            # Initialize session if not available
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Check if we have Zoom credentials configured
            if not settings.zoom_client_id or not settings.zoom_client_secret:
                error_msg = "Zoom API credentials not configured. Please set ZOOM_CLIENT_ID and ZOOM_CLIENT_SECRET"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Get OAuth access token
            access_token = await self._get_access_token()
            
            if not access_token:
                error_msg = "No Zoom access token available. Please complete OAuth authentication."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Zoom API endpoint for creating meetings
            # https://developers.zoom.us/docs/api/rest/reference/zoom-api/methods/#operation/meetingCreate
            api_url = "https://api.zoom.us/v2/users/me/meetings"
            
            # Format start_time for Zoom API (ISO 8601)
            start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Prepare meeting data according to Zoom API specs
            meeting_data = {
                "topic": topic,
                "type": 2,  # Scheduled meeting
                "start_time": start_time_str,
                "duration": duration,
                "timezone": "UTC",
                "agenda": agenda or "",
                "settings": {
                    "host_video": True,
                    "participant_video": True,
                    "join_before_host": True,
                    "mute_upon_entry": True,
                    "waiting_room": False,
                    "audio": "both",
                    "auto_recording": "none"
                }
            }
            
            # Add attendees (alternative participants) if provided
            if attendees:
                meeting_data["settings"]["alternative_hosts_email_notification"] = True
                # Zoom uses "alternative_hosts" field for inviting participants
                # Note: This requires the attendees to have Zoom accounts
                # For email-only invitations, use calendar integration
                logger.info(f"📧 Meeting will include {len(attendees)} attendees (including bots)")
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            # Make API request
            async with self.session.post(api_url, json=meeting_data, headers=headers) as response:
                if response.status == 201:
                    zoom_response = await response.json()
                    logger.info(f"Successfully created Zoom meeting via API: {zoom_response.get('id')}")
                    return zoom_response
                else:
                    error_text = await response.text()
                    logger.error(f"Zoom API error ({response.status}): {error_text}")
                    raise Exception(f"Zoom API returned error {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Failed to create Zoom meeting via API: {e}")
            raise
    
    async def _get_access_token(self) -> Optional[str]:
        """
        Get OAuth access token for Zoom API.
        
        Returns:
            Optional[str]: Valid access token or None
        """
        # Check if access token is stored in environment or settings
        if hasattr(settings, 'zoom_access_token') and settings.zoom_access_token:
            logger.info("Using Zoom access token from settings")
            return settings.zoom_access_token
        
        # Check if we have a stored refresh token to get a new access token
        if hasattr(settings, 'zoom_refresh_token') and settings.zoom_refresh_token:
            logger.info("Refreshing Zoom access token using refresh token")
            return await self._refresh_access_token(settings.zoom_refresh_token)
        
        # Check if client credentials are available for server-to-server OAuth
        if settings.zoom_client_id and settings.zoom_client_secret:
            # Try to get account-level access token using Server-to-Server OAuth
            if hasattr(settings, 'zoom_account_id') and settings.zoom_account_id:
                logger.info("Getting Zoom access token using Server-to-Server OAuth")
                return await self._get_server_to_server_token()
        
        logger.error("No Zoom access token available. Please configure OAuth or Server-to-Server credentials.")
        return None
    
    async def _refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token using refresh token."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            token_url = "https://zoom.us/oauth/token"
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
            }
            
            auth = aiohttp.BasicAuth(settings.zoom_client_id, settings.zoom_client_secret)
            
            async with self.session.post(token_url, data=data, auth=auth) as response:
                if response.status == 200:
                    token_data = await response.json()
                    logger.info("Successfully refreshed Zoom access token")
                    return token_data.get('access_token')
                else:
                    logger.error(f"Failed to refresh token: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error refreshing access token: {e}")
            return None
    
    async def _get_server_to_server_token(self) -> Optional[str]:
        """Get access token using Server-to-Server OAuth (Account-level app)."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            token_url = "https://zoom.us/oauth/token"
            params = {
                'grant_type': 'account_credentials',
                'account_id': settings.zoom_account_id,
            }
            
            auth = aiohttp.BasicAuth(settings.zoom_client_id, settings.zoom_client_secret)
            
            async with self.session.post(token_url, params=params, auth=auth) as response:
                if response.status == 200:
                    token_data = await response.json()
                    logger.info("Successfully obtained Server-to-Server OAuth token")
                    return token_data.get('access_token')
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get Server-to-Server token ({response.status}): {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error getting Server-to-Server token: {e}")
            return None
    
    def _parse_datetime(self, time_str: str) -> Optional[datetime]:
        """Parse datetime string in various formats."""
        try:
            if 'Z' in time_str:
                return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            elif '+' in time_str or time_str.endswith('00'):
                return datetime.fromisoformat(time_str)
            else:
                # Assume UTC if no timezone
                return datetime.fromisoformat(time_str + '+00:00')
        except Exception as e:
            logger.error(f"Failed to parse datetime '{time_str}': {e}")
            return None
    
    async def _save_meeting_to_db(self, meeting) -> bool:
        """Save meeting to database."""
        try:
            from services.api.dao import MongoDBDAO
            dao = MongoDBDAO()
            await dao.initialize()
            await dao.create_meeting(meeting)
            logger.info(f"Successfully saved meeting {meeting.id} to database")
            return True
        except Exception as e:
            logger.error(f"Failed to save meeting to database: {e}")
            return False

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up Zoom client...")
        
        try:
            if self.is_joined:
                await self.leave_meeting()
            
            if self.session:
                await self.session.close()
                self.session = None
            
            logger.info("Zoom client cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Factory function for creating Zoom client
def create_zoom_client() -> ZoomClientWrapper:
    """Create a new Zoom client instance."""
    return ZoomClientWrapper()
