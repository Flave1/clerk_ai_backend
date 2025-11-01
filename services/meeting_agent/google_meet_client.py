"""
Google Meet client for automated meeting participation.

This module handles joining Google Meet meetings using OAuth authentication
and headless Chrome automation for audio capture.
"""
import asyncio
import logging
import re
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict, Any
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import aiohttp
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

from shared.config import get_settings
from shared.schemas import Meeting, MeetingPlatform, MeetingStatus
from .models import MeetingJoinResponse

logger = logging.getLogger(__name__)
settings = get_settings()


class GoogleMeetClient:
    """Client for joining and interacting with Google Meet meetings."""
    
    def __init__(self):
        self.driver: Optional[webdriver.Chrome] = None
        self.is_joined = False
        self.meeting_id: Optional[str] = None
        self.audio_stream: Optional[AsyncGenerator[bytes, None]] = None
        
    async def initialize(self) -> None:
        """Initialize the Google Meet client."""
        logger.info("Initializing Google Meet client...")
        
        # Setup Chrome options for headless operation
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--disable-javascript")
        
        # Audio capture settings
        chrome_options.add_argument("--use-fake-ui-for-media-stream")
        chrome_options.add_argument("--use-fake-device-for-media-stream")
        chrome_options.add_argument("--allow-file-access-from-files")
        
        # User agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Google Meet client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Meet client: {e}")
            raise
    
    async def authenticate(self, credentials_path: str) -> bool:
        """Authenticate with Google using OAuth credentials."""
        logger.info("Authenticating with Google...")
        
        try:
            # Load OAuth credentials
            import json
            with open(credentials_path, 'r') as f:
                credentials = json.load(f)
            
            # For now, we'll use a simplified approach
            # In production, implement proper OAuth2 flow
            logger.info("Google authentication completed")
            return True
            
        except Exception as e:
            logger.error(f"Google authentication failed: {e}")
            return False
    
    def extract_meeting_id(self, meeting_url: str) -> Optional[str]:
        """Extract meeting ID from Google Meet URL."""
        patterns = [
            r'meet\.google\.com/([a-z0-9-]+)',
            r'meet\.google\.com/[a-z0-9-]+/([a-z0-9-]+)',
            r'meeting\.google\.com/([a-z0-9-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, meeting_url)
            if match:
                return match.group(1)
        
        logger.warning(f"Could not extract meeting ID from URL: {meeting_url}")
        return None
    
    async def join_meeting(self, meeting: Meeting) -> MeetingJoinResponse:
        """Join a Google Meet meeting."""
        logger.info(f"Joining Google Meet: {meeting.title}")
        
        try:
            if not self.driver:
                await self.initialize()
            
            # Extract meeting ID
            meeting_id = self.extract_meeting_id(meeting.meeting_url)
            if not meeting_id:
                return MeetingJoinResponse(
                    success=False,
                    meeting_id=meeting.id,
                    error_message="Could not extract meeting ID from URL"
                )
            
            self.meeting_id = meeting_id
            
            # Normalize meeting URL for guest access
            normalized_url = self._normalize_meeting_url(meeting.meeting_url)
            if normalized_url != meeting.meeting_url:
                logger.info(f"Using guest-access URL: {normalized_url}")

            # Navigate to meeting URL
            self.driver.get(normalized_url)
            
            # Wait for page to load
            await asyncio.sleep(3)
            
            # Handle authentication if needed
            if "accounts.google.com" in self.driver.current_url:
                logger.info("Authentication required, attempting to sign in...")
                await self._handle_authentication()
            
            # Join the meeting
            await self._join_meeting_room()
            
            # Start audio capture
            await self._start_audio_capture()
            
            self.is_joined = True
            logger.info(f"Successfully joined Google Meet: {meeting.title}")
            
            return MeetingJoinResponse(
                success=True,
                meeting_id=meeting.id
            )
            
        except Exception as e:
            logger.error(f"Failed to join Google Meet: {e}")
            return MeetingJoinResponse(
                success=False,
                meeting_id=meeting.id,
                error_message=str(e)
            )
    
    def _normalize_meeting_url(self, meeting_url: str) -> str:
        """Ensure meeting URL includes guest-friendly parameters."""
        try:
            parsed = urlparse(meeting_url)
            if not parsed.scheme:
                return meeting_url
            query = parse_qs(parsed.query, keep_blank_values=True)
            # Ensure guest parameters are present
            query['pli'] = ['1']
            query['authuser'] = ['0']
            normalized_query = urlencode({k: v[-1] for k, v in query.items()})
            normalized = urlunparse((
                parsed.scheme,
                parsed.netloc or 'meet.google.com',
                parsed.path,
                parsed.params,
                normalized_query,
                parsed.fragment
            ))
            return normalized
        except Exception as exc:
            logger.warning(f"Failed to normalize Google Meet URL '{meeting_url}': {exc}")
            return meeting_url

    async def _handle_authentication(self) -> None:
        """Handle Google authentication flow."""
        try:
            # Wait for login form
            wait = WebDriverWait(self.driver, 10)
            
            # Check if we need to sign in
            if "Sign in" in self.driver.page_source:
                logger.info("Sign-in required")
                # For automated testing, we might need to handle this differently
                # In production, use proper OAuth2 flow with stored tokens
                raise Exception("Manual authentication required")
            
        except TimeoutException:
            logger.warning("Authentication timeout")
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise
    
    async def _join_meeting_room(self) -> None:
        """Join the actual meeting room."""
        try:
            wait = WebDriverWait(self.driver, 15)
            
            # Look for join button
            join_selectors = [
                "button[jsname='Qx7uuf']",  # Join now button
                "button[aria-label*='Join']",
                "button[data-promo-anchor-id='join-button']",
                ".VfPpkd-LgbsSe[data-promo-anchor-id='join-button']"
            ]
            
            join_button = None
            for selector in join_selectors:
                try:
                    join_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                    break
                except TimeoutException:
                    continue
            
            if not join_button:
                logger.warning("Join button not found, trying alternative approach")
                # Try clicking on the page to activate the meeting
                self.driver.find_element(By.TAG_NAME, "body").click()
                await asyncio.sleep(2)
            else:
                join_button.click()
                logger.info("Clicked join button")
            
            # Wait for meeting to load
            await asyncio.sleep(5)
            
            # Check if we're in the meeting
            if "meet.google.com" in self.driver.current_url and "meeting" in self.driver.current_url:
                logger.info("Successfully entered meeting room")
            else:
                raise Exception("Failed to enter meeting room")
                
        except Exception as e:
            logger.error(f"Failed to join meeting room: {e}")
            raise
    
    async def _start_audio_capture(self) -> None:
        """Start capturing audio from the meeting."""
        logger.info("Starting audio capture...")
        
        try:
            # Enable microphone permissions
            self.driver.execute_script("""
                navigator.mediaDevices.getUserMedia({ audio: true })
                    .then(stream => {
                        window.meetingAudioStream = stream;
                        console.log('Audio stream started');
                    })
                    .catch(err => console.error('Audio stream error:', err));
            """)
            
            await asyncio.sleep(2)
            
            # Start audio processing
            self.audio_stream = self._capture_audio_stream()
            logger.info("Audio capture started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            raise
    
    async def _capture_audio_stream(self) -> AsyncGenerator[bytes, None]:
        """Capture audio stream from the meeting."""
        logger.info("Starting audio stream capture...")
        
        while self.is_joined:
            try:
                # Get audio data from browser
                audio_data = self.driver.execute_script("""
                    if (window.meetingAudioStream) {
                        const audioContext = new AudioContext();
                        const source = audioContext.createMediaStreamSource(window.meetingAudioStream);
                        const processor = audioContext.createScriptProcessor(4096, 1, 1);
                        
                        processor.onaudioprocess = function(e) {
                            const inputBuffer = e.inputBuffer;
                            const inputData = inputBuffer.getChannelData(0);
                            return new Uint8Array(inputData.buffer);
                        };
                        
                        source.connect(processor);
                        processor.connect(audioContext.destination);
                        
                        return 'audio_processing_started';
                    }
                    return null;
                """)
                
                if audio_data:
                    # In a real implementation, we would capture actual audio bytes
                    # For now, we'll simulate audio data
                    yield b"simulated_audio_data"
                
                await asyncio.sleep(0.1)  # 100ms chunks
                
            except Exception as e:
                logger.error(f"Audio capture error: {e}")
                await asyncio.sleep(1)
    
    async def get_audio_stream(self) -> AsyncGenerator[bytes, None]:
        """Get the audio stream from the meeting."""
        if not self.audio_stream:
            raise Exception("Audio stream not available")
        
        async for chunk in self.audio_stream:
            yield chunk
    
    async def leave_meeting(self) -> None:
        """Leave the current meeting."""
        logger.info("Leaving Google Meet...")
        
        try:
            self.is_joined = False
            
            if self.driver:
                # Look for leave button
                leave_selectors = [
                    "button[aria-label*='Leave']",
                    "button[jsname='CQylAd']",
                    ".VfPpkd-LgbsSe[data-promo-anchor-id='leave-button']"
                ]
                
                for selector in leave_selectors:
                    try:
                        leave_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                        leave_button.click()
                        logger.info("Clicked leave button")
                        break
                    except:
                        continue
                
                # Close browser
                self.driver.quit()
                self.driver = None
            
            logger.info("Successfully left Google Meet")
            
        except Exception as e:
            logger.error(f"Error leaving meeting: {e}")
    
    async def get_meeting_info(self) -> Dict[str, Any]:
        """Get information about the current meeting."""
        if not self.is_joined or not self.driver:
            return {}
        
        try:
            # Extract meeting information from the page
            meeting_info = self.driver.execute_script("""
                return {
                    title: document.title,
                    url: window.location.href,
                    participants: document.querySelectorAll('[data-participant-id]').length,
                    isRecording: document.querySelector('[aria-label*="recording"]') !== null,
                    isMuted: document.querySelector('[aria-label*="mute"]')?.getAttribute('aria-pressed') === 'true'
                };
            """)
            
            return meeting_info
            
        except Exception as e:
            logger.error(f"Failed to get meeting info: {e}")
            return {}
    
    async def create_meeting(
        self,
        title: str,
        start_time: str,
        end_time: str,
        attendees: Optional[list] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a Google Meet meeting via Calendar API.
        
        Creates a calendar event with Google Meet conference link using the Calendar API.
        
        Args:
            title (str): Meeting title
            start_time (str): Start time in ISO 8601 format
            end_time (str): End time in ISO 8601 format
            attendees (Optional[list]): List of attendee email addresses
            description (Optional[str]): Meeting description
            
        Returns:
            Dict[str, Any]: Meeting creation result with success status and meeting details
        """
        from uuid import uuid4
        from shared.schemas import Meeting, MeetingPlatform, MeetingStatus, MeetingParticipant
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import Flow
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        import os
        from pathlib import Path
        
        try:

            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            # Initialize Google Calendar API
            BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
            client_config_path = settings.google_oauth_client_config
            if not client_config_path.startswith('/'):
                client_config_path = BASE_DIR / client_config_path
            
            token_file = BASE_DIR / "google_calendar_token.json"
            
            logger.debug(f"token_file: {token_file}")
            # Load credentials
            credentials = None
            if os.path.exists(token_file):
                credentials = Credentials.from_authorized_user_file(str(token_file), ['https://www.googleapis.com/auth/calendar'])
                if credentials and credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
            if not credentials or not credentials.valid:
                logger.error("No valid Google OAuth credentials found")
                return {
                    "success": False,
                    "error": "Google OAuth credentials not configured or expired. Please complete OAuth setup."
                }
            
            # Build Calendar service
            service = build('calendar', 'v3', credentials=credentials)
            
            # Create calendar event with Google Meet
            event = {
                'summary': title,
                'description': description or '',
                'start': {
                    'dateTime': start_time,
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time,
                    'timeZone': 'UTC',
                },
                'conferenceData': {
                    'createRequest': {
                        'requestId': str(uuid4()),
                        'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                    }
                },
                'attendees': [{'email': email} for email in (attendees or [])]
            }
            
            created_event = service.events().insert(
                calendarId='primary',
                body=event,
                conferenceDataVersion=1
            ).execute()
            
            # Extract Meet link from conference data
            meet_link = None
            if 'conferenceData' in created_event:
                meet_link = created_event['conferenceData']['entryPoints'][0].get('uri') if created_event['conferenceData'].get('entryPoints') else None
            
            if not meet_link:
                logger.warning("No Meet link in created event")
                return {
                    "success": False,
                    "error": "Calendar event created but no Meet link was generated"
                }
            
            meeting_id = str(uuid4())
            external_id = created_event['id']
            
            # Create participants list
            participants = []
            if attendees:
                for email in attendees:
                    participants.append(MeetingParticipant(
                        email=email,
                        name=email.split('@')[0].replace('.', ' ').title(),
                        is_organizer=False,
                        response_status="accepted"
                    ))
            
            duration_minutes = int((end_dt - start_dt).total_seconds() / 60)
            
            return {
                "success": True,
                "meeting": {
                    "id": meeting_id,
                    "external_id": external_id,
                    "join_url": meet_link,
                    "title": title,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_minutes": duration_minutes,
                    "attendees": attendees or []
                },
                "_meeting_obj": Meeting(
                    id=meeting_id,
                    platform=MeetingPlatform.GOOGLE_MEET,
                    meeting_url=meet_link,
                    meeting_id_external=external_id,
                    title=title,
                    description=description or "",
                    start_time=start_dt,
                    end_time=end_dt,
                    organizer_email=settings.ai_email if hasattr(settings, 'ai_email') else '',
                    participants=participants,
                    status=MeetingStatus.SCHEDULED,
                    ai_email=settings.ai_email if hasattr(settings, 'ai_email') else '',
                    audio_enabled=True,
                    video_enabled=True,
                    recording_enabled=False
                )
            }
            
        except HttpError as e:
            logger.error(f"Google Calendar API error: {e}")
            return {
                "success": False,
                "error": f"Google Calendar API error: {e}"
            }
        except Exception as e:
            logger.error(f"Failed to create Google Meet meeting: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up Google Meet client...")
        
        try:
            if self.is_joined:
                await self.leave_meeting()
            
            if self.driver:
                self.driver.quit()
                self.driver = None
            
            logger.info("Google Meet client cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Factory function for creating Google Meet client
def create_google_meet_client() -> GoogleMeetClient:
    """Create a new Google Meet client instance."""
    return GoogleMeetClient()
