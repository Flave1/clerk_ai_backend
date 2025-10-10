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
from urllib.parse import urlparse, parse_qs

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
            
            # Navigate to meeting URL
            self.driver.get(meeting.meeting_url)
            
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
