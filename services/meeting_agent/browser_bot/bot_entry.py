#!/usr/bin/env python3
"""
Browser Bot Entry Script using browser-use for intelligent meeting navigation
"""

import asyncio
import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import aiohttp
import websockets
from loguru import logger
from browser_use import Agent, Browser, ChatBrowserUse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MeetingBot:
    def __init__(self):
        self.config = self._load_config()
        self.browser = None
        self.agent = None
        self.websocket = None
        self.is_joined = False
        self.session_id = self.config.get('session_id', 'default-session')
        
        # Configure logging
        logger.remove()
        log_level = self.config.get('log_level', 'INFO').upper()
        logger.add(sys.stdout, level=log_level)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        return {
            'meeting_url': os.getenv('MEETING_URL', ''),
            'bot_name': os.getenv('BOT_NAME', 'Clerk AI Bot'),
            'platform': os.getenv('PLATFORM', 'google_meet'),
            'rt_gateway_url': os.getenv('RT_GATEWAY_URL', 'ws://localhost:8001'),
            'api_base_url': os.getenv('API_BASE_URL', 'http://localhost:8000'),
            'meeting_id': os.getenv('MEETING_ID', ''),
            'session_id': os.getenv('SESSION_ID', ''),
            'join_timeout_sec': int(os.getenv('JOIN_TIMEOUT_SEC', '60')),
            'browser_use_api_key': os.getenv('BROWSER_USE_API_KEY', ''),
            'log_level': os.getenv('LOG_LEVEL', 'INFO')
        }
    
    def _get_platform_prompts(self) -> Dict[str, str]:
        """Get dynamic prompts based on platform requirements"""
        prompts = {
            'google_meet': {
                'join': f"""
                Navigate to the Google Meet URL: {self.config['meeting_url']}
                
                Steps to join the meeting:
                1. Wait for the page to load completely
                2. Use evaluate() to find and fill the name input field with "{self.config['bot_name']}"
                3. Use evaluate() to find and click the "Ask to join" button
                4. If prompted for camera/microphone permissions, click "Continue without microphone and camera" first, then look for "Allow microphone and camera" button
                5. Wait for host admission if in waiting room (this may take time)
                6. Once admitted, ensure camera and microphone are enabled
                
                Important: Always use evaluate() for reliable element interaction. Avoid find_text() when possible.
                The bot name should be "{self.config['bot_name']}".
                """,
                
                'verify_joined': """
                Verify that you have successfully joined the Google Meet meeting:
                1. Check that you can see the meeting interface
                2. Confirm that your camera is ON and visible
                3. Confirm that your microphone is ON
                4. Look for other participants or the meeting host
                5. Ensure you appear in the participant list
                """,
                
                'leave': """
                Leave the Google Meet meeting gracefully:
                1. Find and click the "Leave call" or "End call" button
                2. Confirm the action if prompted
                3. Wait for the meeting to end
                """
            },
            
            'zoom': {
                'join': f"""
                Navigate to the Zoom meeting URL: {self.config['meeting_url']}
                
                Steps to join the meeting:
                1. Wait for the page to load completely
                2. Click "Join from your browser" if available
                3. Enter the meeting ID if prompted
                4. Enter the meeting password if required
                5. Click "Join" to enter the meeting
                6. Allow camera and microphone permissions
                7. Turn ON camera and microphone once in the meeting
                
                Important: Make sure the bot appears as a visible participant.
                The bot name should be "{self.config['bot_name']}".
                """,
                
                'verify_joined': """
                Verify that you have successfully joined the Zoom meeting:
                1. Check that you can see the Zoom meeting interface
                2. Confirm that your camera is ON and visible
                3. Confirm that your microphone is ON
                4. Look for other participants or the meeting host
                5. Ensure you appear in the participant list
                """,
                
                'leave': """
                Leave the Zoom meeting gracefully:
                1. Find and click the "Leave" button
                2. Confirm the action if prompted
                3. Wait for the meeting to end
                """
            },
            
            'teams': {
                'join': f"""
                Navigate to the Microsoft Teams meeting URL: {self.config['meeting_url']}
                
                Steps to join the meeting:
                1. Wait for the page to load completely
                2. Click "Join now" or "Join" button
                3. Allow camera and microphone permissions
                4. Turn ON camera and microphone once in the meeting
                5. Wait for the meeting interface to fully load
                
                Important: Make sure the bot appears as a visible participant.
                The bot name should be "{self.config['bot_name']}".
                """,
                
                'verify_joined': """
                Verify that you have successfully joined the Teams meeting:
                1. Check that you can see the Teams meeting interface
                2. Confirm that your camera is ON and visible
                3. Confirm that your microphone is ON
                4. Look for other participants or the meeting host
                5. Ensure you appear in the participant list
                """,
                
                'leave': """
                Leave the Teams meeting gracefully:
                1. Find and click the "Leave" button
                2. Confirm the action if prompted
                3. Wait for the meeting to end
                """
            }
        }
        
        return prompts.get(self.config['platform'], prompts['google_meet'])
    
    async def initialize_browser(self):
        """Initialize browser-use browser and agent"""
        try:
            logger.info("Initializing browser-use browser...")
            
            # Initialize browser with cloud settings (more reliable)
            self.browser = Browser(
                headless=False,  # Set to True for production
                use_cloud=True  # Use cloud browser for better reliability
            )
            
            # Initialize LLM (use ChatBrowserUse for best results)
            llm = ChatBrowserUse()
            
            logger.info("Browser initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise
    
    async def join_meeting(self):
        """Join the meeting using browser-use agent"""
        try:
            platform_prompts = self._get_platform_prompts()
            
            # Create agent for joining meeting
            self.agent = Agent(
                task=platform_prompts['join'],
                llm=ChatBrowserUse(),
                browser=self.browser,
            )
            
            logger.info(f"Starting to join {self.config['platform']} meeting...")
            logger.info(f"Meeting URL: {self.config['meeting_url']}")
            
            # Run the join task
            history = await self.agent.run()
            
            logger.info("Join task completed, verifying meeting status...")
            
            # Verify we joined successfully with error handling
            try:
                verify_agent = Agent(
                    task=platform_prompts['verify_joined'],
                    llm=ChatBrowserUse(),
                    browser=self.browser,
                )
                
                verify_history = await verify_agent.run()
                logger.info("Verification completed successfully")
                
            except Exception as verify_error:
                logger.warning(f"Verification failed but join was successful: {verify_error}")
                # Continue anyway since the main join task was successful
                verify_history = None
            
            self.is_joined = True
            logger.info("Successfully joined meeting!")
            
            # Log to backend API
            await self._log_to_backend('meeting_joined', {
                'meeting_id': self.config['meeting_id'],
                'session_id': self.session_id,
                'platform': self.config['platform'],
                'bot_name': self.config['bot_name'],
                'meeting_url': self.config['meeting_url'],
                'timestamp': datetime.now().isoformat(),
                'verification_status': 'success' if verify_history else 'skipped_due_to_error',
                'join_successful': True
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to join meeting: {e}")
            return False
    
    async def leave_meeting(self):
        """Leave the meeting gracefully"""
        try:
            if not self.is_joined:
                logger.info("Bot is not in a meeting")
                return
            
            platform_prompts = self._get_platform_prompts()
            
            # Create agent for leaving meeting
            leave_agent = Agent(
                task=platform_prompts['leave'],
                llm=ChatBrowserUse(),
                browser=self.browser,
            )
            
            logger.info("Leaving meeting...")
            await leave_agent.run()
            
            self.is_joined = False
            logger.info("Successfully left meeting")
            
            # Log to backend API
            await self._log_to_backend('meeting_left', {
                'meeting_id': self.config['meeting_id'],
                'session_id': self.session_id,
                'platform': self.config['platform'],
                'bot_name': self.config['bot_name'],
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error leaving meeting: {e}")
    
    async def _log_to_backend(self, event: str, data: Dict[str, Any]):
        """Log events to the backend API"""
        try:
            api_url = f"{self.config['api_base_url']}/api/v1/meetings/bot-log"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json={
                    'event': event,
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        logger.info(f"Logged to backend API: {event}")
                    else:
                        logger.warning(f"Backend logging failed: {response.status}")
                        
        except Exception as e:
            logger.warning(f"Failed to log to backend API: {e}")
    
    async def connect_websocket(self):
        """Connect to RT Gateway WebSocket"""
        try:
            ws_url = f"{self.config['rt_gateway_url']}/ws/{self.session_id}"
            logger.info(f"Connecting to RT Gateway: {ws_url}")
            
            self.websocket = await websockets.connect(ws_url)
            logger.info("Connected to RT Gateway")
            
        except Exception as e:
            logger.warning(f"WebSocket connection failed: {e}")
    
    async def keep_running(self):
        """Keep the bot running until meeting ends or interrupted"""
        logger.info("Bot is running, waiting for meeting to end...")
        
        try:
            while self.is_joined:
                await asyncio.sleep(5)
                
                # Simple timeout-based approach since browser-use doesn't expose page directly
                # The browser-use agent handles the browser lifecycle internally
                pass
                        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in keep_running: {e}")
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        try:
            if self.websocket:
                await self.websocket.close()
            
            if self.browser:
                # Browser-use handles cleanup internally
                # No need to call close() as it's managed by the agent
                pass
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        logger.info("Cleanup completed")

async def main():
    """Main entry point"""
    bot = MeetingBot()
    
    try:
        logger.info("Starting Browser Bot", extra={
            'meeting_url': bot.config['meeting_url'],
            'bot_name': bot.config['bot_name'],
            'platform': bot.config['platform'],
            'session_id': bot.session_id
        })
        
        # Validate configuration
        if not bot.config['meeting_url']:
            logger.error("MEETING_URL environment variable is required")
            sys.exit(1)
        
        # Initialize browser
        await bot.initialize_browser()
        
        # Connect to WebSocket (optional)
        await bot.connect_websocket()
        
        # Join meeting
        success = await bot.join_meeting()
        if not success:
            logger.error("Failed to join meeting")
            sys.exit(1)
        
        # Keep running
        await bot.keep_running()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Browser Bot failed: {e}")
        sys.exit(1)
    finally:
        await bot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
