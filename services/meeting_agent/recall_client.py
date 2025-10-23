"""
Recall.ai API client for meeting bot integration.

This client handles sending bots to meetings and retrieving transcriptions.
"""
import logging
from typing import Dict, Any, Optional
import aiohttp
from datetime import datetime

from shared.config import get_settings
from shared.schemas import Meeting

logger = logging.getLogger(__name__)
settings = get_settings()


class RecallClient:
    """Client for Recall.ai API integration."""
    
    def __init__(self):
        self.api_key = settings.recall_api_key
        # Default to us-west-2 region, update if needed
        # Options: us-east-1.recall.ai, us-west-2.recall.ai, eu-central-1.recall.ai, ap-northeast-1.recall.ai
        self.base_url = "https://us-west-2.recall.ai/api/v1"
        self.bot_id: Optional[str] = None
        
    async def send_bot_to_meeting(self, meeting: Meeting) -> Dict[str, Any]:
        """
        Send Recall bot to a meeting.
        
        Args:
            meeting: Meeting object with meeting details
            
        Returns:
            Dict with bot information including bot_id
        """
        if not self.api_key:
            logger.error("Recall API key not configured")
            return {
                "success": False,
                "error": "Recall API key not configured"
            }
        
        try:
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare bot request (simplified based on current API)
            bot_data = {
                "meeting_url": meeting.meeting_url,
                "bot_name": "Clerk AI Assistant",
                "recording": True,
                "transcription": True,
                "summary": True,
                "automatic_leave": {
                    "waiting_room_timeout": 600,  # 10 minutes
                    "noone_joined_timeout": 600
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/bot/",
                    json=bot_data,
                    headers=headers
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        self.bot_id = result.get("id")
                        
                        logger.info(f"✅ Recall bot sent to meeting: {meeting.title}")
                        logger.info(f"🤖 Bot ID: {self.bot_id}")
                        
                        return {
                            "success": True,
                            "bot_id": self.bot_id,
                            "bot_data": result
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Recall API error ({response.status}): {error_text}")
                        return {
                            "success": False,
                            "error": f"Recall API error: {error_text}"
                        }
                        
        except Exception as e:
            logger.error(f"Failed to send Recall bot: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_bot_status(self, bot_id: str) -> Dict[str, Any]:
        """
        Get status of a Recall bot.
        
        Args:
            bot_id: The bot ID from Recall
            
        Returns:
            Dict with bot status information
        """
        if not self.api_key:
            return {"success": False, "error": "API key not configured"}
        
        try:
            headers = {
                "Authorization": f"Token {self.api_key}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/bot/{bot_id}/",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "status": result.get("status_changes", [])[-1].get("code") if result.get("status_changes") else "unknown",
                            "data": result
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": error_text
                        }
                        
        except Exception as e:
            logger.error(f"Failed to get bot status: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_transcript(self, bot_id: str) -> Dict[str, Any]:
        """
        Get transcript from a Recall bot using the correct API endpoint.
        Based on: https://docs.recall.ai/reference/transcript_retrieve
        
        Args:
            bot_id: The bot ID from Recall
            
        Returns:
            Dict with transcript data
        """
        if not self.api_key:
            return {"success": False, "error": "API key not configured"}
        
        try:
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Accept": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                # First get the bot data to find transcript IDs
                async with session.get(
                    f"{self.base_url}/bot/{bot_id}/",
                    headers=headers
                ) as bot_response:
                    if bot_response.status == 200:
                        bot_data = await bot_response.json()
                        
                        # Get transcript IDs from the bot data
                        transcript_ids = bot_data.get("transcript_ids", [])
                        
                        if not transcript_ids or len(transcript_ids) == 0:
                            logger.warning(f"No transcript IDs found for bot {bot_id}")
                            return {
                                "success": True,
                                "transcript": "No transcript available yet. Recording may still be processing.",
                                "raw_data": {}
                            }
                        
                        # Get the first transcript using the new endpoint format
                        transcript_id = transcript_ids[0]
                        
                        async with session.get(
                            f"{self.base_url}/transcript/{transcript_id}/",
                            headers=headers
                        ) as transcript_response:
                            if transcript_response.status == 200:
                                transcript_data = await transcript_response.json()
                                
                                # Extract words and format into readable text
                                words_data = transcript_data.get("words", [])
                                
                                # Group by speaker and format
                                transcript_lines = []
                                current_speaker = None
                                current_text = []
                                
                                for word in words_data:
                                    speaker = word.get("speaker", "Unknown")
                                    text = word.get("text", "")
                                    
                                    if speaker != current_speaker:
                                        if current_text:
                                            transcript_lines.append(f"[{current_speaker}]: {' '.join(current_text)}")
                                        current_speaker = speaker
                                        current_text = [text]
                                    else:
                                        current_text.append(text)
                                
                                # Add the last speaker's text
                                if current_text:
                                    transcript_lines.append(f"[{current_speaker}]: {' '.join(current_text)}")
                                
                                transcript_text = "\n".join(transcript_lines)
                                
                                if not transcript_text:
                                    transcript_text = "Transcript is empty or still processing."
                                
                                return {
                                    "success": True,
                                    "transcript": transcript_text,
                                    "raw_data": transcript_data
                                }
                            else:
                                error_text = await transcript_response.text()
                                logger.error(f"Transcript retrieve error ({transcript_response.status}): {error_text}")
                                return {
                                    "success": False,
                                    "error": error_text
                                }
                    else:
                        error_text = await bot_response.text()
                        logger.error(f"Bot retrieve error ({bot_response.status}): {error_text}")
                        return {
                            "success": False,
                            "error": error_text
                        }
                        
        except Exception as e:
            logger.error(f"Failed to get transcript: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    async def leave_meeting(self, bot_id: str) -> Dict[str, Any]:
        """
        Make the bot leave the meeting.
        
        Args:
            bot_id: The bot ID from Recall
            
        Returns:
            Dict with success status
        """
        if not self.api_key:
            return {"success": False, "error": "API key not configured"}
        
        try:
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/bot/{bot_id}/leave/",
                    headers=headers
                ) as response:
                    if response.status in [200, 201, 204]:
                        logger.info(f"✅ Recall bot left meeting: {bot_id}")
                        return {"success": True}
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": error_text
                        }
                        
        except Exception as e:
            logger.error(f"Failed to make bot leave: {e}")
            return {"success": False, "error": str(e)}


def create_recall_client() -> RecallClient:
    """Factory function to create Recall client."""
    return RecallClient()

