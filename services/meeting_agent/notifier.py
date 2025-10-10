"""
Notification service for sending meeting summaries and updates.

This module handles sending email and Slack notifications with meeting
summaries, action items, and other meeting-related information.
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import aiohttp
from slack_sdk.web.async_client import AsyncWebClient

from shared.config import get_settings
from shared.schemas import Meeting, MeetingSummary, MeetingNotification, ActionItem

logger = logging.getLogger(__name__)
settings = get_settings()


class NotificationService:
    """Service for sending meeting notifications via email and Slack."""
    
    def __init__(self):
        self.slack_client: Optional[AsyncWebClient] = None
        self.smtp_server: Optional[smtplib.SMTP] = None
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the notification service."""
        logger.info("Initializing notification service...")
        
        try:
            # Initialize Slack client
            if settings.slack_bot_token:
                self.slack_client = AsyncWebClient(token=settings.slack_bot_token)
                logger.info("Slack client initialized")
            
            # Initialize SMTP server
            if settings.smtp_username and settings.smtp_password:
                self.smtp_server = smtplib.SMTP(settings.smtp_host, settings.smtp_port)
                self.smtp_server.starttls()
                self.smtp_server.login(settings.smtp_username, settings.smtp_password)
                logger.info("SMTP server initialized")
            
            self.is_initialized = True
            logger.info("Notification service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize notification service: {e}")
            raise
    
    async def send_meeting_summary(self, meeting: Meeting, summary: MeetingSummary) -> bool:
        """Send meeting summary to participants."""
        logger.info(f"Sending meeting summary for: {meeting.title}")
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Prepare notification content
            notification = await self._prepare_summary_notification(meeting, summary)
            
            # Send via email
            email_success = await self._send_email_notification(notification)
            
            # Send via Slack
            slack_success = await self._send_slack_notification(notification)
            
            success = email_success or slack_success
            
            if success:
                logger.info(f"Successfully sent meeting summary for: {meeting.title}")
            else:
                logger.error(f"Failed to send meeting summary for: {meeting.title}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending meeting summary: {e}")
            return False
    
    async def send_action_items(self, meeting: Meeting, action_items: List[ActionItem]) -> bool:
        """Send action items to relevant participants."""
        logger.info(f"Sending action items for: {meeting.title}")
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Prepare notification content
            notification = await self._prepare_action_items_notification(meeting, action_items)
            
            # Send via email
            email_success = await self._send_email_notification(notification)
            
            # Send via Slack
            slack_success = await self._send_slack_notification(notification)
            
            success = email_success or slack_success
            
            if success:
                logger.info(f"Successfully sent action items for: {meeting.title}")
            else:
                logger.error(f"Failed to send action items for: {meeting.title}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending action items: {e}")
            return False
    
    async def send_meeting_reminder(self, meeting: Meeting) -> bool:
        """Send meeting reminder to participants."""
        logger.info(f"Sending meeting reminder for: {meeting.title}")
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Prepare notification content
            notification = await self._prepare_reminder_notification(meeting)
            
            # Send via email
            email_success = await self._send_email_notification(notification)
            
            # Send via Slack
            slack_success = await self._send_slack_notification(notification)
            
            success = email_success or slack_success
            
            if success:
                logger.info(f"Successfully sent meeting reminder for: {meeting.title}")
            else:
                logger.error(f"Failed to send meeting reminder for: {meeting.title}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending meeting reminder: {e}")
            return False
    
    async def _prepare_summary_notification(self, meeting: Meeting, summary: MeetingSummary) -> MeetingNotification:
        """Prepare meeting summary notification."""
        # Create email content
        email_content = f"""
Meeting Summary: {meeting.title}

Date: {meeting.start_time.strftime('%Y-%m-%d %H:%M')}
Duration: {summary.duration_minutes} minutes
Platform: {meeting.platform.value}

Summary:
{summary.summary_text}

Topics Discussed:
{chr(10).join(f"• {topic}" for topic in summary.topics_discussed)}

Key Decisions:
{chr(10).join(f"• {decision}" for decision in summary.key_decisions)}

Action Items:
{chr(10).join(f"• {item.description} (Assignee: {item.assignee or 'TBD'}, Due: {item.due_date or 'TBD'}, Priority: {item.priority})" for item in summary.action_items)}

Sentiment: {summary.sentiment}

---
This summary was generated by AI Meeting Assistant.
"""
        
        # Create Slack content
        slack_content = f"""
*Meeting Summary: {meeting.title}*

📅 *Date:* {meeting.start_time.strftime('%Y-%m-%d %H:%M')}
⏱️ *Duration:* {summary.duration_minutes} minutes
🖥️ *Platform:* {meeting.platform.value}

*Summary:*
{summary.summary_text}

*Topics Discussed:*
{chr(10).join(f"• {topic}" for topic in summary.topics_discussed)}

*Key Decisions:*
{chr(10).join(f"• {decision}" for decision in summary.key_decisions)}

*Action Items:*
{chr(10).join(f"• {item.description} (Assignee: {item.assignee or 'TBD'}, Due: {item.due_date or 'TBD'}, Priority: {item.priority})" for item in summary.action_items)}

*Sentiment:* {summary.sentiment}

---
_This summary was generated by AI Meeting Assistant._
"""
        
        # Get recipients
        recipients = [participant.email for participant in meeting.participants]
        
        return MeetingNotification(
            meeting_id=meeting.id,
            notification_type="summary",
            recipients=recipients,
            subject=f"Meeting Summary: {meeting.title}",
            content=email_content,
            sent_at=datetime.utcnow(),
            delivery_status="pending"
        )
    
    async def _prepare_action_items_notification(self, meeting: Meeting, action_items: List[ActionItem]) -> MeetingNotification:
        """Prepare action items notification."""
        # Create email content
        email_content = f"""
Action Items from Meeting: {meeting.title}

Date: {meeting.start_time.strftime('%Y-%m-%d %H:%M')}

Action Items:
{chr(10).join(f"• {item.description} (Assignee: {item.assignee or 'TBD'}, Due: {item.due_date or 'TBD'}, Priority: {item.priority})" for item in action_items)}

Please review and update the status of your assigned action items.

---
This notification was generated by AI Meeting Assistant.
"""
        
        # Create Slack content
        slack_content = f"""
*Action Items from Meeting: {meeting.title}*

📅 *Date:* {meeting.start_time.strftime('%Y-%m-%d %H:%M')}

*Action Items:*
{chr(10).join(f"• {item.description} (Assignee: {item.assignee or 'TBD'}, Due: {item.due_date or 'TBD'}, Priority: {item.priority})" for item in action_items)}

Please review and update the status of your assigned action items.

---
_This notification was generated by AI Meeting Assistant._
"""
        
        # Get recipients
        recipients = [participant.email for participant in meeting.participants]
        
        return MeetingNotification(
            meeting_id=meeting.id,
            notification_type="action_items",
            recipients=recipients,
            subject=f"Action Items from Meeting: {meeting.title}",
            content=email_content,
            sent_at=datetime.utcnow(),
            delivery_status="pending"
        )
    
    async def _prepare_reminder_notification(self, meeting: Meeting) -> MeetingNotification:
        """Prepare meeting reminder notification."""
        # Create email content
        email_content = f"""
Meeting Reminder: {meeting.title}

Date: {meeting.start_time.strftime('%Y-%m-%d %H:%M')}
Duration: {meeting.end_time - meeting.start_time}
Platform: {meeting.platform.value}

Meeting URL: {meeting.meeting_url}

Description:
{meeting.description or 'No description available'}

---
This reminder was sent by AI Meeting Assistant.
"""
        
        # Create Slack content
        slack_content = f"""
*Meeting Reminder: {meeting.title}*

📅 *Date:* {meeting.start_time.strftime('%Y-%m-%d %H:%M')}
⏱️ *Duration:* {meeting.end_time - meeting.start_time}
🖥️ *Platform:* {meeting.platform.value}

*Meeting URL:* {meeting.meeting_url}

*Description:*
{meeting.description or 'No description available'}

---
_This reminder was sent by AI Meeting Assistant._
"""
        
        # Get recipients
        recipients = [participant.email for participant in meeting.participants]
        
        return MeetingNotification(
            meeting_id=meeting.id,
            notification_type="reminder",
            recipients=recipients,
            subject=f"Meeting Reminder: {meeting.title}",
            content=email_content,
            sent_at=datetime.utcnow(),
            delivery_status="pending"
        )
    
    async def _send_email_notification(self, notification: MeetingNotification) -> bool:
        """Send email notification."""
        try:
            if not self.smtp_server:
                logger.warning("SMTP server not configured")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = settings.smtp_username
            msg['To'] = ', '.join(notification.recipients)
            msg['Subject'] = notification.subject
            
            # Add body
            msg.attach(MIMEText(notification.content, 'plain'))
            
            # Send email
            self.smtp_server.send_message(msg)
            
            logger.info(f"Email notification sent to: {', '.join(notification.recipients)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    async def _send_slack_notification(self, notification: MeetingNotification) -> bool:
        """Send Slack notification."""
        try:
            if not self.slack_client:
                logger.warning("Slack client not configured")
                return False
            
            # Send to each recipient's Slack channel
            for recipient in notification.recipients:
                try:
                    # In a real implementation, you would look up the user's Slack ID
                    # For now, we'll send to a general channel
                    channel = "#meetings"  # Default channel
                    
                    await self.slack_client.chat_postMessage(
                        channel=channel,
                        text=notification.content,
                        username="AI Meeting Assistant",
                        icon_emoji=":robot_face:"
                    )
                    
                    logger.info(f"Slack notification sent to: {recipient}")
                    
                except Exception as e:
                    logger.error(f"Failed to send Slack notification to {recipient}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    async def send_custom_notification(self, recipients: List[str], subject: str, content: str) -> bool:
        """Send custom notification to specified recipients."""
        logger.info(f"Sending custom notification to: {', '.join(recipients)}")
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Create notification
            notification = MeetingNotification(
                meeting_id="custom",
                notification_type="custom",
                recipients=recipients,
                subject=subject,
                content=content,
                sent_at=datetime.utcnow(),
                delivery_status="pending"
            )
            
            # Send via email
            email_success = await self._send_email_notification(notification)
            
            # Send via Slack
            slack_success = await self._send_slack_notification(notification)
            
            success = email_success or slack_success
            
            if success:
                logger.info(f"Successfully sent custom notification to: {', '.join(recipients)}")
            else:
                logger.error(f"Failed to send custom notification to: {', '.join(recipients)}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending custom notification: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup notification service resources."""
        logger.info("Cleaning up notification service...")
        
        try:
            if self.smtp_server:
                self.smtp_server.quit()
                self.smtp_server = None
            
            self.is_initialized = False
            logger.info("Notification service cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Factory function for creating notification service
def create_notification_service() -> NotificationService:
    """Create a new notification service instance."""
    return NotificationService()
