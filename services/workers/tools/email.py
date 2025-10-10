"""
Email integration tool using SMTP.
"""
import logging
from typing import Any, Dict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmailTool:
    """Email integration tool using SMTP."""

    def __init__(self):
        self.smtp_host = None
        self.smtp_port = None
        self.smtp_username = None
        self.smtp_password = None
        self._initialize_settings()

    def _initialize_settings(self):
        """Initialize SMTP settings."""
        try:
            self.smtp_host = settings.smtp_host
            self.smtp_port = settings.smtp_port
            self.smtp_username = settings.smtp_username
            self.smtp_password = settings.smtp_password

            if not all([self.smtp_host, self.smtp_port, self.smtp_username, self.smtp_password]):
                logger.warning("SMTP settings not fully configured")
                return

            logger.info("Email tool initialized with SMTP settings")

        except Exception as e:
            logger.error(f"Failed to initialize email tool: {e}")

    async def send_email(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send an email via SMTP."""
        try:
            if not all([self.smtp_host, self.smtp_port, self.smtp_username, self.smtp_password]):
                logger.warning("SMTP settings not configured")
                return {
                    "success": False,
                    "error": "SMTP settings not configured. Please configure SMTP_HOST, SMTP_PORT, SMTP_USERNAME, and SMTP_PASSWORD."
                }

            # Extract parameters
            to_email = parameters.get("to_email")
            subject = parameters.get("subject", "Message from AI Receptionist")
            message = parameters.get("message", "")
            from_email = parameters.get("from_email", self.smtp_username)

            if not to_email or not message:
                return {
                    "success": False,
                    "error": "to_email and message are required"
                }

            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject

            # Add body to email
            msg.attach(MIMEText(message, 'plain'))

            # Create SMTP session
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()  # Enable security
            server.login(self.smtp_username, self.smtp_password)

            # Send email
            text = msg.as_string()
            server.sendmail(from_email, to_email, text)
            server.quit()

            logger.info(f"Sent email to {to_email}: {subject}")

            return {
                "success": True,
                "result": {
                    "to_email": to_email,
                    "from_email": from_email,
                    "subject": subject,
                    "message": message,
                },
            }

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication error: {e}")
            return {
                "success": False,
                "error": f"SMTP authentication failed: {str(e)}"
            }
        except smtplib.SMTPRecipientsRefused as e:
            logger.error(f"SMTP recipient refused: {e}")
            return {
                "success": False,
                "error": f"Recipient email refused: {str(e)}"
            }
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return {
                "success": False,
                "error": f"SMTP error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return {"success": False, "error": str(e)}

    async def send_html_email(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send an HTML email via SMTP."""
        try:
            if not all([self.smtp_host, self.smtp_port, self.smtp_username, self.smtp_password]):
                return {
                    "success": False,
                    "error": "SMTP settings not configured"
                }

            # Extract parameters
            to_email = parameters.get("to_email")
            subject = parameters.get("subject", "Message from AI Receptionist")
            html_message = parameters.get("html_message", "")
            from_email = parameters.get("from_email", self.smtp_username)

            if not to_email or not html_message:
                return {
                    "success": False,
                    "error": "to_email and html_message are required"
                }

            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject

            # Add HTML body to email
            msg.attach(MIMEText(html_message, 'html'))

            # Create SMTP session
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()  # Enable security
            server.login(self.smtp_username, self.smtp_password)

            # Send email
            text = msg.as_string()
            server.sendmail(from_email, to_email, text)
            server.quit()

            logger.info(f"Sent HTML email to {to_email}: {subject}")

            return {
                "success": True,
                "result": {
                    "to_email": to_email,
                    "from_email": from_email,
                    "subject": subject,
                    "html_message": html_message,
                },
            }

        except Exception as e:
            logger.error(f"Failed to send HTML email: {e}")
            return {"success": False, "error": str(e)}
