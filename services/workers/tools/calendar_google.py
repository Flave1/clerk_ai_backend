"""
Google Calendar integration tool using OAuth authentication.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from shared.config import get_settings

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # points to clerk_backend/

logger = logging.getLogger(__name__)
settings = get_settings()

# Google Calendar API scopes
SCOPES = ["https://www.googleapis.com/auth/calendar"]


class GoogleCalendarTool:
    """Google Calendar integration tool using OAuth."""

    def __init__(self):
        self.service = None
        self.credentials = None
        self.token_file = None
        self._initialize_service()

    def _initialize_service(self):
        """Initialize Google Calendar service with OAuth."""
        try:
            # Check if OAuth client configuration exists
            if not settings.google_oauth_client_config:
                logger.warning("Google OAuth client configuration not found. Calendar will use mock responses.")
                return
            
            # Handle relative paths by resolving them relative to the project root
            client_config_path = settings.google_oauth_client_config

            if not client_config_path.startswith('/'):
                client_config_path = BASE_DIR / client_config_path
            # Set up token file path
            self.token_file = BASE_DIR / "google_calendar_token.json"
            
            # Load or create credentials
            self.credentials = self._load_or_create_credentials(str(client_config_path))
            
            if self.credentials and self.credentials.valid:
                # Build service with valid credentials
                self.service = build("calendar", "v3", credentials=self.credentials)
                logger.info("Google Calendar service initialized with OAuth")
            else:
                logger.warning("Google Calendar OAuth credentials not valid. Calendar will use mock responses.")
                self.service = None

        except Exception as e:
            logger.error(f"Failed to initialize Google Calendar service: {e}")
            self.service = None

    def _load_or_create_credentials(self, client_config_path: str) -> Optional[Credentials]:
        """Load existing credentials or create new ones via OAuth flow."""
        try:
            # Load existing token
            if os.path.exists(self.token_file):
                credentials = Credentials.from_authorized_user_file(str(self.token_file), SCOPES)
                if credentials and credentials.valid:
                    return credentials
                elif credentials and credentials.expired and credentials.refresh_token:
                    # Refresh expired credentials
                    credentials.refresh(Request())
                    self._save_credentials(credentials)
                    return credentials
            
            # No valid credentials - return None (will use mock responses)
            logger.warning("No valid OAuth credentials found. Please run the OAuth setup.")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load/create credentials: {e}")
            return None

    def _save_credentials(self, credentials: Credentials):
        """Save credentials to token file."""
        try:
            with open(self.token_file, 'w') as token:
                token.write(credentials.to_json())
            logger.info("OAuth credentials saved successfully")
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")

    def setup_oauth_flow(self):
        """Setup OAuth flow and return authorization URL."""
        try:
            if not settings.google_oauth_client_config:
                raise ValueError("Google OAuth client configuration not found")
            
            client_config_path = settings.google_oauth_client_config
            if not client_config_path.startswith('/'):
                client_config_path = BASE_DIR / client_config_path
            
            # Create flow from client configuration
            flow = Flow.from_client_secrets_file(
                str(client_config_path),
                scopes=SCOPES,
                redirect_uri='http://localhost'  # Match the redirect URI in the config
            )
            
            # Get authorization URL
            auth_url, _ = flow.authorization_url(
                access_type='offline',
                prompt='consent'
            )
            
            return auth_url
            
        except Exception as e:
            logger.error(f"Failed to setup OAuth flow: {e}")
            raise

    def complete_oauth_flow(self, authorization_code: str):
        """Complete OAuth flow with authorization code."""
        try:
            if not settings.google_oauth_client_config:
                raise ValueError("Google OAuth client configuration not found")
            
            client_config_path = settings.google_oauth_client_config
            if not client_config_path.startswith('/'):
                client_config_path = BASE_DIR / client_config_path
            # Create flow from client configuration
            flow = Flow.from_client_secrets_file(
                str(client_config_path),
                scopes=SCOPES,
                redirect_uri='http://localhost'  # Match the redirect URI in the config
            )
            
            # Exchange authorization code for credentials
            flow.fetch_token(code=authorization_code)
            credentials = flow.credentials
            
            # Save credentials
            self._save_credentials(credentials)
            self.credentials = credentials
            
            # Build service
            self.service = build("calendar", "v3", credentials=credentials)
            logger.info("OAuth flow completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete OAuth flow: {e}")
            return False

    async def create_event(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a calendar event."""
        try:
            if not self.service:
                logger.warning("Google Calendar service not initialized")
                return {
                    "success": False,
                    "error": "Google Calendar OAuth not configured. Please run the OAuth setup."
                }

            # Extract parameters
            title = parameters.get("title", "New Event")
            start_time = parameters.get("start_time")
            end_time = parameters.get("end_time")
            attendees = parameters.get("attendees", [])
            description = parameters.get("description", "")

            if not start_time or not end_time:
                return {
                    "success": False,
                    "error": "start_time and end_time are required",
                }

            # Create event
            event = {
                "summary": title,
                "description": description,
                "start": {
                    "dateTime": start_time,
                    "timeZone": "UTC",
                },
                "end": {
                    "dateTime": end_time,
                    "timeZone": "UTC",
                },
                "attendees": [{"email": email} for email in attendees],
                "reminders": {
                    "useDefault": False,
                    "overrides": [
                        {"method": "email", "minutes": 24 * 60},
                        {"method": "popup", "minutes": 10},
                    ],
                },
            }

            # Insert event
            event_result = (
                self.service.events().insert(calendarId="primary", body=event).execute()
            )

            logger.info(f"Created calendar event: {event_result['id']}")

            return {
                "success": True,
                "result": {
                    "event_id": event_result["id"],
                    "event_link": event_result.get("htmlLink"),
                    "title": title,
                    "start_time": start_time,
                    "end_time": end_time,
                    "attendees": attendees,
                },
            }

        except HttpError as e:
            logger.error(f"Google Calendar API error: {e}")
            return {"success": False, "error": f"Calendar API error: {e}"}
        except Exception as e:
            logger.error(f"Failed to create calendar event: {e}")
            return {"success": False, "error": str(e)}

    async def update_event(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update a calendar event."""
        try:
            if not self.service:
                logger.warning("Google Calendar service not initialized")
                return {
                    "success": False,
                    "error": "Google Calendar OAuth not configured. Please run the OAuth setup."
                }

            event_id = parameters.get("event_id")
            if not event_id:
                return {"success": False, "error": "event_id is required"}

            # Get existing event
            event = (
                self.service.events()
                .get(calendarId="primary", eventId=event_id)
                .execute()
            )

            # Update fields
            if "title" in parameters:
                event["summary"] = parameters["title"]
            if "start_time" in parameters:
                event["start"]["dateTime"] = parameters["start_time"]
            if "end_time" in parameters:
                event["end"]["dateTime"] = parameters["end_time"]
            if "description" in parameters:
                event["description"] = parameters["description"]
            if "attendees" in parameters:
                event["attendees"] = [
                    {"email": email} for email in parameters["attendees"]
                ]

            # Update event
            updated_event = (
                self.service.events()
                .update(calendarId="primary", eventId=event_id, body=event)
                .execute()
            )

            logger.info(f"Updated calendar event: {event_id}")

            return {
                "success": True,
                "result": {
                    "event_id": updated_event["id"],
                    "title": updated_event.get("summary"),
                    "updated_at": datetime.utcnow().isoformat(),
                },
            }

        except HttpError as e:
            logger.error(f"Google Calendar API error: {e}")
            return {"success": False, "error": f"Calendar API error: {e}"}
        except Exception as e:
            logger.error(f"Failed to update calendar event: {e}")
            return {"success": False, "error": str(e)}

    async def delete_event(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a calendar event."""
        try:
            if not self.service:
                logger.warning("Google Calendar service not initialized")
                return {
                    "success": False,
                    "error": "Google Calendar OAuth not configured. Please run the OAuth setup."
                }

            event_id = parameters.get("event_id")
            if not event_id:
                return {"success": False, "error": "event_id is required"}

            # Delete event
            self.service.events().delete(
                calendarId="primary", eventId=event_id
            ).execute()

            logger.info(f"Deleted calendar event: {event_id}")

            return {
                "success": True,
                "result": {
                    "event_id": event_id,
                    "deleted_at": datetime.utcnow().isoformat(),
                },
            }

        except HttpError as e:
            logger.error(f"Google Calendar API error: {e}")
            return {"success": False, "error": f"Calendar API error: {e}"}
        except Exception as e:
            logger.error(f"Failed to delete calendar event: {e}")
            return {"success": False, "error": str(e)}

    async def list_events(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """List calendar events."""
        try:
            if not self.service:
                logger.warning("Google Calendar service not initialized")
                return {
                    "success": False,
                    "error": "Google Calendar OAuth not configured. Please run the OAuth setup."
                }

            # Get parameters
            start_time = parameters.get("start_time", datetime.utcnow().isoformat())
            end_time = parameters.get("end_time")
            max_results = parameters.get("max_results", 10)

            # List events
            events_result = (
                self.service.events()
                .list(
                    calendarId="primary",
                    timeMin=start_time,
                    timeMax=end_time,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])

            # Format events
            formatted_events = []
            for event in events:
                start = event["start"].get("dateTime", event["start"].get("date"))
                formatted_events.append(
                    {
                        "id": event["id"],
                        "title": event.get("summary", "No Title"),
                        "start": start,
                        "description": event.get("description", ""),
                        "attendees": [
                            attendee.get("email")
                            for attendee in event.get("attendees", [])
                        ],
                    }
                )

            return {
                "success": True,
                "result": {"events": formatted_events, "count": len(formatted_events)},
            }

        except HttpError as e:
            logger.error(f"Google Calendar API error: {e}")
            return {"success": False, "error": f"Calendar API error: {e}"}
        except Exception as e:
            logger.error(f"Failed to list calendar events: {e}")
            return {"success": False, "error": str(e)}
