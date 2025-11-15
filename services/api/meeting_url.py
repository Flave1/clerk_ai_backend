"""
Utility helpers for creating meeting URLs across services.
"""
import logging
from datetime import datetime, timedelta

from fastapi import HTTPException

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def create_platform_meeting_url(platform: str, meeting_id: str) -> str:
    """
    Create a meeting URL using platform-specific clients.

    Returns the join URL for the meeting.
    """
    platform_lower = (platform or "").lower()
    now = datetime.utcnow()
    start_time = now.isoformat() + "Z"
    end_time = (now + timedelta(minutes=30)).isoformat() + "Z"

    if platform_lower in {"aurray", "internal"}:
        frontend_base = settings.frontend_base_url or "http://localhost:3000"
        return f"{frontend_base.rstrip('/')}/meeting_room?meetingId={meeting_id}"

    if platform_lower == "zoom":
        try:
            from services.meeting_agent.zoom_client import create_zoom_client

            client = create_zoom_client()
            await client.initialize()
            res = await client.create_meeting(
                title=f"Aurray Zoom Meeting {meeting_id[:8]}",
                start_time=start_time,
                end_time=end_time,
                attendees=[],
                description="Created via service layer",
            )
            if res.get("success"):
                return res["meeting"]["join_url"]
            error_msg = res.get("error", "Zoom API call did not return a valid join URL")
            logger.error(f"Zoom API call succeeded but didn't return a valid URL: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Failed to create Zoom meeting: {error_msg}")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Zoom URL generation via API failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Zoom URL generation failed: {exc}")

    if platform_lower in {"teams", "microsoft_teams"}:
        try:
            from services.meeting_agent.teams_client import create_teams_client

            client = create_teams_client()
            await client.initialize()
            await client.authenticate()
            res = await client.create_meeting(
                title=f"Aurray Teams Meeting {meeting_id[:8]}",
                start_time=start_time,
                end_time=end_time,
                attendees=[],
                description="Created via service layer",
            )
            if res.get("success"):
                return res["meeting"]["join_url"]
            logger.warning(f"Teams API call failed: {res.get('error', 'Unknown error')}. Using fallback URL.")
        except Exception as exc:
            logger.warning(f"Teams URL generation via API failed: {exc}. Using fallback URL.")

        # Fallback to static Teams meeting URL
        return "https://teams.live.com/meet/9318960718018?p=J453ke6nEPHvg5kJGq"

    if platform_lower == "google_meet":
        try:
            from services.meeting_agent.google_meet_client import create_google_meet_client

            client = create_google_meet_client()
            res = await client.create_meeting(
                title=f"Aurray Google Meet {meeting_id[:8]}",
                start_time=start_time,
                end_time=end_time,
                attendees=[],
                description="Created via service layer",
            )
            if res.get("success"):
                return res["meeting"]["join_url"]
            error_msg = res.get("error", "Failed to create Google Meet")
            logger.error(f"Google Meet creation failed: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Failed to create Google Meet: {error_msg}")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Google Meet creation failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Failed to create Google Meet: {exc}")

    raise HTTPException(status_code=400, detail=f"Unsupported platform: {platform}")


__all__ = ["create_platform_meeting_url"]


