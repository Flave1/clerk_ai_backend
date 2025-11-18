"""
Meeting context helper utilities with in-process caching.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from shared.schemas import Meeting, MeetingContext, MeetingRole, TonePersonality

from .service_tool import ServiceTool

_CACHE_LOCK = asyncio.Lock()
_CACHE_STORE: Dict[str, Tuple[float, str]] = {}


class ServiceMeetingContext:
    """Provides meeting context utilities, including caching payload preparation."""

    CACHE_TTL_SECONDS = 60 * 60  # 1 hour

    def __init__(self, dao):
        self.dao = dao
        self.tool_service = ServiceTool()

    async def create_context(self, meeting_context: MeetingContext) -> MeetingContext:
        """Persist a new meeting context."""
        return await self.dao.create_meeting_context(meeting_context)

    async def get_context(self, context_id: str, user_id: str) -> Optional[MeetingContext]:
        """Fetch a meeting context by ID for a specific user."""
        return await self.dao.get_meeting_context(context_id, user_id)

    async def get_context_by_id(self, context_id: str) -> Optional[MeetingContext]:
        """Fetch a meeting context without enforcing user ownership."""
        return await self.dao.get_meeting_context_by_id(context_id)

    async def get_contexts_by_user(self, user_id: str) -> List[MeetingContext]:
        """List all meeting contexts for a user."""
        return await self.dao.get_meeting_contexts_by_user(user_id)

    async def update_context(self, meeting_context: MeetingContext) -> MeetingContext:
        """Update and persist a meeting context."""
        meeting_context.updated_at = datetime.now(timezone.utc)
        return await self.dao.update_meeting_context(meeting_context)

    async def clear_default_contexts(self, user_id: str, exclude_context_id: Optional[str] = None) -> None:
        """Ensure only one default meeting context per user by clearing others."""
        await self.dao.clear_default_meeting_contexts(user_id, exclude_context_id)

    async def delete_context(self, context_id: str, user_id: str) -> bool:
        """Delete a meeting context if it belongs to the user."""
        return await self.dao.delete_meeting_context(context_id, user_id)

    async def fetch_context_payload(self, context_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the meeting context and build a payload suitable for caching."""
        context: Optional[MeetingContext] = await self.get_context(context_id, user_id)
        if not context:
            return None

        tools = self.tool_service.resolve_tools(context.tools_integrations or [])

        payload: Dict[str, Any] = {
            "id": str(context.id),
            "user_id": str(context.user_id),
            "name": context.name,
            "voice_id": context.voice_id,
            "context_description": context.context_description,
            "tools": tools,
            "meeting_role": context.meeting_role.value,
            "tone_personality": context.tone_personality.value,
            "custom_tone": context.custom_tone,
            "is_default": context.is_default,
        }
        payload["system_prompt"] = self.build_system_prompt(payload)
        return payload

    def build_system_prompt(self, payload: Dict[str, Any]) -> str:
        """Construct a system prompt incorporating meeting context details."""
        lines = [
            payload.get("context_description", "").strip(),
            "",
            f"Meeting role: {payload.get('meeting_role', '').capitalize()}",
        ]

        tone = payload.get("tone_personality")
        custom_tone = payload.get("custom_tone")
        if tone:
            lines.append(f"Tone personality: {tone.capitalize()}")
        if custom_tone:
            lines.append(f"Custom tone guidance: {custom_tone}")

        tools = payload.get("tools") or []
        if tools:
            lines.append("")
            lines.append("Available tools:")
            for tool in tools:
                lines.append(f"- {tool.get('name')}: {tool.get('description')} (function: {tool.get('function')})")

        return "\n".join(filter(None, lines)).strip()

    async def cache_payload(self, meeting_id: str, payload: Dict[str, Any]) -> None:
        """Store the context payload in the in-process cache."""
        expires_at = time.monotonic() + self.CACHE_TTL_SECONDS
        serialized = json.dumps(payload)
        key = self._cache_key(meeting_id)
        async with _CACHE_LOCK:
            _CACHE_STORE[key] = (expires_at, serialized)

    async def get_cached_payload(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached context payload from the in-process cache."""
        key = self._cache_key(meeting_id)
        async with _CACHE_LOCK:
            entry = _CACHE_STORE.get(key)
            if not entry:
                return None
            expires_at, serialized = entry
            if expires_at <= time.monotonic():
                del _CACHE_STORE[key]
                return None
        return json.loads(serialized)

    async def clear_cached_payload(self, meeting_id: str) -> None:
        """Remove a cached context payload if it exists."""
        key = self._cache_key(meeting_id)
        async with _CACHE_LOCK:
            _CACHE_STORE.pop(key, None)

    @staticmethod
    def _cache_key(meeting_id: str) -> str:
        return f"meeting:context:{meeting_id}"

    async def fetch_default_context(self, user_id: str) -> Optional[MeetingContext]:
        """Return the default meeting context for a user if one is configured."""
        user_id_str = str(user_id)
        contexts = await self.get_contexts_by_user(user_id_str)

        for context in contexts:
            if context.is_default:
                return context
        return None

    async def apply_default_context_to_meeting(self, meeting: Meeting) -> Meeting:
        """Ensure the meeting has a context by applying the user's default if necessary."""
        if meeting.context_id:
            return meeting

        if not getattr(meeting, "user_id", None):
            return meeting

        default_context = await self.fetch_default_context(str(meeting.user_id))
        if default_context:
            meeting.context_id = str(default_context.id)
            if getattr(meeting, "voice_id", None) in (None, "", "default"):
                meeting.voice_id = default_context.voice_id
        return meeting


__all__ = ["ServiceMeetingContext"]
