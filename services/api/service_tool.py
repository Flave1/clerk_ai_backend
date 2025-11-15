"""
Utility service for resolving tool metadata used by meeting contexts.
"""
from typing import Dict, List


class ServiceTool:
    """Maps logical tool identifiers to concrete tool metadata."""

    def __init__(self) -> None:
        self._tool_catalog: Dict[str, Dict[str, str]] = {
            "google_calendar": {
                "id": "google_calendar",
                "name": "Google Calendar",
                "function": "create_calendar_event",
                "description": "Creates calendar events and manages attendee invitations.",
            },
            "slack": {
                "id": "slack",
                "name": "Slack Messaging",
                "function": "send_slack_message",
                "description": "Sends messages to Slack channels and direct messages.",
            },
            "email": {
                "id": "email",
                "name": "Email Sender",
                "function": "send_email",
                "description": "Composes and delivers emails to specified recipients.",
            },
            "knowledge_search": {
                "id": "knowledge_search",
                "name": "Knowledge Search",
                "function": "search_knowledge",
                "description": "Retrieves relevant documents from the internal knowledge base.",
            },
            "crm": {
                "id": "crm",
                "name": "CRM Manager",
                "function": "update_crm",
                "description": "Creates or updates CRM contacts and retrieves contact information.",
            },
        }

    def resolve_tools(self, tool_ids: List[str]) -> List[Dict[str, str]]:
        """Return metadata for the requested tool identifiers."""
        resolved: List[Dict[str, str]] = []
        for tool_id in tool_ids:
            metadata = self._tool_catalog.get(tool_id)
            if metadata:
                resolved.append(metadata)
            else:
                resolved.append(
                    {
                        "id": tool_id,
                        "name": tool_id.replace("_", " ").title(),
                        "function": "custom_tool",
                        "description": "Custom integration configured for this meeting.",
                    }
                )
        return resolved


__all__ = ["ServiceTool"]


