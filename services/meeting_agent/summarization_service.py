"""
Summarization service for meeting content analysis.

This module handles summarizing meeting transcriptions using LangChain
and OpenAI/Claude for extracting key insights, decisions, and action items.
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import sys

# Python 3.12 compatibility fix for pydantic v1 ForwardRef._evaluate()
if sys.version_info >= (3, 12):
    import pydantic.v1.typing as pydantic_v1_typing
    original_evaluate = pydantic_v1_typing.evaluate_forwardref
    def patched_evaluate_forwardref(type_, globalns, localns):
        # Python 3.12 requires recursive_guard as a keyword argument
        return type_._evaluate(globalns, localns, set(), recursive_guard=set())
    pydantic_v1_typing.evaluate_forwardref = patched_evaluate_forwardref

# LangChain imports - REMOVED, service will use OpenAI directly or fail gracefully
# from langchain_community.llms import OpenAI  # REMOVED: LangChain removed
# from langchain_community.chat_models import ChatOpenAI  # REMOVED: LangChain removed
# from langchain.schema import HumanMessage, SystemMessage  # REMOVED: LangChain removed
# from langchain.prompts import ChatPromptTemplate, PromptTemplate  # REMOVED: LangChain removed
# from langchain.chains import LLMChain  # REMOVED: LangChain removed
# from langchain.output_parsers import PydanticOutputParser  # REMOVED: LangChain removed
from pydantic import BaseModel, Field

from shared.config import get_settings
from shared.schemas import Meeting, MeetingSummary, ActionItem, TranscriptionChunk

logger = logging.getLogger(__name__)
settings = get_settings()


class MeetingAnalysis(BaseModel):
    """Pydantic model for structured meeting analysis."""
    
    topics_discussed: List[str] = Field(description="List of main topics discussed in the meeting")
    key_decisions: List[str] = Field(description="List of key decisions made during the meeting")
    action_items: List[Dict[str, Any]] = Field(description="List of action items with assignee, due date, and priority")
    summary_text: str = Field(description="Concise summary of the meeting")
    sentiment: str = Field(description="Overall sentiment of the meeting (positive, negative, neutral)")
    duration_minutes: Optional[int] = Field(description="Duration of the meeting in minutes")


class SummarizationService:
    """Service for summarizing meeting content and extracting insights."""
    
    def __init__(self):
        # LangChain removed - service will return empty summaries
        self.llm = None  # Removed: ChatOpenAI
        self.parser = None  # Removed: PydanticOutputParser
        self.prompt_template = None  # Removed: ChatPromptTemplate
        self.is_initialized = False
        logger.warning("⚠️ SummarizationService: LangChain removed - summarization will return empty summaries")
        
    async def initialize(self) -> None:
        """Initialize the summarization service."""
        logger.warning("⚠️ SummarizationService.initialize() called but LangChain is removed - service will not work")
        self.is_initialized = False
        # Don't raise - allow service to exist but be unusable
    
    async def summarize_meeting(self, meeting: Meeting, transcription_chunks: List[TranscriptionChunk]) -> MeetingSummary:
        """Summarize a meeting from transcription chunks."""
        logger.warning(f"⚠️ Summarize meeting called for {meeting.title} but LangChain is removed - returning empty summary")
        
        # LangChain removed - return empty summary
        return self._create_empty_summary(meeting.id)
    
    def _combine_transcription_chunks(self, chunks: List[TranscriptionChunk]) -> str:
        """Combine transcription chunks into a single text."""
        if not chunks:
            return ""
        
        # Sort chunks by timestamp
        sorted_chunks = sorted(chunks, key=lambda x: x.timestamp)
        
        # Combine text
        combined_text = " ".join(chunk.text for chunk in sorted_chunks)
        
        return combined_text
    
    async def _analyze_transcription(self, transcription: str) -> MeetingAnalysis:
        """Analyze transcription using LLM - DISABLED: LangChain removed."""
        logger.warning("⚠️ _analyze_transcription called but LangChain is removed")
        # Return default analysis since LangChain is removed
        return MeetingAnalysis(
            topics_discussed=["Meeting topics"],
            key_decisions=["Key decisions"],
            action_items=[],
            summary_text="Meeting summary could not be generated - LangChain summarization service removed.",
            sentiment="neutral",
            duration_minutes=None
        )
    
    def _parse_due_date(self, due_date_str: Optional[str]) -> Optional[datetime]:
        """Parse due date string to datetime."""
        if not due_date_str:
            return None
        
        try:
            # Try to parse common date formats
            from dateutil import parser
            return parser.parse(due_date_str)
        except Exception:
            logger.warning(f"Could not parse due date: {due_date_str}")
            return None
    
    def _calculate_duration(self, meeting: Meeting) -> Optional[int]:
        """Calculate meeting duration in minutes."""
        if meeting.start_time and meeting.end_time:
            duration = meeting.end_time - meeting.start_time
            return int(duration.total_seconds() / 60)
        return None
    
    def _create_empty_summary(self, meeting_id: str) -> MeetingSummary:
        """Create an empty summary when analysis fails."""
        return MeetingSummary(
            meeting_id=meeting_id,
            topics_discussed=[],
            key_decisions=[],
            action_items=[],
            summary_text="No summary available.",
            sentiment="neutral",
            duration_minutes=None,
            created_at=datetime.utcnow()
        )
    
    async def generate_action_items(self, transcription: str) -> List[ActionItem]:
        """Generate action items from transcription - DISABLED: LangChain removed."""
        logger.warning("⚠️ generate_action_items called but LangChain is removed - returning empty list")
        # LangChain removed - return empty list
        return []
    
    async def generate_meeting_notes(self, meeting: Meeting, transcription_chunks: List[TranscriptionChunk]) -> str:
        """Generate formatted meeting notes."""
        logger.info(f"Generating meeting notes for: {meeting.title}")
        
        try:
            # Combine transcription
            full_transcription = self._combine_transcription_chunks(transcription_chunks)
            
            if not full_transcription.strip():
                return "No meeting content available."
            
            # Generate summary
            summary = await self.summarize_meeting(meeting, transcription_chunks)
            
            # Format meeting notes
            notes = f"""
# Meeting Notes: {meeting.title}

**Date:** {meeting.start_time.strftime('%Y-%m-%d %H:%M')}
**Duration:** {summary.duration_minutes} minutes
**Platform:** {meeting.platform.value}
**Participants:** {len(meeting.participants)}

## Summary
{summary.summary_text}

## Topics Discussed
{chr(10).join(f"- {topic}" for topic in summary.topics_discussed)}

## Key Decisions
{chr(10).join(f"- {decision}" for decision in summary.key_decisions)}

## Action Items
{chr(10).join(f"- {item.description} (Assignee: {item.assignee or 'TBD'}, Due: {item.due_date or 'TBD'}, Priority: {item.priority})" for item in summary.action_items)}

## Sentiment
Overall meeting sentiment: {summary.sentiment}

---
*Generated by AI Meeting Assistant*
"""
            
            return notes.strip()
            
        except Exception as e:
            logger.error(f"Meeting notes generation error: {e}")
            return f"Error generating meeting notes: {str(e)}"
    
    async def cleanup(self) -> None:
        """Cleanup summarization service resources."""
        logger.info("Cleaning up summarization service...")
        
        try:
            self.is_initialized = False
            logger.info("Summarization service cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Factory function for creating summarization service
def create_summarization_service() -> SummarizationService:
    """Create a new summarization service instance."""
    return SummarizationService()
