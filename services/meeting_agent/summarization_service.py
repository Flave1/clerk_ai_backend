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

from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
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
        self.llm: Optional[ChatOpenAI] = None
        self.parser: Optional[PydanticOutputParser] = None
        self.prompt_template: Optional[ChatPromptTemplate] = None
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the summarization service."""
        logger.info("Initializing summarization service...")
        
        try:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.3,
                api_key=settings.openai_api_key
            )
            
            # Initialize output parser
            self.parser = PydanticOutputParser(pydantic_object=MeetingAnalysis)
            
            # Create prompt template
            self.prompt_template = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an expert meeting analyst. Your task is to analyze meeting transcriptions and extract key insights, decisions, and action items.

Please analyze the provided meeting transcription and return a structured analysis including:
1. Main topics discussed
2. Key decisions made
3. Action items with assignees and due dates
4. A concise summary
5. Overall sentiment
6. Meeting duration

Be thorough but concise. Focus on actionable insights and important decisions."""),
                HumanMessage(content="""Meeting Transcription:
{transcription}

{format_instructions}

Please analyze this meeting transcription and provide a structured analysis.""")
            ])
            
            self.is_initialized = True
            logger.info("Summarization service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize summarization service: {e}")
            raise
    
    async def summarize_meeting(self, meeting: Meeting, transcription_chunks: List[TranscriptionChunk]) -> MeetingSummary:
        """Summarize a meeting from transcription chunks."""
        logger.info(f"Summarizing meeting: {meeting.title}")
        
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Combine transcription chunks
            full_transcription = self._combine_transcription_chunks(transcription_chunks)
            
            if not full_transcription.strip():
                logger.warning("No transcription content to summarize")
                return self._create_empty_summary(meeting.id)
            
            # Generate summary using LLM
            analysis = await self._analyze_transcription(full_transcription)
            
            # Convert action items to proper format
            action_items = []
            for item_data in analysis.action_items:
                action_item = ActionItem(
                    description=item_data.get('description', ''),
                    assignee=item_data.get('assignee'),
                    due_date=self._parse_due_date(item_data.get('due_date')),
                    priority=item_data.get('priority', 'medium'),
                    status='pending'
                )
                action_items.append(action_item)
            
            # Create meeting summary
            summary = MeetingSummary(
                meeting_id=meeting.id,
                topics_discussed=analysis.topics_discussed,
                key_decisions=analysis.key_decisions,
                action_items=action_items,
                summary_text=analysis.summary_text,
                sentiment=analysis.sentiment,
                duration_minutes=analysis.duration_minutes or self._calculate_duration(meeting),
                created_at=datetime.utcnow()
            )
            
            logger.info(f"Successfully summarized meeting: {meeting.title}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to summarize meeting: {e}")
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
        """Analyze transcription using LLM."""
        try:
            # Format the prompt
            formatted_prompt = self.prompt_template.format_messages(
                transcription=transcription,
                format_instructions=self.parser.get_format_instructions()
            )
            
            # Get response from LLM
            response = await self.llm.agenerate([formatted_prompt])
            
            # Parse response
            analysis = self.parser.parse(response.generations[0][0].text)
            
            return analysis
            
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            # Return default analysis if parsing fails
            return MeetingAnalysis(
                topics_discussed=["Meeting topics"],
                key_decisions=["Key decisions"],
                action_items=[],
                summary_text="Meeting summary could not be generated due to analysis error.",
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
        """Generate action items from transcription."""
        logger.info("Generating action items from transcription")
        
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Create specific prompt for action items
            action_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an expert at extracting action items from meeting transcriptions. 

Extract all action items mentioned in the transcription. For each action item, identify:
1. The description of the task
2. Who is responsible (assignee)
3. When it's due (due date)
4. Priority level (low, medium, high)

Return the action items in a structured format."""),
                HumanMessage(content=f"Meeting Transcription:\n{transcription}\n\nExtract all action items from this transcription.")
            ])
            
            # Get response from LLM
            response = await self.llm.agenerate([action_prompt.format_messages()])
            
            # Parse response (simplified for now)
            action_items = []
            response_text = response.generations[0][0].text
            
            # Simple parsing - in production, use more sophisticated parsing
            lines = response_text.split('\n')
            for line in lines:
                if line.strip() and ('action' in line.lower() or 'task' in line.lower() or 'todo' in line.lower()):
                    action_item = ActionItem(
                        description=line.strip(),
                        assignee=None,
                        due_date=None,
                        priority='medium',
                        status='pending'
                    )
                    action_items.append(action_item)
            
            return action_items
            
        except Exception as e:
            logger.error(f"Action item generation error: {e}")
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
