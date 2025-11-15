"""
LLM service using LangGraph for stateful AI workflows with tool calling.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, TypedDict, Annotated, AsyncIterator
from uuid import UUID

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from shared.config import get_settings
from shared.schemas import LLMRequest, LLMResponse, Turn, TurnType
from services.api.service_meeting_context import ServiceMeetingContext
from services.api.service_meeting import ServiceMeeting
from services.api.dao import DynamoDBDAO

logger = logging.getLogger(__name__)
settings = get_settings()


class ConversationState(TypedDict, total=False):
    """State for the conversation graph."""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    turn: Turn
    context: List[Dict[str, str]]
    response: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]
    conversation_id: str
    system_prompt: Optional[str]


class LLMService:
    """LangGraph-based LLM service with tool calling capabilities."""

    def __init__(self, dao: Optional[DynamoDBDAO] = None):
        self.llm = None
        self.tools = []
        self.graph = None
        # Use injected DAO or get from global instance
        if dao is not None:
            self._dao = dao
        else:
            # Try to get from global instance (initialized in unified_app)
            try:
                from services.api.dao import get_dao
                self._dao = get_dao()
            except RuntimeError:
                # DAO not initialized yet, will be set later
                self._dao = None
        self._dao_lock = asyncio.Lock()
        try:
            self._initialize_llm()
            self._initialize_tools()
            self._build_graph()
        except Exception as e:
            logger.error(f"Error during LLM service initialization: {e}", exc_info=True)
            # Don't raise - service object should still exist
            # generate_response_streaming can work without self.llm if OpenAI API key is available

    def _initialize_llm(self):
        """Initialize the LLM client."""
        try:
            if not settings.openai_api_key:
                logger.warning("⚠️ OPENAI_API_KEY not found. LLM service will not be available.")
                logger.warning("   Set OPENAI_API_KEY environment variable or in settings to enable LLM service.")
                self.llm = None
                # Don't return - service object should still exist
                # Methods will check if llm is None before using it
                return
            
            masked_key = f"{settings.openai_api_key[:4]}...{settings.openai_api_key[-4:]}" if len(settings.openai_api_key) > 8 else "<hidden>"
            logger.warning("🔐 Detected OpenAI API key in configuration", extra={
                "key_preview": masked_key,
                "key_length": len(settings.openai_api_key)
            })

            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                api_key=settings.openai_api_key,
            )
            logger.info("✅ OpenAI LLM initialized for LangGraph")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
            self.llm = None
            # Don't raise - allow service to exist but be unusable

    def _initialize_tools(self):
        """Initialize tools for the AI assistant."""
        try:
            # Import tool classes
            from services.workers.tools.calendar_google import GoogleCalendarTool
            from services.workers.tools.slack import SlackTool
            from services.workers.tools.email import EmailTool
            from services.workers.tools.rag import RAGTool
            from services.workers.tools.crm_mock import MockCRMTool
            
            # Initialize tool instances (may fail if dependencies missing)
            try:
                self.calendar_tool = GoogleCalendarTool()
            except Exception as e:
                logger.warning(f"Failed to initialize calendar tool: {e}")
                self.calendar_tool = None
            
            try:
                self.slack_tool = SlackTool()
            except Exception as e:
                logger.warning(f"Failed to initialize Slack tool: {e}")
                self.slack_tool = None
            
            try:
                self.email_tool = EmailTool()
            except Exception as e:
                logger.warning(f"Failed to initialize email tool: {e}")
                self.email_tool = None
            
            try:
                self.rag_tool = RAGTool()
            except Exception as e:
                logger.warning(f"Failed to initialize RAG tool: {e}")
                self.rag_tool = None
            
            try:
                self.crm_tool = MockCRMTool()
            except Exception as e:
                logger.warning(f"Failed to initialize CRM tool: {e}")
                self.crm_tool = None
        except Exception as e:
            logger.warning(f"Failed to import or initialize tools: {e}")
            # Set all tools to None so decorated functions can check
            self.calendar_tool = None
            self.slack_tool = None
            self.email_tool = None
            self.rag_tool = None
            self.crm_tool = None
        
        @tool
        def create_calendar_event(
            title: str,
            start_time: str,
            end_time: str,
            attendees: Optional[List[str]] = None,
            description: Optional[str] = None,
            meeting_location: Optional[str] = None
        ) -> str:
            """
            Create a calendar event or meeting.
            
            Args:
                title (str): The title/subject of the calendar event
                start_time (str): Start time in ISO 8601 format (e.g., "2025-10-08T14:00:00Z")
                end_time (str): End time in ISO 8601 format (e.g., "2025-10-08T15:00:00Z")
                attendees (Optional[List[str]]): List of attendee email addresses
                description (Optional[str]): Detailed description of the event
                meeting_location (Optional[str]): Meeting location (physical address or Zoom/Google Meet link)
            
            Returns:
                str: Success or error message about the calendar event creation
            """
            import asyncio
            
            parameters = {
                "title": title,
                "start_time": start_time,
                "end_time": end_time,
                "attendees": attendees or [],
                "description": description or "",
                "location": meeting_location or ""
            }
            
            try:
                if self.calendar_tool is None:
                    return "❌ Calendar tool not available"
                result = asyncio.run(self.calendar_tool.create_event(parameters))
                if result["success"]:
                    event_id = result["result"]["event_id"]
                    event_link = result["result"].get("event_link", "")
                    
                    response = f"✅ Calendar event '{title}' created successfully!"
                    if event_link:
                        response += f" Link: {event_link}"
                    if meeting_location:
                        response += f" Location: {meeting_location}"
                    return response
                else:
                    return f"❌ Failed to create calendar event: {result['error']}"
            except Exception as e:
                logger.error(f"Calendar tool error: {e}")
                return f"❌ Error creating calendar event: {str(e)}"

        @tool
        def send_slack_message(channel: str, message: str) -> str:
            """
            Send a message to a Slack channel.
            
            Args:
                channel (str): The Slack channel name or ID (e.g., "#general", "#team-updates")
                message (str): The message content to send
            
            Returns:
                str: Success or error message about the Slack message delivery
            """
            import asyncio
            
            parameters = {
                "channel": channel,
                "message": message
            }
            
            try:
                if self.slack_tool is None:
                    return "❌ Slack tool not available"
                result = asyncio.run(self.slack_tool.send_message(parameters))
                if result["success"]:
                    message_ts = result["result"]["message_ts"]
                    return f"✅ Slack message sent to {channel}!"
                else:
                    return f"❌ Failed to send Slack message: {result['error']}"
            except Exception as e:
                logger.error(f"Slack tool error: {e}")
                return f"❌ Error sending Slack message: {str(e)}"

        @tool
        def send_email(to: str, subject: str, body: str) -> str:
            """
            Send an email to a recipient.
            
            Args:
                to (str): Recipient's email address
                subject (str): Email subject line
                body (str): Email body/message content
            
            Returns:
                str: Success or error message about the email delivery
            """
            import asyncio
            
            parameters = {
                "to_email": to,
                "subject": subject,
                "message": body
            }
            
            try:
                if self.email_tool is None:
                    return "❌ Email tool not available"
                result = asyncio.run(self.email_tool.send_email(parameters))
                if result["success"]:
                    return f"✅ Email sent to {to} with subject '{subject}'!"
                else:
                    return f"❌ Failed to send email: {result['error']}"
            except Exception as e:
                logger.error(f"Email tool error: {e}")
                return f"❌ Error sending email: {str(e)}"

        @tool
        def search_knowledge(query: str) -> str:
            """
            Search the internal knowledge base for information.
            
            Args:
                query (str): Search query to find relevant information in the knowledge base
            
            Returns:
                str: Formatted search results with relevant documents or error message
            """
            import asyncio
            
            parameters = {
                "query": query,
                "limit": 3
            }
            
            try:
                if self.rag_tool is None:
                    return "❌ Knowledge base search tool not available"
                result = asyncio.run(self.rag_tool.search(parameters))
                if result["success"]:
                    documents = result["result"]["documents"]
                    count = result["result"]["count"]
                    
                    if count == 0:
                        return f"🔍 No information found for '{query}'"
                    
                    response = f"🔍 Found {count} result(s) for '{query}':\n\n"
                    for i, doc in enumerate(documents, 1):
                        response += f"{i}. **{doc['title']}**\n"
                        response += f"   {doc['content'][:200]}{'...' if len(doc['content']) > 200 else ''}\n\n"
                    
                    return response
                else:
                    return f"❌ Failed to search knowledge base: {result['error']}"
            except Exception as e:
                logger.error(f"RAG tool error: {e}")
                return f"❌ Error searching knowledge base: {str(e)}"

        @tool
        def update_crm(contact_email: str, action: str, data: Dict[str, Any]) -> str:
            """
            Manage CRM contacts - create, update, or retrieve contact information.
            
            Args:
                contact_email (str): Contact's email address (primary identifier)
                action (str): Action to perform - "create", "update", or "get"
                data (Dict[str, Any]): Contact data dictionary with fields:
                    - name (str): Contact's full name
                    - phone (str): Contact's phone number
                    - company (str): Contact's company name
                    - Additional custom fields as needed
            
            Returns:
                str: Success or error message about the CRM operation
            """
            import asyncio
            
            try:
                if self.crm_tool is None:
                    return "❌ CRM tool not available"
                if action == "create":
                    parameters = {
                        "email": contact_email,
                        "name": data.get("name", ""),
                        "phone": data.get("phone", ""),
                        "company": data.get("company", "")
                    }
                    result = asyncio.run(self.crm_tool.create_contact(parameters))
                elif action == "update":
                    parameters = {"email": contact_email, **data}
                    result = asyncio.run(self.crm_tool.update_contact(parameters))
                elif action == "get":
                    parameters = {"email": contact_email}
                    result = asyncio.run(self.crm_tool.get_contact(parameters))
                else:
                    return f"❌ Unknown CRM action: {action}. Supported actions: create, update, get"
                
                if result["success"]:
                    if action == "create":
                        contact_id = result["result"]["contact_id"]
                        return f"✅ Created CRM contact for {contact_email} (ID: {contact_id})"
                    elif action == "update":
                        return f"✅ Updated CRM contact for {contact_email}"
                    elif action == "get":
                        contact = result["result"]
                        return f"✅ CRM contact for {contact_email}: {contact.get('name', 'N/A')} at {contact.get('company', 'N/A')}"
                else:
                    return f"❌ Failed to {action} CRM contact: {result['error']}"
            except Exception as e:
                logger.error(f"CRM tool error: {e}")
                return f"❌ Error with CRM {action}: {str(e)}"
        
        def save_meeting_to_db(meeting_obj, meeting_id):
                async def _persist():
                    # Use the global DAO instance (initialized in unified_app)
                    from services.api.dao import get_dao
                    dao = get_dao()
                    context_service = ServiceMeetingContext(dao)
                    meeting_service = ServiceMeeting(dao)
                    await context_service.apply_default_context_to_meeting(meeting_obj)
                    await meeting_service.save_meeting(meeting_obj)

                try:
                    asyncio.run(_persist())
                    logger.info(f"Saved Zoom meeting {meeting_id} to database")
                except Exception as db_err:
                    logger.error(f"Failed to save Zoom meeting to DB: {db_err}")
                    # Continue anyway - meeting was created on Zoom
        
        @tool
        def create_zoom_meeting(
            title: str,
            start_time: str,
            end_time: str,
            attendees: Optional[List[str]] = None,
            description: Optional[str] = None
        ) -> str:
            """
            Create a Zoom meeting with specified attendees and save to database.
            
            Args:
                title (str): The title/subject of the Zoom meeting
                start_time (str): Start time in ISO 8601 format (e.g., "2025-10-08T14:00:00Z")
                end_time (str): End time in ISO 8601 format (e.g., "2025-10-08T15:00:00Z")
                attendees (Optional[List[str]]): List of attendee email addresses
                description (Optional[str]): Detailed description of the meeting
            
            Returns:
                str: Success message with meeting details or error message
            """
            import asyncio
            from services.meeting_agent.zoom_client import create_zoom_client
            
            try:
                # Create Zoom client and meeting
                zoom_client = create_zoom_client()
                
                # Check if we're already in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an event loop, use create_task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        result = executor.submit(
                            lambda: asyncio.run(zoom_client.create_meeting(
                                title=title,
                                start_time=start_time,
                                end_time=end_time,
                                attendees=attendees,
                                description=description
                            ))
                        ).result()
                except RuntimeError:
                    # No event loop, use asyncio.run
                    result = asyncio.run(zoom_client.create_meeting(
                        title=title,
                        start_time=start_time,
                        end_time=end_time,
                        attendees=attendees,
                        description=description
                    ))
                
                if result["success"]:
                    meeting = result["meeting"]

                    if "_meeting_obj" in result:
                        save_meeting_to_db(result["_meeting_obj"], meeting['id'])
                    
                    response = f"✅ Zoom meeting '{title}' created and saved to database!\n"
                    response += f"📅 Meeting ID: {meeting['id']}\n"
                    response += f"🔗 Join URL: {meeting['join_url']}\n"
                    response += f"⏰ Start Time: {start_time}\n"
                    response += f"⏱️ Duration: {meeting['duration_minutes']} minutes"
                    
                    if attendees:
                        response += f"\n👥 Attendees: {', '.join(attendees)}"
                    
                    response += f"\n💾 Status: Saved to database"
                    return response
                else:
                    error_msg = f"❌ Failed to create Zoom meeting: {result.get('error', 'Unknown error')}\n"
                    error_msg += f"Please try again or contact support if the issue persists."
                    return error_msg
                    
            except Exception as e:
                logger.error(f"Zoom meeting creation error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return f"❌ Unable to create Zoom meeting due to an error: {str(e)}. The meeting was NOT scheduled."

        @tool
        def create_teams_meeting(
            title: str,
            start_time: str,
            end_time: str,
            attendees: Optional[List[str]] = None,
            description: Optional[str] = None
        ) -> str:
            """
            Create a Microsoft Teams meeting with specified attendees and save to database.
            
            Args:
                title (str): The title/subject of the Microsoft Teams meeting
                start_time (str): Start time in ISO 8601 format (e.g., "2025-10-08T14:00:00Z")
                end_time (str): End time in ISO 8601 format (e.g., "2025-10-08T15:00:00Z")
                attendees (Optional[List[str]]): List of attendee email addresses
                description (Optional[str]): Detailed description of the meeting
            
            Returns:
                str: Success message with meeting details or error message
            """
            import asyncio
            from services.meeting_agent.teams_client import create_teams_client
            
            try:
                # Create Teams client and meeting
                teams_client = create_teams_client()
                
                # Check if we're already in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an event loop, use create_task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        result = executor.submit(
                            lambda: asyncio.run(teams_client.create_meeting(
                                title=title,
                                start_time=start_time,
                                end_time=end_time,
                                attendees=attendees,
                                description=description
                            ))
                        ).result()
                except RuntimeError:
                    # No event loop, use asyncio.run
                    result = asyncio.run(teams_client.create_meeting(
                        title=title,
                        start_time=start_time,
                        end_time=end_time,
                        attendees=attendees,
                        description=description
                    ))
                
                if result["success"]:
                    meeting = result["meeting"]
                    
                    if "_meeting_obj" in result:
                        save_meeting_to_db(result["_meeting_obj"], meeting['id'])
                    
                    response = f"✅ Microsoft Teams meeting '{title}' created and saved to database!\n"
                    response += f"📅 Meeting ID: {meeting['id']}\n"
                    response += f"🔗 Join URL: {meeting['join_url']}\n"
                    response += f"⏰ Start Time: {start_time}\n"
                    response += f"⏱️ Duration: {meeting['duration_minutes']} minutes"
                    
                    if attendees:
                        response += f"\n👥 Attendees: {', '.join(attendees)}"
                    
                    response += f"\n💾 Status: Saved to database"
                    return response
                else:
                    error_msg = f"❌ Failed to create Microsoft Teams meeting: {result.get('error', 'Unknown error')}\n"
                    error_msg += f"Please try again or contact support if the issue persists."
                    return error_msg
                    
            except Exception as e:
                logger.error(f"Microsoft Teams meeting creation error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return f"❌ Unable to create Microsoft Teams meeting due to an error: {str(e)}. The meeting was NOT scheduled."

        self.tools = [
            create_calendar_event,
            send_slack_message,
            send_email,
            search_knowledge,
            update_crm,
            create_zoom_meeting,
            create_teams_meeting,
        ]
        logger.info(f"Initialized {len(self.tools)} tools for LangGraph with real implementations")

    def _build_graph(self):
        """Build the LangGraph workflow."""
        # Only build graph if LLM is initialized
        if self.llm is None:
            logger.warning("LLM not initialized, skipping graph construction")
            return
        
        try:
            def should_continue(state: ConversationState) -> str:
                """Determine if we should continue with tool calls or end."""
                messages = state["messages"]
                last_message = messages[-1]
                
                # If the last message has tool calls, continue to tools
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    return "tools"
                return "end"

            def call_model(state: ConversationState) -> ConversationState:
                """Call the LLM model."""
                try:
                    system_prompt = state.get("system_prompt") or self._get_system_prompt()
                    # Prepare messages with system prompt
                    messages = [
                        SystemMessage(content=system_prompt),
                        *state["messages"]
                    ]
                    
                    # Get response from LLM
                    response = self.llm.bind_tools(self.tools).invoke(messages)
                    
                    # Update state with response
                    state["messages"].append(response)
                    state["response"] = response.content
                    
                    # Extract tool calls if any
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        import json
                        state["tool_calls"] = [
                            {
                                "id": tool_call["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_call["name"],
                                    "arguments": json.dumps(tool_call["args"]),
                                },
                            }
                            for tool_call in response.tool_calls
                        ]
                    
                    logger.info(f"LLM response generated: {len(response.content)} chars")
                    return state
                    
                except Exception as e:
                    logger.error(f"Failed to call model: {e}")
                    state["response"] = "I'm sorry, I encountered an error. Please try again."
                    return state

            def call_tools(state: ConversationState) -> ConversationState:
                """Execute tool calls."""
                try:
                    # Get the last message which should have tool calls
                    last_message = state["messages"][-1]
                    
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        # Execute each tool call
                        for tool_call in last_message.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            
                            # Find the tool function
                            tool_func = None
                            for tool in self.tools:
                                if tool.name == tool_name:
                                    tool_func = tool
                                    break
                            
                            if tool_func:
                                try:
                                    # Execute the tool
                                    result = tool_func.invoke(tool_args)
                                    
                                    # Add tool result message
                                    from langchain_core.messages import ToolMessage
                                    tool_message = ToolMessage(
                                        content=str(result),
                                        tool_call_id=tool_call["id"]
                                    )
                                    state["messages"].append(tool_message)
                                    
                                    logger.info(f"Executed tool {tool_name}: {result}")
                                except Exception as tool_error:
                                    logger.error(f"Tool execution error for {tool_name}: {tool_error}")
                                    # Add error message
                                    from langchain_core.messages import ToolMessage
                                    error_message = ToolMessage(
                                        content=f"Error executing {tool_name}: {str(tool_error)}",
                                        tool_call_id=tool_call["id"]
                                    )
                                    state["messages"].append(error_message)
                    
                    logger.info(f"Executed {len(last_message.tool_calls) if hasattr(last_message, 'tool_calls') and last_message.tool_calls else 0} tool calls")
                    return state

                except Exception as e:
                    logger.error(f"Failed to execute tools: {e}")
                    return state

            # Build the graph
            workflow = StateGraph(ConversationState)
            
            # Add nodes
            workflow.add_node("agent", call_model)
            workflow.add_node("tools", call_tools)
            
            # Add edges
            workflow.set_entry_point("agent")
            workflow.add_conditional_edges(
                "agent",
                should_continue,
                {
                    "tools": "tools",
                    "end": END,
                },
            )
            workflow.add_edge("tools", "agent")
            
            # Compile the graph
            self.graph = workflow.compile()
            logger.info("LangGraph workflow compiled successfully")
        except Exception as e:
            logger.error(f"Failed to build LangGraph workflow: {e}", exc_info=True)
            # Don't raise - service can still work for basic chat without graph

    async def _get_dao(self) -> DynamoDBDAO:
        """Get the DynamoDB DAO instance (must be initialized in unified_app)."""
        if self._dao is None:
            # Try to get from global instance
            try:
                from services.api.dao import get_dao
                self._dao = get_dao()
            except RuntimeError:
                raise RuntimeError(
                    "DAO instance not initialized. Ensure unified_app.py initializes DAO before using LLMService."
                )
        return self._dao

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate LLM response using LangGraph."""
        try:
            # Convert messages to LangChain format
            messages = []
            for msg in request.messages:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
                elif msg["role"] == "system":
                    messages.append(SystemMessage(content=msg["content"]))

            # Create initial state
            initial_state = ConversationState(
                messages=messages,
                turn=None,  # Will be set in process_turn
                context=request.messages,
                response=None,
                tool_calls=None,
                conversation_id=str(request.conversation_id),
                system_prompt=request.system_prompt,
            )

            # Run the graph
            result = await asyncio.to_thread(self.graph.invoke, initial_state)

            # Extract response
            response_content = result.get("response", "")
            tool_calls = result.get("tool_calls", None)

            return LLMResponse(
                content=response_content,
                tool_calls=tool_calls,
                usage=None,  # LangGraph doesn't provide usage stats directly
                finish_reason="stop",
            )

        except Exception as e:
            logger.error(f"LangGraph generation failed: {e}")
            raise

    
  

    async def process_turn(
        self,
        turn: Turn,
        context: List[Dict[str, str]],
        *,
        meeting_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Optional[LLMResponse]:
        """Process a conversation turn using LangGraph."""
        try:
            # Build messages from context
            messages = context + [{"role": "user", "content": turn.content}]

            if meeting_id:
                system_prompt = await self._get_meeting_system_prompt(meeting_id)

            request = LLMRequest(
                conversation_id=turn.conversation_id,
                messages=messages,
                tools=None,  # Tools are handled by LangGraph
                system_prompt=system_prompt,
            )

            response = await self.generate_response(request)

            # Log the interaction
            logger.info(
                f"Processed turn {turn.id}: {len(turn.content)} chars -> {len(response.content)} chars"
            )

            return response

        except Exception as e:
            logger.error(f"Failed to process turn {turn.id}: {e}")
            return None

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI assistant."""
        from datetime import datetime

        current_date = datetime.now().strftime("%A, %B %d, %Y")
        current_time = datetime.now().strftime("%I:%M %p")

        return f"""You are Aurray, an AI assistant speaking directly to users through voice. 
        Today is {current_date}, and the current time is {current_time}.

        IMPORTANT: You are speaking to users through voice, not writing text. Respond naturally as if you're having a conversation.

        Your tasks:
        1. Greet users warmly and professionally through voice.
        2. Understand their intent and respond efficiently in spoken conversation.
        3. Schedule or join meetings upon request.
        4. Communicate via Slack or email when needed.
        5. Search or update records as requested.

        Always use the available tools instead of describing actions.

        Voice Communication Guidelines:
        - Speak naturally and conversationally, as if you're talking to someone in person.
        - Use contractions (I'll, you're, we'll) to sound more natural.
        - Keep responses concise and easy to follow when spoken aloud.
        - Use verbal confirmations like "Got it" or "I understand" instead of written acknowledgments.
        - Ask clarifying questions conversationally when unsure.
        - Use {current_date} as reference for scheduling or time-based actions.

        Available tools:
        - create_calendar_event(title, start_time, end_time, attendees, description, meeting_location): Schedule meetings.
        - create_zoom_meeting(title, start_time, end_time, attendees, description): Create Zoom meetings and save to database.
          * IMPORTANT: Always include attendees parameter as a list of email addresses
          * Times must be in ISO 8601 format with timezone (e.g., "2025-10-08T14:00:00Z" or "2025-10-08T14:00:00")
          * Example: create_zoom_meeting("Team Meeting", "2025-10-08T14:00:00Z", "2025-10-08T15:00:00Z", ["user@example.com"], "Project update")
        - create_teams_meeting(title, start_time, end_time, attendees, description): Create Microsoft Teams meetings and save to database.
          * IMPORTANT: Always include attendees parameter as a list of email addresses
          * Times must be in ISO 8601 format with timezone (e.g., "2025-10-08T14:00:00Z" or "2025-10-08T14:00:00")
          * Example: create_teams_meeting("Product Review", "2025-10-08T14:00:00Z", "2025-10-08T15:00:00Z", ["user@example.com"], "Quarterly review")
        - send_slack_message(channel, message): Send Slack messages.
        - send_email(to, subject, body): Send emails.
        - search_knowledge(query): Search internal knowledge base.
        - update_crm(contact_email, action, data): Manage CRM contacts.

        Always perform the appropriate tool action instead of explaining what you'd do.
        Remember: You're speaking through voice, so respond naturally and conversationally.
        """

    async def _get_meeting_system_prompt(self, meeting_id: str) -> str:
        """Retrieve a meeting-specific system prompt from cache or storage."""
        try:
            dao = await self._get_dao()
            context_service = ServiceMeetingContext(dao)
            meeting_service = ServiceMeeting(dao)
            payload = await context_service.get_cached_payload(meeting_id)

            if not payload:
                meeting = await meeting_service.get_meeting(meeting_id)
                if not meeting or not meeting.context_id:
                    return self._get_system_prompt()
                payload = await context_service.fetch_context_payload(
                    str(meeting.context_id),
                    str(meeting.user_id),
                )
                if payload:
                    await context_service.cache_payload(meeting_id, payload)

            if payload and payload.get("context_description"):
                return payload["context_description"]
        except Exception as error:
            logger.warning(
                "Falling back to default system prompt; failed to load meeting context",
                extra={"meeting_id": meeting_id, "error": str(error)},
            )
        return self._get_system_prompt()

    async def _stream_response_with_prompt(
        self,
        system_prompt: str,
        text: Optional[str] = None,
        audio_data: Optional[bytes] = None,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        try:
            import io
            import wave

            if audio_data:
                if not settings.openai_api_key:
                    raise RuntimeError("OpenAI API key required for audio streaming")

                from openai import OpenAI

                client = OpenAI(api_key=settings.openai_api_key)

                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(audio_data)

                wav_buffer.seek(0)

                logger.debug("🎤 Transcribing audio with Whisper...")
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("audio.wav", wav_buffer.read(), "audio/wav"),
                    language="en",
                )

                transcribed_text = transcription.text.strip()
                if not transcribed_text:
                    logger.warning("⚠️ Whisper returned empty transcription")
                    return

                logger.info(f"📝 Whisper transcription: '{transcribed_text}'")

                messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

                if conversation_context:
                    for msg in conversation_context:
                        if msg["role"] == "user":
                            content = msg.get("content")
                            if isinstance(content, dict) and "audio" in content:
                                continue
                            if isinstance(content, str):
                                messages.append({"role": "user", "content": content})
                        elif msg["role"] == "assistant":
                            messages.append({"role": "assistant", "content": msg["content"]})

                messages.append({"role": "user", "content": transcribed_text})

                logger.debug("🤖 Sending transcription to GPT-4o...")
                stream = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    stream=True,
                )

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            else:
                if model is None:
                    if settings.openai_api_key:
                        fast_llm = ChatOpenAI(
                            model="gpt-4o-mini",
                            temperature=0.7,
                            api_key=settings.openai_api_key,
                            streaming=True,
                        )
                    else:
                        raise RuntimeError("OpenAI API key is required for streaming")
                else:
                    fast_llm = self.llm

                messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

                if conversation_context:
                    for msg in conversation_context:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            messages.append(AIMessage(content=msg["content"]))

                if text:
                    messages.append(HumanMessage(content=text))

                async for chunk in fast_llm.astream(messages):
                    if hasattr(chunk, "content") and chunk.content:
                        yield chunk.content

        except Exception as e:
            logger.error(f"Streaming LLM generation failed: {e}", exc_info=True)
            yield f"I'm sorry, I encountered an error: {str(e)}"

    async def generate_response_streaming(
        self,
        text: Optional[str] = None,
        audio_data: Optional[bytes] = None,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Streaming response using the default system prompt."""
        async for chunk in self._stream_response_with_prompt(
            self._get_system_prompt(),
            text,
            audio_data,
            conversation_context,
            model,
        ):
            yield chunk

    async def generate_meeting_response_streaming_with_context(
        self,
        meeting_id: str,
        text: Optional[str] = None,
        audio_data: Optional[bytes] = None,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Streaming response that leverages meeting-specific context."""
        system_prompt = await self._get_meeting_system_prompt(meeting_id)
        async for chunk in self._stream_response_with_prompt(
            system_prompt,
            text,
            audio_data,
            conversation_context,
            model,
        ):
            yield chunk


