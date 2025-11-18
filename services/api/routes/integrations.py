"""
Integrations API routes.
"""
import base64
import logging
import secrets
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from shared.schemas import UserIntegration, IntegrationStatus
from shared.config import get_settings

from ..auth import get_current_user
from ..dao import MongoDBDAO, get_dao
from services.integrations.oauth_service import get_oauth_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Available integrations configuration
AVAILABLE_INTEGRATIONS = [
    {
        "id": "google_workspace",
        "name": "Google Workspace",
        "description": "Connect Google Calendar, Gmail, Drive, and Docs to schedule meetings, send emails, and access files.",
        "category": "Productivity",
        "image_url": "/images/integrations/google.png",
    },
    {
        "id": "microsoft_365",
        "name": "Microsoft 365",
        "description": "Integrate Outlook, Teams, OneDrive, and SharePoint for calendar management and file access.",
        "category": "Productivity",
        "image_url": "/images/integrations/microsoft-office.png",
    },
    {
        "id": "slack",
        "name": "Slack",
        "description": "Send messages, create channels, and get notifications in your Slack workspace.",
        "category": "Communication",
        "image_url": "/images/integrations/slack.png",
    },
    {
        "id": "notion",
        "name": "Notion",
        "description": "Create and update pages, databases, and notes in your Notion workspace.",
        "category": "Productivity",
        "image_url": "https://www.notion.so/images/logo-ios.png",
    },
    {
        "id": "zoom",
        "name": "Zoom",
        "description": "Create and manage Zoom meetings, join calls, and access meeting recordings.",
        "category": "Video Conferencing",
        "image_url": "/images/integrations/zoom.png",
    },
    {
        "id": "hubspot",
        "name": "HubSpot",
        "description": "Sync contacts, deals, and activities with your HubSpot CRM.",
        "category": "CRM",
        "image_url": "https://www.hubspot.com/hubfs/HubSpot_Logos/HSLogo_color.svg",
    },
    {
        "id": "salesforce",
        "name": "Salesforce",
        "description": "Connect with Salesforce to manage leads, opportunities, and customer data.",
        "category": "CRM",
        "image_url": "/images/integrations/salesforce.png",
    },
    {
        "id": "github",
        "name": "GitHub",
        "description": "Create issues, pull requests, and manage repositories from your meetings.",
        "category": "Development",
        "image_url": "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
    },
    {
        "id": "jira",
        "name": "Jira",
        "description": "Create and update issues, track sprints, and manage projects in Jira.",
        "category": "Project Management",
        "image_url": "https://wac-cdn.atlassian.com/assets/img/favicons/atlassian/favicon.png",
    },
    {
        "id": "asana",
        "name": "Asana",
        "description": "Create tasks, update projects, and track work in Asana.",
        "category": "Project Management",
        "image_url": "https://asana.com/favicon.ico",
    },
    {
        "id": "trello",
        "name": "Trello",
        "description": "Create cards, update boards, and manage your Trello workspace.",
        "category": "Project Management",
        "image_url": "https://trello.com/favicon.ico",
    },
    {
        "id": "stripe",
        "name": "Stripe",
        "description": "Access payment information, invoices, and customer data from Stripe.",
        "category": "Payments",
        "image_url": "https://stripe.com/favicon.ico",
    },
    {
        "id": "zapier",
        "name": "Zapier",
        "description": "Connect with Zapier to automate workflows and integrate with 5000+ apps.",
        "category": "Automation",
        "image_url": "https://zapier.com/favicon.ico",
    },
    {
        "id": "airtable",
        "name": "Airtable",
        "description": "Create and update records in your Airtable bases and tables.",
        "category": "Database",
        "image_url": "https://www.airtable.com/favicon.ico",
    },
    {
        "id": "dropbox",
        "name": "Dropbox",
        "description": "Access and manage files in your Dropbox account.",
        "category": "Storage",
        "image_url": "https://cfl.dropboxstatic.com/static/images/favicon-vflUeLeeY.ico",
    },
    {
        "id": "box",
        "name": "Box",
        "description": "Connect with Box to access and manage your files and folders.",
        "category": "Storage",
        "image_url": "https://www.box.com/favicon.ico",
    },
    {
        "id": "intercom",
        "name": "Intercom",
        "description": "Send messages, create conversations, and manage customer interactions.",
        "category": "Customer Support",
        "image_url": "https://www.intercom.com/favicon.ico",
    },
    {
        "id": "zendesk",
        "name": "Zendesk",
        "description": "Create tickets, update support requests, and manage customer service.",
        "category": "Customer Support",
        "image_url": "https://www.zendesk.com/favicon.ico",
    },
]


class IntegrationResponse(BaseModel):
    """Integration response model."""
    
    id: str
    name: str
    description: str
    category: str
    image_url: str
    connected: bool = False


class UserIntegrationResponse(BaseModel):
    """User integration response model."""
    
    id: str
    user_id: str
    integration_id: str
    status: str
    connected_at: Optional[str] = None
    last_used_at: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class OAuthAuthorizeResponse(BaseModel):
    """OAuth authorization URL response."""
    
    oauth_url: str
    state: str


class OAuthCallbackRequest(BaseModel):
    """OAuth callback request."""
    
    code: str
    state: str


@router.get("/", response_model=List[IntegrationResponse])
async def list_integrations(
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """List all available integrations with connection status for the current user."""
    try:
        user_id = current_user["user_id"]
        
        # Get user's connected integrations
        user_integrations = await dao.get_user_integrations_by_user(user_id)
        connected_integration_ids = {
            ui.integration_id for ui in user_integrations 
            if ui.status == IntegrationStatus.CONNECTED
        }
        
        # Build response with connection status
        integrations = []
        for integration in AVAILABLE_INTEGRATIONS:
            integrations.append(IntegrationResponse(
                id=integration["id"],
                name=integration["name"],
                description=integration["description"],
                category=integration["category"],
                image_url=integration["image_url"],
                connected=integration["id"] in connected_integration_ids,
            ))
        
        return integrations

    except Exception as e:
        logger.error(f"Failed to list integrations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list integrations",
        )


class ConnectedIntegrationResponse(BaseModel):
    """Response model for connected integration with full details."""
    id: str
    user_id: str
    integration_id: str
    status: str
    connected_at: Optional[str] = None
    last_used_at: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str
    # Integration details
    name: str
    description: str
    category: str
    image_url: str


@router.get("/connected", response_model=List[ConnectedIntegrationResponse])
async def get_connected_integrations(
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """Get all connected integrations for the current user with full integration details."""
    try:
        user_id = current_user["user_id"]
        logger.info(f"Getting connected integrations for user_id: {user_id}")
        
        user_integrations = await dao.get_user_integrations_by_user(user_id)
        logger.info(f"Found {len(user_integrations)} total integrations for user {user_id}")
        
        # If no integrations found, return empty array
        if not user_integrations:
            logger.info(f"No integrations found for user {user_id}")
            return []
        
        # Filter only connected ones
        connected = [
            ui for ui in user_integrations 
            if ui.status == IntegrationStatus.CONNECTED
        ]
        logger.info(f"Found {len(connected)} connected integrations for user {user_id}")
        
        # Return empty array if no connected integrations
        if not connected:
            logger.info(f"No connected integrations found for user {user_id}")
            return []
        
        # Build response with integration details
        result = []
        for ui in connected:
            # Find integration details
            integration = next(
                (i for i in AVAILABLE_INTEGRATIONS if i["id"] == ui.integration_id),
                None
            )
            
            if integration:
                result.append(ConnectedIntegrationResponse(
                    id=str(ui.id),
                    user_id=str(ui.user_id),
                    integration_id=ui.integration_id,
                    status=ui.status.value,
                    connected_at=ui.connected_at.isoformat() if ui.connected_at else None,
                    last_used_at=ui.last_used_at.isoformat() if ui.last_used_at else None,
                    error_message=ui.error_message,
                    created_at=ui.created_at.isoformat(),
                    updated_at=ui.updated_at.isoformat(),
                    name=integration["name"],
                    description=integration["description"],
                    category=integration["category"],
                    image_url=integration["image_url"],
                ))
        
        return result

    except Exception as e:
        logger.error(f"Failed to get connected integrations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get connected integrations",
        )


@router.get("/{integration_id}", response_model=IntegrationResponse)
async def get_integration(
    integration_id: str,
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """Get details of a specific integration."""
    try:
        # Find integration
        integration = next(
            (i for i in AVAILABLE_INTEGRATIONS if i["id"] == integration_id),
            None
        )
        
        if not integration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Integration '{integration_id}' not found",
            )
        
        # Check if user has it connected
        user_id = current_user["user_id"]
        user_integration = await dao.get_user_integration(user_id, integration_id)
        connected = (
            user_integration is not None 
            and user_integration.status == IntegrationStatus.CONNECTED
        )
        
        return IntegrationResponse(
            id=integration["id"],
            name=integration["name"],
            description=integration["description"],
            category=integration["category"],
            image_url=integration["image_url"],
            connected=connected,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get integration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get integration",
        )


@router.get("/{integration_id}/oauth/authorize", response_model=OAuthAuthorizeResponse)
async def get_oauth_authorize_url(
    integration_id: str,
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """Get OAuth authorization URL for an integration."""
    try:
        # Verify integration exists
        integration = next(
            (i for i in AVAILABLE_INTEGRATIONS if i["id"] == integration_id),
            None
        )
        
        if not integration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Integration '{integration_id}' not found",
            )
        
        # Get OAuth service
        oauth_service = get_oauth_service()
        
        # Check if OAuth provider is configured
        if not oauth_service.is_configured(integration_id):
            # Provide helpful error message with required environment variables
            env_vars_map = {
                "google_workspace": "GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET",
                "microsoft_365": "MS_CLIENT_ID, MS_CLIENT_SECRET, MS_TENANT_ID (optional)",
                "slack": "SLACK_CLIENT_ID, SLACK_CLIENT_SECRET",
                "zoom": "ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET",
                "hubspot": "HUBSPOT_CLIENT_ID, HUBSPOT_CLIENT_SECRET",
                "salesforce": "SALESFORCE_CLIENT_ID, SALESFORCE_CLIENT_SECRET",
            }
            required_vars = env_vars_map.get(integration_id, "OAuth credentials")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"OAuth provider for '{integration_id}' is not configured. "
                    f"Please set the following environment variables: {required_vars}. "
                    f"These are app-level OAuth credentials that allow users to connect their accounts."
                ),
            )
        
        # Generate state parameter with user_id encoded (CSRF protection)
        # Format: base64(user_id|random_token)
        user_id = current_user["user_id"]
        random_token = secrets.token_urlsafe(32)
        state_data = f"{user_id}|{random_token}"
        state = base64.urlsafe_b64encode(state_data.encode()).decode().rstrip('=')
        
        # Get platform-specific OAuth URL
        oauth_url = oauth_service.get_authorization_url(integration_id, state)
        
        if not oauth_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate OAuth URL for '{integration_id}'",
            )
        
        logger.info(f"Generated OAuth URL for integration {integration_id} for user {current_user['user_id']}")
        
        return OAuthAuthorizeResponse(
            oauth_url=oauth_url,
            state=state,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get OAuth URL: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get OAuth authorization URL",
        )


@router.get("/{integration_id}/oauth/callback")
async def oauth_callback(
    integration_id: str,
    code: str = Query(...),
    state: str = Query(...),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
    dao: MongoDBDAO = Depends(get_dao),
):
    """Handle OAuth callback after user authorization."""
    try:
        # Check for OAuth errors
        if error:
            logger.warning(f"OAuth error for integration {integration_id}: {error} - {error_description}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"OAuth authorization failed: {error_description or error}",
            )
        
        # Decode state parameter to get user_id (CSRF protection)
        try:
            # Add padding if needed
            state_padded = state + '=' * (4 - len(state) % 4)
            state_data = base64.urlsafe_b64decode(state_padded).decode()
            user_id_str, random_token = state_data.split('|', 1)
            user_id = UUID(user_id_str)
        except (ValueError, Exception) as e:
            logger.error(f"Invalid state parameter for integration {integration_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or tampered state parameter",
            )
        
        # TODO: Exchange code for tokens (platform-specific)
        # TODO: Store tokens in database
        
        # Check if integration already exists
        existing = await dao.get_user_integration(str(user_id), integration_id)
        
        if existing:
            # Update existing integration
            existing.status = IntegrationStatus.CONNECTED
            existing.access_token = "token_placeholder"  # TODO: Store actual token
            existing.connected_at = datetime.utcnow()
            existing.updated_at = datetime.utcnow()
            await dao.update_user_integration(existing)
        else:
            # Create new integration
            user_integration = UserIntegration(
                id=uuid4(),
                user_id=user_id,
                integration_id=integration_id,
                status=IntegrationStatus.CONNECTED,
                access_token="token_placeholder",  # TODO: Store actual token
                connected_at=datetime.utcnow(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            await dao.create_user_integration(user_integration)
        
        logger.info(f"OAuth callback successful for integration {integration_id} for user {user_id}")
        
        # Return HTML page that closes popup and notifies parent window
        from fastapi.responses import HTMLResponse
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Connection Successful</title>
        </head>
        <body>
            <div style="display: flex; align-items: center; justify-content: center; height: 100vh; font-family: Arial, sans-serif;">
                <div style="text-align: center;">
                    <h2 style="color: #10b981;">✓ Connection Successful!</h2>
                    <p>You can close this window.</p>
                </div>
            </div>
            <script>
                // Notify parent window and close popup
                if (window.opener) {{
                    window.opener.postMessage({{ type: 'OAUTH_SUCCESS', integration_id: '{integration_id}' }}, '*');
                }}
                setTimeout(() => {{
                    window.close();
                }}, 2000);
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete OAuth flow",
        )


@router.post("/{integration_id}/disconnect", status_code=status.HTTP_204_NO_CONTENT)
async def disconnect_integration(
    integration_id: str,
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """Disconnect an integration."""
    try:
        user_id = current_user["user_id"]
        
        # Get user integration
        user_integration = await dao.get_user_integration(user_id, integration_id)
        
        if not user_integration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Integration '{integration_id}' is not connected",
            )
        
        # TODO: Revoke tokens (if revoke endpoint available)
        
        # Update status to disconnected
        user_integration.status = IntegrationStatus.DISCONNECTED
        user_integration.access_token = None
        user_integration.refresh_token = None
        user_integration.updated_at = datetime.utcnow()
        await dao.update_user_integration(user_integration)
        
        logger.info(f"Disconnected integration {integration_id} for user {user_id}")
        
        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disconnect integration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to disconnect integration",
        )


@router.get("/{integration_id}/status", response_model=UserIntegrationResponse)
async def get_integration_status(
    integration_id: str,
    current_user: dict = Depends(get_current_user),
    dao: MongoDBDAO = Depends(get_dao),
):
    """Get connection status for a specific integration."""
    try:
        user_id = current_user["user_id"]
        user_integration = await dao.get_user_integration(user_id, integration_id)
        
        if not user_integration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Integration '{integration_id}' is not connected",
            )
        
        return UserIntegrationResponse(
            id=str(user_integration.id),
            user_id=str(user_integration.user_id),
            integration_id=user_integration.integration_id,
            status=user_integration.status.value,
            connected_at=user_integration.connected_at.isoformat() if user_integration.connected_at else None,
            last_used_at=user_integration.last_used_at.isoformat() if user_integration.last_used_at else None,
            error_message=user_integration.error_message,
            created_at=user_integration.created_at.isoformat(),
            updated_at=user_integration.updated_at.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get integration status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get integration status",
        )

