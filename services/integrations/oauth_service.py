"""
OAuth service for managing platform-specific OAuth providers.
"""
import logging
from typing import List, Optional

from shared.config import get_settings

from .base import OAuthProvider
from .google import GoogleOAuthProvider
from .microsoft import MicrosoftOAuthProvider
from .slack import SlackOAuthProvider
from .zoom import ZoomOAuthProvider
from .hubspot import HubSpotOAuthProvider
from .salesforce import SalesforceOAuthProvider

logger = logging.getLogger(__name__)

settings = get_settings()


class OAuthService:
    """Service for managing OAuth providers."""
    
    def __init__(self):
        """Initialize OAuth service."""
        self._providers: dict[str, OAuthProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize OAuth providers from settings."""
        # Get API base URL for redirect URIs
        api_base_url = settings.api_base_url
        api_prefix = settings.api_prefix
        
        # Google Workspace
        if settings.google_client_id and settings.google_client_secret:
            google_redirect_uri = f"{api_base_url}{api_prefix}/integrations/google_workspace/oauth/callback"
            self._providers["google_workspace"] = GoogleOAuthProvider(
                client_id=settings.google_client_id,
                client_secret=settings.google_client_secret,
                redirect_uri=google_redirect_uri,
            )
            logger.info("Google OAuth provider initialized")
        else:
            logger.warning("Google OAuth credentials not configured")
        
        # Microsoft 365
        if settings.ms_client_id and settings.ms_client_secret:
            ms_redirect_uri = f"{api_base_url}{api_prefix}/integrations/microsoft_365/oauth/callback"
            ms_tenant_id = settings.ms_tenant_id or "common"
            self._providers["microsoft_365"] = MicrosoftOAuthProvider(
                client_id=settings.ms_client_id,
                client_secret=settings.ms_client_secret,
                redirect_uri=ms_redirect_uri,
                tenant_id=ms_tenant_id,
            )
            logger.info(f"Microsoft OAuth provider initialized with redirect_uri: {ms_redirect_uri}, tenant_id: {ms_tenant_id}")
        else:
            logger.warning("Microsoft OAuth credentials not configured")
        
        # Slack
        if settings.slack_client_id and settings.slack_client_secret:
            slack_redirect_uri = f"{api_base_url}{api_prefix}/integrations/slack/oauth/callback"
            self._providers["slack"] = SlackOAuthProvider(
                client_id=settings.slack_client_id,
                client_secret=settings.slack_client_secret,
                redirect_uri=slack_redirect_uri,
            )
            logger.info(f"Slack OAuth provider initialized with redirect_uri: {slack_redirect_uri}")
        else:
            logger.warning("Slack OAuth credentials not configured")
        
        # Zoom
        if settings.zoom_client_id and settings.zoom_client_secret:
            zoom_redirect_uri = f"{api_base_url}{api_prefix}/integrations/zoom/oauth/callback"
            self._providers["zoom"] = ZoomOAuthProvider(
                client_id=settings.zoom_client_id,
                client_secret=settings.zoom_client_secret,
                redirect_uri=zoom_redirect_uri,
            )
            logger.info(f"Zoom OAuth provider initialized with redirect_uri: {zoom_redirect_uri}")
        else:
            logger.warning("Zoom OAuth credentials not configured")
        
        # HubSpot
        if settings.hubspot_client_id and settings.hubspot_client_secret:
            hubspot_redirect_uri = f"{api_base_url}{api_prefix}/integrations/hubspot/oauth/callback"
            self._providers["hubspot"] = HubSpotOAuthProvider(
                client_id=settings.hubspot_client_id,
                client_secret=settings.hubspot_client_secret,
                redirect_uri=hubspot_redirect_uri,
            )
            logger.info(f"HubSpot OAuth provider initialized with redirect_uri: {hubspot_redirect_uri}")
        else:
            logger.warning("HubSpot OAuth credentials not configured")
        
        # Salesforce
        if settings.salesforce_client_id and settings.salesforce_client_secret:
            salesforce_redirect_uri = f"{api_base_url}{api_prefix}/integrations/salesforce/oauth/callback"
            self._providers["salesforce"] = SalesforceOAuthProvider(
                client_id=settings.salesforce_client_id,
                client_secret=settings.salesforce_client_secret,
                redirect_uri=salesforce_redirect_uri,
            )
            logger.info(f"Salesforce OAuth provider initialized with redirect_uri: {salesforce_redirect_uri}")
        else:
            logger.warning("Salesforce OAuth credentials not configured")
    
    def get_provider(self, integration_id: str) -> Optional[OAuthProvider]:
        """
        Get OAuth provider for an integration.
        
        Args:
            integration_id: Integration ID (e.g., "google_workspace", "microsoft_365", "slack")
            
        Returns:
            OAuth provider instance or None if not found/configured
        """
        return self._providers.get(integration_id)
    
    def get_authorization_url(
        self, 
        integration_id: str, 
        state: str,
        scopes: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Get OAuth authorization URL for an integration.
        
        Args:
            integration_id: Integration ID
            state: CSRF state parameter
            scopes: Optional list of scopes (uses default if not provided)
            
        Returns:
            Authorization URL or None if provider not found/configured
        """
        provider = self.get_provider(integration_id)
        if not provider:
            return None
        
        return provider.get_authorization_url(state, scopes)
    
    def is_configured(self, integration_id: str) -> bool:
        """
        Check if OAuth provider is configured for an integration.
        
        Args:
            integration_id: Integration ID
            
        Returns:
            True if provider is configured, False otherwise
        """
        return integration_id in self._providers


# Global OAuth service instance
_oauth_service: Optional[OAuthService] = None


def get_oauth_service() -> OAuthService:
    """Get global OAuth service instance."""
    global _oauth_service
    if _oauth_service is None:
        _oauth_service = OAuthService()
    return _oauth_service

