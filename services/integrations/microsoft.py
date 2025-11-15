"""
Microsoft 365 OAuth provider implementation.
"""
from typing import List, Optional
from urllib.parse import urlencode

from .base import OAuthProvider


class MicrosoftOAuthProvider(OAuthProvider):
    """Microsoft 365 OAuth 2.0 provider (Microsoft Identity Platform)."""
    
    def __init__(
        self, 
        client_id: str, 
        client_secret: str, 
        redirect_uri: str,
        tenant_id: str = "common"
    ):
        """
        Initialize Microsoft OAuth provider.
        
        Args:
            client_id: Azure AD application (client) ID
            client_secret: Azure AD client secret
            redirect_uri: OAuth redirect URI
            tenant_id: Azure AD tenant ID (default: "common" for multi-tenant)
        """
        super().__init__(client_id, client_secret, redirect_uri)
        self.tenant_id = tenant_id
    
    @property
    def authorization_base_url(self) -> str:
        """Get authorization URL based on tenant."""
        return f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/authorize"
    
    @property
    def token_url(self) -> str:
        """Get token URL based on tenant."""
        return f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
    
    # Default scopes for Microsoft 365
    DEFAULT_SCOPES = [
        "https://graph.microsoft.com/Calendars.ReadWrite",  # Outlook Calendar
        "https://graph.microsoft.com/Mail.ReadWrite",  # Outlook Mail
        "https://graph.microsoft.com/Files.ReadWrite",  # OneDrive
        "https://graph.microsoft.com/Sites.ReadWrite.All",  # SharePoint
        "offline_access",  # Required for refresh token
    ]
    
    def get_authorization_url(self, state: str, scopes: Optional[List[str]] = None) -> str:
        """
        Generate Microsoft OAuth authorization URL.
        
        Args:
            state: CSRF state parameter
            scopes: List of OAuth scopes (defaults to DEFAULT_SCOPES)
            
        Returns:
            Authorization URL
        """
        if scopes is None:
            scopes = self.DEFAULT_SCOPES
        
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "response_mode": "query",
            "scope": " ".join(scopes),
            "state": state,
        }
        
        return self.build_authorization_url(self.authorization_base_url, params)
    
    def get_token_url(self) -> str:
        """Get Microsoft token exchange URL."""
        return self.token_url
    
    def get_default_scopes(self) -> List[str]:
        """Get default Microsoft OAuth scopes."""
        return self.DEFAULT_SCOPES.copy()

