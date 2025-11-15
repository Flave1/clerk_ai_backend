"""
Salesforce OAuth provider implementation.
"""
from typing import List, Optional
from urllib.parse import urlencode

from .base import OAuthProvider


class SalesforceOAuthProvider(OAuthProvider):
    """Salesforce OAuth 2.0 provider."""
    
    # Note: Salesforce uses instance-specific URLs, but we'll use the common login URL
    # The actual instance URL will be determined during the OAuth flow
    AUTHORIZATION_BASE_URL = "https://login.salesforce.com/services/oauth2/authorize"
    TOKEN_URL = "https://login.salesforce.com/services/oauth2/token"
    
    # Default scopes for Salesforce
    DEFAULT_SCOPES = [
        "api",           # Access the REST API
        "refresh_token", # Get refresh tokens
        "full",          # Full access to all data accessible by the logged-in user
    ]
    
    def get_authorization_url(self, state: str, scopes: Optional[List[str]] = None) -> str:
        """
        Generate Salesforce OAuth authorization URL.
        
        Args:
            state: CSRF state parameter
            scopes: List of OAuth scopes (defaults to DEFAULT_SCOPES)
            
        Returns:
            Authorization URL
        """
        if scopes is None:
            scopes = self.DEFAULT_SCOPES
        
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(scopes),  # Salesforce uses space-separated scopes
            "state": state,
        }
        
        return self.build_authorization_url(self.AUTHORIZATION_BASE_URL, params)
    
    def get_token_url(self) -> str:
        """Get Salesforce token exchange URL."""
        return self.TOKEN_URL
    
    def get_default_scopes(self) -> List[str]:
        """Get default Salesforce OAuth scopes."""
        return self.DEFAULT_SCOPES.copy()

