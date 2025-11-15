"""
HubSpot OAuth provider implementation.
"""
from typing import List, Optional
from urllib.parse import urlencode

from .base import OAuthProvider


class HubSpotOAuthProvider(OAuthProvider):
    """HubSpot OAuth 2.0 provider."""
    
    AUTHORIZATION_BASE_URL = "https://app.hubspot.com/oauth/authorize"
    TOKEN_URL = "https://api.hubapi.com/oauth/v1/token"
    
    # Default scopes for HubSpot
    DEFAULT_SCOPES = [
        "contacts",      # Read and write contacts
        "deals",         # Read and write deals
        "timeline",      # Read and write timeline events
    ]
    
    def get_authorization_url(self, state: str, scopes: Optional[List[str]] = None) -> str:
        """
        Generate HubSpot OAuth authorization URL.
        
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
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(scopes),  # HubSpot uses space-separated scopes
            "state": state,
        }
        
        return self.build_authorization_url(self.AUTHORIZATION_BASE_URL, params)
    
    def get_token_url(self) -> str:
        """Get HubSpot token exchange URL."""
        return self.TOKEN_URL
    
    def get_default_scopes(self) -> List[str]:
        """Get default HubSpot OAuth scopes."""
        return self.DEFAULT_SCOPES.copy()

