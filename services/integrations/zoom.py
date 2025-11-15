"""
Zoom OAuth provider implementation.
"""
from typing import List, Optional
from urllib.parse import urlencode

from .base import OAuthProvider


class ZoomOAuthProvider(OAuthProvider):
    """Zoom OAuth 2.0 provider."""
    
    AUTHORIZATION_BASE_URL = "https://zoom.us/oauth/authorize"
    TOKEN_URL = "https://zoom.us/oauth/token"
    
    # Default scopes for Zoom
    DEFAULT_SCOPES = [
        "meeting:write",      # Create and manage meetings
        "user:read",          # Read user information
        "recording:read",     # Read meeting recordings
    ]
    
    def get_authorization_url(self, state: str, scopes: Optional[List[str]] = None) -> str:
        """
        Generate Zoom OAuth authorization URL.
        
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
            "scope": " ".join(scopes),  # Zoom uses space-separated scopes
            "state": state,
        }
        
        return self.build_authorization_url(self.AUTHORIZATION_BASE_URL, params)
    
    def get_token_url(self) -> str:
        """Get Zoom token exchange URL."""
        return self.TOKEN_URL
    
    def get_default_scopes(self) -> List[str]:
        """Get default Zoom OAuth scopes."""
        return self.DEFAULT_SCOPES.copy()

