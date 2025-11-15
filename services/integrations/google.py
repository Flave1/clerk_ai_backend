"""
Google OAuth provider implementation.
"""
from typing import List, Optional
from urllib.parse import urlencode

from .base import OAuthProvider


class GoogleOAuthProvider(OAuthProvider):
    """Google OAuth 2.0 provider."""
    
    AUTHORIZATION_BASE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    
    # Default scopes for Google Workspace
    DEFAULT_SCOPES = [
        "https://www.googleapis.com/auth/calendar",  # Calendar
        "https://www.googleapis.com/auth/gmail.send",  # Gmail send
        "https://www.googleapis.com/auth/drive.readonly",  # Drive read
        "https://www.googleapis.com/auth/documents.readonly",  # Docs read
    ]
    
    def get_authorization_url(self, state: str, scopes: Optional[List[str]] = None) -> str:
        """
        Generate Google OAuth authorization URL.
        
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
            "response_type": "code",
            "scope": " ".join(scopes),
            "state": state,
            "access_type": "offline",  # Required to get refresh token
            "prompt": "consent",  # Force consent screen to get refresh token
        }
        
        return self.build_authorization_url(self.AUTHORIZATION_BASE_URL, params)
    
    def get_token_url(self) -> str:
        """Get Google token exchange URL."""
        return self.TOKEN_URL
    
    def get_default_scopes(self) -> List[str]:
        """Get default Google OAuth scopes."""
        return self.DEFAULT_SCOPES.copy()

