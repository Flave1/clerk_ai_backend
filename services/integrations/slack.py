"""
Slack OAuth provider implementation.
"""
from typing import List, Optional
from urllib.parse import urlencode

from .base import OAuthProvider


class SlackOAuthProvider(OAuthProvider):
    """Slack OAuth 2.0 provider."""
    
    AUTHORIZATION_BASE_URL = "https://slack.com/oauth/v2/authorize"
    TOKEN_URL = "https://slack.com/api/oauth.v2.access"
    
    # Default scopes for Slack (Bot Token Scopes)
    # These must match the scopes configured in your Slack app
    DEFAULT_SCOPES = [
        "channels:read",        # View basic information about public channels
        "chat:write",           # Send messages as the bot
        "chat:write.public",    # Send messages to channels the bot isn't a member of
        "files:read",           # View files shared in channels
        "users:read",           # View people in a workspace
        "users:read.email",     # View email addresses of people in the workspace
    ]
    
    def get_authorization_url(self, state: str, scopes: Optional[List[str]] = None) -> str:
        """
        Generate Slack OAuth authorization URL.
        
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
            "scope": ",".join(scopes),  # Slack uses comma-separated scopes
            "redirect_uri": self.redirect_uri,
            "state": state,
        }
        
        return self.build_authorization_url(self.AUTHORIZATION_BASE_URL, params)
    
    def get_token_url(self) -> str:
        """Get Slack token exchange URL."""
        return self.TOKEN_URL
    
    def get_default_scopes(self) -> List[str]:
        """Get default Slack OAuth scopes."""
        return self.DEFAULT_SCOPES.copy()

