"""
Base OAuth provider interface.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from urllib.parse import urlencode, urlparse, parse_qs


class OAuthProvider(ABC):
    """Base class for OAuth providers."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        """
        Initialize OAuth provider.
        
        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret
            redirect_uri: OAuth redirect URI
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
    
    @abstractmethod
    def get_authorization_url(self, state: str, scopes: List[str]) -> str:
        """
        Generate OAuth authorization URL.
        
        Args:
            state: CSRF state parameter
            scopes: List of OAuth scopes to request
            
        Returns:
            Authorization URL
        """
        pass
    
    @abstractmethod
    def get_token_url(self) -> str:
        """
        Get token exchange URL.
        
        Returns:
            Token URL
        """
        pass
    
    @abstractmethod
    def get_default_scopes(self) -> List[str]:
        """
        Get default OAuth scopes for this provider.
        
        Returns:
            List of default scopes
        """
        pass
    
    def build_authorization_url(
        self, 
        base_url: str, 
        params: Dict[str, str]
    ) -> str:
        """
        Build authorization URL with query parameters.
        
        Args:
            base_url: Base authorization URL
            params: Query parameters
            
        Returns:
            Complete authorization URL
        """
        return f"{base_url}?{urlencode(params)}"

