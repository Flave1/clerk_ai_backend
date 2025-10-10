"""Unit tests for configuration."""

import pytest
from shared.config import get_settings


class TestConfig:
    """Test configuration loading."""

    def test_get_settings(self):
        """Test getting settings."""
        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, 'aws_access_key_id')
        assert hasattr(settings, 'aws_secret_access_key')
        assert hasattr(settings, 'aws_region')

    def test_environment_variables(self):
        """Test environment variable loading."""
        settings = get_settings()
        # Test that environment variables are loaded
        assert settings.aws_region == "us-east-1"
        assert settings.aws_access_key_id == "testing"
        assert settings.aws_secret_access_key == "testing"