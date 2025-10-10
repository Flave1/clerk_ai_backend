"""
Configuration management for the AI Receptionist system.
"""
import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
class Settings(BaseSettings):
    """Application settings."""

    # Application
    app_name: str = "AI Receptionist"
    app_version: str = "1.0.0"
    debug: bool = True

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    api_base_url: Optional[str] = Field(None, env="API_BASE_URL")  # e.g., "https://api.yourcompany.com"
    rt_gateway_host: str = "0.0.0.0"
    rt_gateway_port: int = 8001

    # WebSocket
    ws_port: int = 8001

    # AWS Configuration
    aws_region: Optional[str] = Field(None, env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(None, env="AWS_SECRET_ACCESS_KEY")

    # DynamoDB
    dynamodb_table_prefix: str = "clerk_"
    conversations_table: str = "conversations"
    turns_table: str = "turns"
    actions_table: str = "actions"
    users_table: str = "users"
    meetings_table: str = "meetings"
    meeting_summaries_table: str = "meeting_summaries"
    meeting_transcriptions_table: str = "meeting_transcriptions"
    meeting_notifications_table: str = "meeting_notifications"

    # SQS/SNS
    sqs_queue_prefix: str = "clerk_"
    actions_queue: str = "actions"
    transcripts_queue: str = "transcripts"
    sns_topic_prefix: str = "clerk_"
    events_topic: str = "events"

    # LiveKit
    livekit_url: Optional[str] = Field(None, env="LIVEKIT_URL")
    livekit_api_key: Optional[str] = Field(None, env="LIVEKIT_API_KEY")
    livekit_api_secret: Optional[str] = Field(None, env="LIVEKIT_API_SECRET")
    livekit_voice_webhook_secret: Optional[str] = Field(None, env="LIVEKIT_VOICE_WEBHOOK_SECRET")
    phone_number: Optional[str] = Field(None, env="PHONE_NUMBER")
    webhook_base_url: Optional[str] = Field(None, env="WEBHOOK_BASE_URL")

    # AI Services
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    elevenlabs_api_key: Optional[str] = Field(None, env="ELEVENLABS_API_KEY")

    # STT/TTS Configuration
    default_voice_id: str = "default"
    default_language: str = "en"
    whisper_model: str = "base"

    # External Integrations
    google_oauth_client_config: Optional[str] = Field(
        None, env="GOOGLE_OAUTH_CLIENT_CONFIG"
    )
    slack_bot_token: Optional[str] = Field(None, env="SLACK_BOT_TOKEN")
    hubspot_api_key: Optional[str] = Field(None, env="HUBSPOT_API_KEY")
    fireflies_api_key: Optional[str] = Field(None, env="FIREFLIES_API_KEY")
    recall_api_key: Optional[str] = Field(None, env="RECALL_API_KEY")
    
    # Meeting Agent OAuth Credentials
    google_client_id: Optional[str] = Field(None, env="GOOGLE_CLIENT_ID")
    google_client_secret: Optional[str] = Field(None, env="GOOGLE_CLIENT_SECRET")
    
    zoom_client_id: Optional[str] = Field(None, env="ZOOM_CLIENT_ID")
    zoom_client_secret: Optional[str] = Field(None, env="ZOOM_CLIENT_SECRET")
    zoom_redirect_uri: Optional[str] = Field(None, env="ZOOM_REDIRECT_URI")
    zoom_access_token: Optional[str] = Field(None, env="ZOOM_ACCESS_TOKEN")
    zoom_refresh_token: Optional[str] = Field(None, env="ZOOM_REFRESH_TOKEN")
    zoom_account_id: Optional[str] = Field(None, env="ZOOM_ACCOUNT_ID")  # For Server-to-Server OAuth
    
    ms_client_id: Optional[str] = Field(None, env="MS_CLIENT_ID")
    ms_client_secret: Optional[str] = Field(None, env="MS_CLIENT_SECRET")
    ms_tenant_id: Optional[str] = Field(None, env="MS_TENANT_ID")
    
    # Meeting Agent Configuration
    ai_email: Optional[str] = Field(None, env="AI_EMAIL")
    meeting_auto_join_enabled: bool = Field(True, env="MEETING_AUTO_JOIN_ENABLED")
    meeting_join_buffer_minutes: int = Field(5, env="MEETING_JOIN_BUFFER_MINUTES")
    meeting_max_join_attempts: int = Field(3, env="MEETING_MAX_JOIN_ATTEMPTS")
    meeting_transcription_enabled: bool = Field(True, env="MEETING_TRANSCRIPTION_ENABLED")
    meeting_summarization_enabled: bool = Field(True, env="MEETING_SUMMARIZATION_ENABLED")
    meeting_email_notifications_enabled: bool = Field(True, env="MEETING_EMAIL_NOTIFICATIONS_ENABLED")
    meeting_slack_notifications_enabled: bool = Field(True, env="MEETING_SLACK_NOTIFICATIONS_ENABLED")

    # Security
    secret_key: str = Field("dev-secret-key", env="SECRET_KEY")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Additional fields for compatibility
    aws_polly_voice_id: str = "Joanna"
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = Field(None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(None, env="SMTP_PASSWORD")


    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Performance
    max_concurrent_conversations: int = 100
    max_conversation_duration_minutes: int = 60
    audio_chunk_size: int = 4096
    audio_sample_rate: int = 16000

    class Config:
        env_file = BASE_DIR / ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
