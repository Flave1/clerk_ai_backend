"""
Configuration management for the Aurray system.
"""
import os
from typing import Optional, List

from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
class Settings(BaseSettings):
    """Application settings."""

    # Application
    app_name: str = "Aurray"
    app_version: str = "1.0.0"
    debug: bool = True

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    api_base_url: str = "https://api.auray.net"
    frontend_base_url: str = "https://auray.net"
    rt_gateway_host: str = "0.0.0.0"
    rt_gateway_port: int = 8001
    rt_gateway_base_url: str = "ws://api.auray.net"
    allowed_cors_origins: str = "https://auray.net,https://www.auray.net,https://www.aurray.co.uk,http://www.aurray.co.uk,https://localhost:3443,http://localhost:3000"
    cors_allow_credentials: bool = True

    # WebSocket
    ws_port: int = 8001

    # AWS Configuration (Optional - for other AWS services)
    aws_region: str = Field("us-east-1", env="AWS_REGION")  # Default to us-east-1
    aws_access_key_id: Optional[str] = Field("AKIAYSAA7FLJBBJVGB7U", env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field("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", env="AWS_SECRET_ACCESS_KEY")

    # MongoDB Configuration
    mongodb_url: str = Field(
        "mongodb+srv://aurray_db_user:Aurray_pass_11@aurraycluster.etyfomt.mongodb.net/?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=true",
        env="MONGODB_URL"
    )
    mongodb_database: str = Field("aurray", env="MONGODB_DATABASE")
    mongodb_collection_prefix: str = "aurray_"
    conversations_collection: str = "conversations"
    turns_collection: str = "turns"
    actions_collection: str = "actions"
    users_collection: str = "users"
    meetings_collection: str = "meetings"
    meeting_summaries_collection: str = "meeting_summaries"
    meeting_transcriptions_collection: str = "meeting_transcriptions"
    meeting_notifications_collection: str = "meeting_notifications"
    meeting_contexts_collection: str = "meeting_contexts"
    api_keys_collection: str = "api_keys"
    user_integrations_collection: str = "user_integrations"
    newsletter_collection: str = "newsletter_subscriptions"
    rooms_collection: str = "rooms"

    # SQS/SNS (Optional - for AWS services)
    sqs_queue_prefix: str = "aurray_"
    actions_queue: str = "actions"
    transcripts_queue: str = "transcripts"
    sns_topic_prefix: str = "aurray_"
    events_topic: str = "events"


    # AI Services
    openai_api_key: Optional[str] = Field("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", env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    elevenlabs_api_key: Optional[str] = Field("sk_09e9eedbc6f49b5cdd0e9c85235f1f7f62cd8984e60f5091", env="ELEVENLABS_API_KEY")
    deepgram_api_key: Optional[str] = Field(None, env="DEEPGRAM_API_KEY")

    # STT/TTS Configuration
    default_voice_id: str = "default"
    default_language: str = "en"
    whisper_model: str = "base"

    # External Integrations
    google_oauth_client_config: Optional[str] = Field(
        None, env="GOOGLE_OAUTH_CLIENT_CONFIG"
    )

    ms_client_id: Optional[str] = Field("2ad3ba05-8462-438f-9957-427b1f5c7b16", env="MS_CLIENT_ID")
    ms_client_secret: Optional[str] = Field("r2o8Q~Y6ZZY9-TzX.3V8KQ~ScxqvlIT_czcUgc9e", env="MS_CLIENT_SECRET")
    ms_tenant_id: Optional[str] = Field("common", env="MS_TENANT_ID")
    
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

    # Meeting Agent Configuration
    ai_email: str = "aurray@auray.net"
    meeting_auto_join_enabled: bool = True
    meeting_join_buffer_minutes: int = 5
    meeting_max_join_attempts: int = 3
    meeting_transcription_enabled: bool = True
    meeting_summarization_enabled: bool = True
    meeting_email_notifications_enabled: bool = True
    meeting_slack_notifications_enabled: bool = True

    # Phone/Webhook (if needed for other telephony providers)
    phone_number: Optional[str] = Field(None, env="PHONE_NUMBER")
    webhook_base_url: Optional[str] = Field(None, env="WEBHOOK_BASE_URL")

    # AI Services
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    elevenlabs_api_key: Optional[str] = Field(None, env="ELEVENLABS_API_KEY")
    deepgram_api_key: Optional[str] = Field(None, env="DEEPGRAM_API_KEY")
    aurray_api_key: Optional[str] = Field("sk_live_SCSAwAz3pD-fRJmUQ9qnWIm3vdGgGimSGZ74da3TvFg", env="AURRAY_API_KEY")

    # STT/TTS Configuration
    default_voice_id: str = "f5HLTX707KIM4SzJYzSz"
    default_language: str = "en"
    whisper_model: str = "base"

    # External Integrations
    google_oauth_client_config: Optional[str] = Field(
        None, env="GOOGLE_OAUTH_CLIENT_CONFIG"
    )
    slack_bot_token: Optional[str] = Field("xoxb-1266716749669-9911008266324-giNAvdXEQ4pTjEyBZ6J7ccH5", env="SLACK_BOT_TOKEN")
    slack_client_id: Optional[str] = Field("1266716749669.9907403390882", env="SLACK_CLIENT_ID")
    slack_client_secret: Optional[str] = Field("745c33a2d777dec004a3c0b815f3c2b7", env="SLACK_CLIENT_SECRET")
    hubspot_api_key: Optional[str] = Field(None, env="HUBSPOT_API_KEY")
    hubspot_client_id: Optional[str] = Field(None, env="HUBSPOT_CLIENT_ID")
    hubspot_client_secret: Optional[str] = Field(None, env="HUBSPOT_CLIENT_SECRET")
    salesforce_client_id: Optional[str] = Field(None, env="SALESFORCE_CLIENT_ID")
    salesforce_client_secret: Optional[str] = Field(None, env="SALESFORCE_CLIENT_SECRET")
    fireflies_api_key: Optional[str] = Field(None, env="FIREFLIES_API_KEY")
    recall_api_key: Optional[str] = Field(None, env="RECALL_API_KEY")
    
    # Meeting Agent OAuth Credentials
    google_client_id: Optional[str] = Field("748365731205-ppd3v9nm4rs67438ak37sjf0tegiaumr.apps.googleusercontent.com", env="GOOGLE_CLIENT_ID")
    google_client_secret: Optional[str] = Field("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", env="GOOGLE_CLIENT_SECRET")
    
    zoom_client_id: Optional[str] = Field("sh25xZ7SCy2WtsBru5Vg", env="ZOOM_CLIENT_ID")
    zoom_client_secret: Optional[str] = Field(None, env="ZOOM_CLIENT_SECRET")
    zoom_access_token: Optional[str] = Field(None, env="ZOOM_ACCESS_TOKEN")
    zoom_refresh_token: Optional[str] = Field(None, env="ZOOM_REFRESH_TOKEN")
    zoom_account_id: Optional[str] = Field("uWsUVfKxQCWe_mI1Ee-JZg", env="ZOOM_ACCOUNT_ID")  # For Server-to-Server OAuth
    

    # Security
    secret_key: str = Field("dev-secret-key", env="SECRET_KEY")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(52560000, env="ACCESS_TOKEN_EXPIRE_MINUTES")  # ~100 years (effectively never expires)

    # Additional fields for compatibility
    aws_polly_voice_id: str = "Joanna"
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = Field(None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(None, env="SMTP_PASSWORD")

    # Browser Bot Configuration
    browser_bot_enabled: bool = Field(True, env="BROWSER_BOT_ENABLED")
    bot_image: str = "bot_staging:v1.0.0"
    bot_container_cpu: str = Field("1024", env="BOT_CONTAINER_CPU")
    bot_container_memory: str = Field("2048", env="BOT_CONTAINER_MEMORY")
    bot_join_timeout_sec: int = Field(60, env="BOT_JOIN_TIMEOUT_SEC")
    max_concurrent_bots: int = Field(5, env="MAX_CONCURRENT_BOTS")
    bot_deployment_method: str = Field("ecs", env="BOT_DEPLOYMENT_METHOD")  # 'docker', 'ecs', 'subprocess', or 'auto'

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Performance
    max_concurrent_conversations: int = 100
    max_conversation_duration_minutes: int = 60
    audio_chunk_size: int = 4096
    audio_sample_rate: int = 16000

    # External Turn Manager (meeting bot)
    use_external_turn_manager: bool = True

    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")

    def get_cors_origins(self) -> List[str]:
        """Parse allowed_cors_origins string into a list."""
        if not self.allowed_cors_origins:
            return []
        return [origin.strip() for origin in self.allowed_cors_origins.split(",") if origin.strip()]

    class Config:
        env_file = BASE_DIR / ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
