#!/bin/bash
# Script to populate AWS Secrets Manager secrets from env.example file
# Uses JSON format to store multiple keys per secret
# Usage: ./populate_secrets.sh [path_to_env_file]
# Default: ./env.example

AWS_REGION="us-east-1"
ENV_FILE="${1:-env.example}"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file '$ENV_FILE' not found!"
    exit 1
fi

echo "Populating AWS Secrets Manager secrets from $ENV_FILE..."

# Source the env file to read variables
set -a
source "$ENV_FILE"
set +a

# Function to update secret with JSON
update_json_secret() {
    local secret_name=$1
    local json_content=$2
    
    echo "Updating $secret_name..."
    aws secretsmanager put-secret-value \
        --secret-id "$secret_name" \
        --secret-string "$json_content" \
        --region $AWS_REGION > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Updated $secret_name"
    else
        echo "❌ Failed to update $secret_name (secret may not exist)"
    fi
}

# AWS Credentials
update_json_secret "clerk/aws" "{\"access_key_id\":\"${AWS_ACCESS_KEY_ID:-placeholder}\",\"secret_access_key\":\"${AWS_SECRET_ACCESS_KEY:-placeholder}\"}"

# API Configuration
update_json_secret "clerk/api" "{\"base_url\":\"${API_BASE_URL:-placeholder}\",\"rt_gateway_url\":\"${RT_GATEWAY_URL:-placeholder}\"}"

# LiveKit Configuration
update_json_secret "clerk/livekit" "{\"url\":\"${LIVEKIT_URL:-placeholder}\",\"api_key\":\"${LIVEKIT_API_KEY:-placeholder}\",\"api_secret\":\"${LIVEKIT_API_SECRET:-placeholder}\",\"voice_webhook_secret\":\"${LIVEKIT_VOICE_WEBHOOK_SECRET:-placeholder}\",\"phone_number\":\"${PHONE_NUMBER:-placeholder}\",\"webhook_base_url\":\"${WEBHOOK_BASE_URL:-placeholder}\"}"

# AI Service Keys
update_json_secret "clerk/ai" "{\"openai_api_key\":\"${OPENAI_API_KEY:-placeholder}\",\"anthropic_api_key\":\"${ANTHROPIC_API_KEY:-placeholder}\",\"elevenlabs_api_key\":\"${ELEVENLABS_API_KEY:-placeholder}\",\"deepgram_api_key\":\"${DEEPGRAM_API_KEY:-placeholder}\"}"

# Google OAuth
update_json_secret "clerk/google" "{\"oauth_client_config\":\"${GOOGLE_OAUTH_CLIENT_CONFIG:-placeholder}\",\"client_id\":\"${GOOGLE_CLIENT_ID:-placeholder}\",\"client_secret\":\"${GOOGLE_CLIENT_SECRET:-placeholder}\"}"

# Zoom OAuth
update_json_secret "clerk/zoom" "{\"account_id\":\"${ZOOM_ACCOUNT_ID:-placeholder}\",\"client_id\":\"${ZOOM_CLIENT_ID:-placeholder}\",\"client_secret\":\"${ZOOM_CLIENT_SECRET:-placeholder}\",\"redirect_uri\":\"${ZOOM_REDIRECT_URI:-placeholder}\",\"access_token\":\"${ZOOM_ACCESS_TOKEN:-placeholder}\",\"refresh_token\":\"${ZOOM_REFRESH_TOKEN:-placeholder}\"}"

# Microsoft OAuth
update_json_secret "clerk/microsoft" "{\"client_id\":\"${MS_CLIENT_ID:-placeholder}\",\"client_secret\":\"${MS_CLIENT_SECRET:-placeholder}\",\"tenant_id\":\"${MS_TENANT_ID:-placeholder}\"}"

# Meeting Configuration
update_json_secret "clerk/meeting" "{\"ai_email\":\"${AI_EMAIL:-placeholder}\"}"

# Slack Integration
update_json_secret "clerk/slack" "{\"bot_token\":\"${SLACK_BOT_TOKEN:-placeholder}\"}"

# SMTP Configuration
update_json_secret "clerk/smtp" "{\"username\":\"${SMTP_USERNAME:-placeholder}\",\"password\":\"${SMTP_PASSWORD:-placeholder}\"}"

# Security
update_json_secret "clerk/security" "{\"secret_key\":\"${SECRET_KEY:-placeholder}\"}"

# Twilio Configuration
update_json_secret "clerk/twilio" "{\"account_sid\":\"${TWILIO_ACCOUNT_SID:-placeholder}\",\"auth_token\":\"${TWILIO_AUTH_TOKEN:-placeholder}\"}"

# Redis Configuration
update_json_secret "clerk/redis" "{\"url\":\"${REDIS_URL:-placeholder}\"}"

# HubSpot Integration
update_json_secret "clerk/hubspot" "{\"api_key\":\"${HUBSPOT_API_KEY:-placeholder}\"}"

# Fireflies Integration
update_json_secret "clerk/fireflies" "{\"api_key\":\"${FIREFLIES_API_KEY:-placeholder}\"}"

# Recall Integration
update_json_secret "clerk/recall" "{\"api_key\":\"${RECALL_API_KEY:-placeholder}\"}"

echo ""
echo "✅ Finished populating secrets!"
