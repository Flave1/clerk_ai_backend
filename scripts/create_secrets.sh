#!/bin/bash
# Script to create AWS Secrets Manager secrets for Clerk Backend ECS Task Definition
# Uses flat string secrets (one secret per environment variable)
# Usage: ./create_secrets.sh

AWS_REGION="us-east-1"

echo "Creating AWS Secrets Manager secrets for Clerk Backend..."

# AWS Credentials
aws secretsmanager create-secret \
  --name aurray/aws-access-key-id \
  --description "AWS Access Key ID for Clerk Backend" \
  --secret-string "REPLACE_WITH_YOUR_AWS_ACCESS_KEY_ID" \
  --region $AWS_REGION || echo "Secret aurray/aws-access-key-id may already exist"

aws secretsmanager create-secret \
  --name aurray/aws-secret-access-key \
  --description "AWS Secret Access Key for Clerk Backend" \
  --secret-string "REPLACE_WITH_YOUR_AWS_SECRET_ACCESS_KEY" \
  --region $AWS_REGION || echo "Secret aurray/aws-secret-access-key may already exist"

# AI Service Keys
aws secretsmanager create-secret \
  --name aurray/openai-api-key \
  --description "OpenAI API Key for Clerk Backend" \
  --secret-string "REPLACE_WITH_OPENAI_API_KEY" \
  --region $AWS_REGION || echo "Secret aurray/openai-api-key may already exist"

aws secretsmanager create-secret \
  --name aurray/anthropic-api-key \
  --description "Anthropic API Key for Clerk Backend" \
  --secret-string "REPLACE_WITH_ANTHROPIC_API_KEY" \
  --region $AWS_REGION || echo "Secret aurray/anthropic-api-key may already exist"

aws secretsmanager create-secret \
  --name aurray/elevenlabs-api-key \
  --description "ElevenLabs API Key for Clerk Backend" \
  --secret-string "REPLACE_WITH_ELEVENLABS_API_KEY" \
  --region $AWS_REGION || echo "Secret aurray/elevenlabs-api-key may already exist"

aws secretsmanager create-secret \
  --name aurray/deepgram-api-key \
  --description "Deepgram API Key for Clerk Backend" \
  --secret-string "REPLACE_WITH_DEEPGRAM_API_KEY" \
  --region $AWS_REGION || echo "Secret aurray/deepgram-api-key may already exist"

# Google OAuth
aws secretsmanager create-secret \
  --name aurray/google-oauth-client-config \
  --description "Google OAuth Client Config for Clerk Backend" \
  --secret-string "google_secrets.json" \
  --region $AWS_REGION || echo "Secret aurray/google-oauth-client-config may already exist"

aws secretsmanager create-secret \
  --name aurray/google-client-id \
  --description "Google Client ID for Clerk Backend" \
  --secret-string "REPLACE_WITH_GOOGLE_CLIENT_ID" \
  --region $AWS_REGION || echo "Secret aurray/google-client-id may already exist"

aws secretsmanager create-secret \
  --name aurray/google-client-secret \
  --description "Google Client Secret for Clerk Backend" \
  --secret-string "REPLACE_WITH_GOOGLE_CLIENT_SECRET" \
  --region $AWS_REGION || echo "Secret aurray/google-client-secret may already exist"

# Zoom OAuth
aws secretsmanager create-secret \
  --name aurray/zoom-account-id \
  --description "Zoom Account ID for Clerk Backend" \
  --secret-string "REPLACE_WITH_ZOOM_ACCOUNT_ID" \
  --region $AWS_REGION || echo "Secret aurray/zoom-account-id may already exist"

aws secretsmanager create-secret \
  --name aurray/zoom-client-id \
  --description "Zoom Client ID for Clerk Backend" \
  --secret-string "REPLACE_WITH_ZOOM_CLIENT_ID" \
  --region $AWS_REGION || echo "Secret aurray/zoom-client-id may already exist"

aws secretsmanager create-secret \
  --name aurray/zoom-client-secret \
  --description "Zoom Client Secret for Clerk Backend" \
  --secret-string "REPLACE_WITH_ZOOM_CLIENT_SECRET" \
  --region $AWS_REGION || echo "Secret aurray/zoom-client-secret may already exist"

aws secretsmanager create-secret \
  --name aurray/zoom-access-token \
  --description "Zoom Access Token for Clerk Backend" \
  --secret-string "" \
  --region $AWS_REGION || echo "Secret aurray/zoom-access-token may already exist"

aws secretsmanager create-secret \
  --name aurray/zoom-refresh-token \
  --description "Zoom Refresh Token for Clerk Backend" \
  --secret-string "" \
  --region $AWS_REGION || echo "Secret aurray/zoom-refresh-token may already exist"

# Microsoft OAuth
aws secretsmanager create-secret \
  --name aurray/ms-client-id \
  --description "Microsoft Client ID for Clerk Backend" \
  --secret-string "REPLACE_WITH_MS_CLIENT_ID" \
  --region $AWS_REGION || echo "Secret aurray/ms-client-id may already exist"

aws secretsmanager create-secret \
  --name aurray/ms-client-secret \
  --description "Microsoft Client Secret for Clerk Backend" \
  --secret-string "REPLACE_WITH_MS_CLIENT_SECRET" \
  --region $AWS_REGION || echo "Secret aurray/ms-client-secret may already exist"

aws secretsmanager create-secret \
  --name aurray/ms-tenant-id \
  --description "Microsoft Tenant ID for Clerk Backend" \
  --secret-string "REPLACE_WITH_MS_TENANT_ID" \
  --region $AWS_REGION || echo "Secret aurray/ms-tenant-id may already exist"

# Meeting Configuration
aws secretsmanager create-secret \
  --name aurray/ai-email \
  --description "AI Email for Clerk Backend" \
  --secret-string "placeholder" \
  --region $AWS_REGION || echo "Secret aurray/ai-email may already exist"

# Slack Integration
aws secretsmanager create-secret \
  --name aurray/slack-bot-token \
  --description "Slack Bot Token for Clerk Backend" \
  --secret-string "REPLACE_WITH_SLACK_BOT_TOKEN" \
  --region $AWS_REGION || echo "Secret aurray/slack-bot-token may already exist"

# SMTP Configuration
aws secretsmanager create-secret \
  --name aurray/smtp-username \
  --description "SMTP Username for Clerk Backend" \
  --secret-string "REPLACE_WITH_SMTP_USERNAME" \
  --region $AWS_REGION || echo "Secret aurray/smtp-username may already exist"

aws secretsmanager create-secret \
  --name aurray/smtp-password \
  --description "SMTP Password for Clerk Backend" \
  --secret-string "REPLACE_WITH_SMTP_PASSWORD" \
  --region $AWS_REGION || echo "Secret aurray/smtp-password may already exist"

# Security
aws secretsmanager create-secret \
  --name aurray/secret-key \
  --description "Secret Key for JWT Tokens" \
  --secret-string "your-secret-key-for-jwt-tokens" \
  --region $AWS_REGION || echo "Secret aurray/secret-key may already exist"

# HubSpot Integration
aws secretsmanager create-secret \
  --name aurray/hubspot-api-key \
  --description "HubSpot API Key for Clerk Backend" \
  --secret-string "placeholder" \
  --region $AWS_REGION || echo "Secret aurray/hubspot-api-key may already exist"

echo ""
echo "✅ All secrets created successfully!"
echo ""
echo "Next steps:"
echo "1. Update any placeholder values using AWS CLI:"
echo "   aws secretsmanager put-secret-value --secret-id aurray/ai-email --secret-string 'your-value' --region $AWS_REGION"
echo "   aws secretsmanager put-secret-value --secret-id aurray/hubspot-api-key --secret-string 'your-value' --region $AWS_REGION"
echo ""
echo "2. Register the updated task definition:"
echo "   aws ecs register-task-definition --cli-input-json file://ecs-task-def.json --region $AWS_REGION"
