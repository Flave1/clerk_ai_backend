#!/bin/bash
# Script to update ECS task definition with correct secret ARNs
# This script fetches the actual ARNs from AWS Secrets Manager and updates the task definition

AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="588412562130"
TASK_DEF_FILE="ecs-task-def.json"

echo "Fetching secret ARNs from AWS Secrets Manager..."

# Function to get full ARN for a secret
get_secret_arn() {
    local secret_name=$1
    aws secretsmanager describe-secret \
        --secret-id "$secret_name" \
        --region $AWS_REGION \
        --query 'ARN' \
        --output text 2>/dev/null
}

# Fetch all ARNs (excluding removed secrets)
AWS_ARN=$(get_secret_arn "clerk/aws")
API_ARN=$(get_secret_arn "clerk/api")
# LIVEKIT_ARN removed - not needed
AI_ARN=$(get_secret_arn "clerk/ai")
GOOGLE_ARN=$(get_secret_arn "clerk/google")
ZOOM_ARN=$(get_secret_arn "clerk/zoom")
MICROSOFT_ARN=$(get_secret_arn "clerk/microsoft")
MEETING_ARN=$(get_secret_arn "clerk/meeting")
SLACK_ARN=$(get_secret_arn "clerk/slack")
SMTP_ARN=$(get_secret_arn "clerk/smtp")
SECURITY_ARN=$(get_secret_arn "clerk/security")
TWILIO_ARN=$(get_secret_arn "clerk/twilio")
REDIS_ARN=$(get_secret_arn "clerk/redis")
HUBSPOT_ARN=$(get_secret_arn "clerk/hubspot")
# FIREFLIES_ARN removed - not needed
RECALL_ARN=$(get_secret_arn "clerk/recall")

echo "Found ARNs:"
echo "  AWS: $AWS_ARN"
echo "  API: $API_ARN"
echo "  AI: $AI_ARN"
echo ""
echo "Updating task definition..."

# Create a backup
cp "$TASK_DEF_FILE" "${TASK_DEF_FILE}.backup"

# Update the task definition using Python for JSON manipulation
python3 << EOF
import json
import sys

with open('$TASK_DEF_FILE', 'r') as f:
    task_def = json.load(f)

# Secret ARNs (from bash variables)
arns = {
    'AWS': '$AWS_ARN',
    'API': '$API_ARN',
    'AI': '$AI_ARN',
    'GOOGLE': '$GOOGLE_ARN',
    'ZOOM': '$ZOOM_ARN',
    'MICROSOFT': '$MICROSOFT_ARN',
    'MEETING': '$MEETING_ARN',
    'SLACK': '$SLACK_ARN',
    'SMTP': '$SMTP_ARN',
    'SECURITY': '$SECURITY_ARN',
    'TWILIO': '$TWILIO_ARN',
    'REDIS': '$REDIS_ARN',
    'HUBSPOT': '$HUBSPOT_ARN',
    'RECALL': '$RECALL_ARN',
}

# Mapping of secret names to their ARNs and keys
secret_mappings = {
    'AWS_ACCESS_KEY_ID': ('AWS', 'access_key_id'),
    'AWS_SECRET_ACCESS_KEY': ('AWS', 'secret_access_key'),
    'API_BASE_URL': ('API', 'base_url'),
    'RT_GATEWAY_URL': ('API', 'rt_gateway_url'),
    'OPENAI_API_KEY': ('AI', 'openai_api_key'),
    'ANTHROPIC_API_KEY': ('AI', 'anthropic_api_key'),
    'ELEVENLABS_API_KEY': ('AI', 'elevenlabs_api_key'),
    'DEEPGRAM_API_KEY': ('AI', 'deepgram_api_key'),
    'GOOGLE_OAUTH_CLIENT_CONFIG': ('GOOGLE', 'oauth_client_config'),
    'GOOGLE_CLIENT_ID': ('GOOGLE', 'client_id'),
    'GOOGLE_CLIENT_SECRET': ('GOOGLE', 'client_secret'),
    'ZOOM_ACCOUNT_ID': ('ZOOM', 'account_id'),
    'ZOOM_CLIENT_ID': ('ZOOM', 'client_id'),
    'ZOOM_CLIENT_SECRET': ('ZOOM', 'client_secret'),
    'ZOOM_REDIRECT_URI': ('ZOOM', 'redirect_uri'),
    'ZOOM_ACCESS_TOKEN': ('ZOOM', 'access_token'),
    'ZOOM_REFRESH_TOKEN': ('ZOOM', 'refresh_token'),
    'MS_CLIENT_ID': ('MICROSOFT', 'client_id'),
    'MS_CLIENT_SECRET': ('MICROSOFT', 'client_secret'),
    'MS_TENANT_ID': ('MICROSOFT', 'tenant_id'),
    'AI_EMAIL': ('MEETING', 'ai_email'),
    'SLACK_BOT_TOKEN': ('SLACK', 'bot_token'),
    'SMTP_USERNAME': ('SMTP', 'username'),
    'SMTP_PASSWORD': ('SMTP', 'password'),
    'SECRET_KEY': ('SECURITY', 'secret_key'),
    'TWILIO_ACCOUNT_SID': ('TWILIO', 'account_sid'),
    'REDIS_URL': ('REDIS', 'url'),
    'HUBSPOT_API_KEY': ('HUBSPOT', 'api_key'),
    'RECALL_API_KEY': ('RECALL', 'api_key'),
}

# Update secrets in container definition
container = task_def['containerDefinitions'][0]
for secret in container['secrets']:
    env_name = secret['name']
    if env_name in secret_mappings:
        arn_key, json_key = secret_mappings[env_name]
        full_arn = arns[arn_key]
        secret['valueFrom'] = f"{full_arn}:{json_key}::"
        print(f"Updated {env_name}: {secret['valueFrom']}")

# Write updated task definition
with open('$TASK_DEF_FILE', 'w') as f:
    json.dump(task_def, f, indent=4)

print(f"\n✅ Task definition updated successfully!")
print(f"📝 Backup saved to: ${TASK_DEF_FILE}.backup")
EOF

echo ""
echo "✅ Task definition ARNs updated!"
echo "📝 Review the changes and redeploy using: ./scripts/deploy.sh"
