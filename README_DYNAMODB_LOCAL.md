# Local DynamoDB Setup

This guide explains how to use local DynamoDB for development and testing.

## Quick Start

1. **Start local DynamoDB:**
   ```bash
   # From clerk_backend directory
   cd clerk_backend
   docker-compose up -d dynamodb
   
   # Or use the helper script (from anywhere)
   ./clerk_backend/scripts/dynamodb_local.sh start
   ```

2. **Create tables:**
   ```bash
   cd clerk_backend
   python scripts/create_tables.py create
   ```

3. **Run your application:**
   - If `AWS_ACCESS_KEY_ID` is not set in your `.env` file, the system will automatically use local DynamoDB
   - Or explicitly set `USE_LOCAL_DYNAMODB=true` in your `.env` file

## How It Works

The system automatically detects which DynamoDB to use:

1. **Auto-detection (default behavior):**
   - If `AWS_ACCESS_KEY_ID` is **not set**, the system tries local DynamoDB first
   - If local DynamoDB is not available, it falls back to AWS (using IAM role if available)

2. **Explicit configuration:**
   - Set `USE_LOCAL_DYNAMODB=true` to force local DynamoDB (will fail if not running)
   - Set `USE_LOCAL_DYNAMODB=false` to force AWS DynamoDB

3. **Custom endpoint:**
   - Set `DYNAMODB_LOCAL_ENDPOINT=http://localhost:8001` to use a different endpoint
   - Default is `http://localhost:8001` (port 8001 to avoid conflict with backend on port 8000)

## Environment Variables

Add these to your `.env` file:

```bash
# Force local DynamoDB (optional)
USE_LOCAL_DYNAMODB=true

# Custom local endpoint (optional, default: http://localhost:8001)
DYNAMODB_LOCAL_ENDPOINT=http://localhost:8001

# AWS credentials (set these to use AWS DynamoDB instead)
# AWS_ACCESS_KEY_ID=your-key
# AWS_SECRET_ACCESS_KEY=your-secret
# AWS_REGION=us-east-1
```

## Helper Script

Use the helper script to manage local DynamoDB:

```bash
# Start local DynamoDB
./clerk_backend/scripts/dynamodb_local.sh start

# Stop local DynamoDB
./clerk_backend/scripts/dynamodb_local.sh stop

# Restart local DynamoDB
./clerk_backend/scripts/dynamodb_local.sh restart

# Check status
./clerk_backend/scripts/dynamodb_local.sh status

# View logs
./clerk_backend/scripts/dynamodb_local.sh logs
```

## Docker Compose

The `docker-compose.yml` file in the `clerk_backend` directory includes a DynamoDB Local service:

```yaml
services:
  dynamodb:
    image: amazon/dynamodb-local:latest
    ports:
      - "8001:8000"  # External port 8001 to avoid conflict with backend on 8000
```

## Switching Between Local and AWS

### To use Local DynamoDB:
1. Remove or comment out `AWS_ACCESS_KEY_ID` in your `.env` file
2. Or set `USE_LOCAL_DYNAMODB=true`
3. Start local DynamoDB: `cd clerk_backend && docker-compose up -d dynamodb`

### To use AWS DynamoDB:
1. Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in your `.env` file
2. Or set `USE_LOCAL_DYNAMODB=false`
3. Stop local DynamoDB: `cd clerk_backend && docker-compose stop dynamodb` (optional)

## Notes

- Local DynamoDB data is persisted in a Docker volume (`dynamodb-data`)
- Local DynamoDB uses dummy credentials (`dummy`/`dummy`) - these are required by boto3 but ignored by DynamoDB Local
- The system will log which DynamoDB it's connecting to on startup
- All tables must be created manually using `create_tables.py` script

