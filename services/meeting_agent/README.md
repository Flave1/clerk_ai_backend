# Browser Bot Container Setup

The Clerk AI Reception system includes a headless browser bot that can automatically join meetings (Zoom, Google Meet, or Teams) and stream audio to the RT Gateway for real-time processing.

## Overview

The browser bot is implemented as a containerized Node.js application using Playwright for browser automation. It can:

- Join meetings automatically using meeting URLs
- Capture and stream audio from meetings
- Inject TTS audio back into meetings
- Handle platform-specific UI interactions
- Communicate with the RT Gateway via WebSocket

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Meeting URL   │───▶│   Browser Bot    │───▶│   RT Gateway    │
│                 │    │   Container      │    │   (WebSocket)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Meeting API    │
                       │   (REST)         │
                       └──────────────────┘
```

## Quick Start

### 1. Build the Container

```bash
cd services/meeting_agent/browser_bot
docker build -t clerk-browser-bot .
```

### 2. Run Locally

```bash
docker run -d \
  --name clerk-bot \
  --network host \
  --shm-size=2g \
  -e MEETING_URL="https://meet.google.com/your-meeting-id" \
  -e BOT_NAME="Clerk AI Bot" \
  -e PLATFORM="google_meet" \
  -e RT_GATEWAY_URL="ws://localhost:8001" \
  -e API_BASE_URL="http://localhost:8000" \
  clerk-browser-bot
```

### 3. Using Docker Compose

```bash
cd infra/docker
docker-compose -f browser_bot-compose.yml up -d
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEETING_URL` | Meeting URL to join | Required |
| `BOT_NAME` | Name of the bot participant | "Clerk AI Bot" |
| `PLATFORM` | Meeting platform (google_meet, zoom, teams) | "google_meet" |
| `RT_GATEWAY_URL` | WebSocket URL for RT Gateway | "ws://localhost:8001" |
| `API_BASE_URL` | REST API base URL | "http://localhost:8000" |
| `MEETING_ID` | Unique meeting identifier | Auto-generated |
| `SESSION_ID` | Unique session identifier | Auto-generated |
| `JOIN_TIMEOUT_SEC` | Timeout for joining meeting | 60 |
| `AUDIO_SAMPLE_RATE` | Audio sample rate | 16000 |
| `AUDIO_CHANNELS` | Number of audio channels | 1 |
| `LOG_LEVEL` | Logging level | "info" |

### Platform-Specific Configuration

#### Google Meet
- Automatically handles guest join flow
- Mutes microphone and camera by default
- Supports breakout rooms

#### Zoom
- Handles waiting room if enabled
- Supports password-protected meetings
- Handles host controls

#### Microsoft Teams
- Supports guest access
- Handles authentication flows
- Compatible with Teams web client

## Development

### Local Development Setup

1. Install dependencies:
```bash
cd services/meeting_agent/browser_bot
npm install
```

2. Run in development mode:
```bash
npm run dev
```

3. Test with a meeting URL:
```bash
MEETING_URL="https://meet.google.com/test" npm start
```

### Testing

Run unit tests:
```bash
npm test
```

Run integration tests:
```bash
pytest tests/integration/meeting_agent/test_browser_bot_integration.py
```

## Production Deployment

### AWS ECS/Fargate

1. Build and push image to ECR:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker build -t clerk-browser-bot .
docker tag clerk-browser-bot:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/clerk-browser-bot:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/clerk-browser-bot:latest
```

2. Deploy using Terraform:
```bash
cd infra/terraform
terraform apply
```

3. Scale bot instances:
```bash
aws ecs update-service --cluster clerk-cluster --service browser-bot-service --desired-count 3
```

### Kubernetes

1. Apply Kubernetes manifests:
```bash
kubectl apply -f infra/kubernetes/
```

2. Scale deployment:
```bash
kubectl scale deployment browser-bot --replicas=5
```

## Monitoring and Logging

### Health Checks

The container includes a health check endpoint:
```bash
curl http://localhost:3000/health
```

### Logs

View container logs:
```bash
docker logs clerk-bot
```

### Metrics

Monitor bot performance:
- Active bot count
- Meeting join success rate
- Audio streaming quality
- Container resource usage

## Troubleshooting

### Common Issues

1. **Bot fails to join meeting**
   - Check meeting URL format
   - Verify network connectivity
   - Check browser console for errors

2. **Audio not streaming**
   - Verify RT Gateway is running
   - Check WebSocket connection
   - Verify audio permissions

3. **Container crashes**
   - Check memory limits
   - Verify Chrome installation
   - Check shared memory size

### Debug Mode

Enable debug logging:
```bash
docker run -e LOG_LEVEL=debug clerk-browser-bot
```

### Performance Tuning

For high-concurrency scenarios:

1. Increase shared memory:
```bash
docker run --shm-size=4g clerk-browser-bot
```

2. Adjust CPU/memory limits:
```bash
docker run --cpus=2 --memory=4g clerk-browser-bot
```

3. Use host networking for better performance:
```bash
docker run --network host clerk-browser-bot
```

## Security Considerations

- Bot containers run with minimal privileges
- Audio data is encrypted in transit
- Meeting credentials are not stored
- Container images are scanned for vulnerabilities
- Network access is restricted to necessary services

## API Integration

### Launch Bot Programmatically

```python
from services.meeting_agent.main import meeting_agent_service

# Launch browser bot for a meeting
success = await meeting_agent_service.launch_browser_bot(
    meeting_id="meeting-123",
    platform="google_meet",
    meeting_url="https://meet.google.com/abc-defg-hij",
    bot_name="AI Assistant"
)
```

### Monitor Bot Status

```python
# Get browser bot status
status = await meeting_agent_service.get_browser_bot_status()
print(f"Active bots: {status['active_bots_count']}")
```

### Stop Bot

```python
# Stop browser bot
success = await meeting_agent_service.stop_browser_bot("meeting-123")
```

## Contributing

1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation
4. Ensure backward compatibility
5. Test with multiple meeting platforms
