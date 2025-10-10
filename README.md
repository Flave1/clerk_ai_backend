# AI Receptionist Backend

A real-time AI receptionist system that can answer calls, join video rooms, transcribe speech, respond naturally, and execute tasks through external integrations.

## 🏗️ Architecture

The backend consists of three main services:

### 1. RT Gateway (`services/rt-gateway/`)
- **Real-time audio/video processing** via LiveKit
- **Speech-to-Text** using Whisper/AWS Transcribe
- **Text-to-Speech** using ElevenLabs/AWS Polly
- **LLM integration** with OpenAI/Anthropic
- **Dialogue management** and conversation flow

### 2. API Service (`services/api/`)
- **REST API** for dashboard and admin operations
- **WebSocket connections** for real-time updates
- **Data access layer** with DynamoDB
- **Authentication** and authorization

### 3. Workers (`services/workers/`)
- **Background processing** via SQS
- **External integrations** (Calendar, Slack, CRM)
- **Tool execution** and result handling

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- AWS CLI configured (optional for local development)
- LiveKit server
- API keys for AI services

### Installation

1. **Clone and setup:**
```bash
cd clerk_backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start services:**

**RT Gateway:**
```bash
cd services/rt-gateway
python app.py
```

**API Service:**
```bash
cd services/api
python main.py
```

**Worker:**
```bash
cd services/workers
python worker.py
```

## 🔧 Configuration

### Required Environment Variables

```bash
# LiveKit (Required)
LIVEKIT_URL="wss://your-livekit-server.com"
LIVEKIT_API_KEY="your_api_key"
LIVEKIT_API_SECRET="your_api_secret"

# AI Services (At least one required)
OPENAI_API_KEY="your_openai_key"
ANTHROPIC_API_KEY="your_anthropic_key"
ELEVENLABS_API_KEY="your_elevenlabs_key"

# Security
SECRET_KEY="your_secret_key"
```

### Optional Integrations

```bash
# Google Calendar

# Slack
SLACK_BOT_TOKEN="xoxb-your-slack-bot-token"

# AWS Services
AWS_ACCESS_KEY_ID="your_aws_key"
AWS_SECRET_ACCESS_KEY="your_aws_secret"
```

## 📡 API Endpoints

### RT Gateway (`http://localhost:8000`)

- `POST /conversations/start` - Start a conversation
- `POST /conversations/{id}/end` - End a conversation
- `POST /stt/transcribe` - Transcribe audio
- `POST /tts/synthesize` - Synthesize speech
- `POST /llm/generate` - Generate LLM response
- `WebSocket /ws/{conversation_id}` - Real-time communication

### API Service (`http://localhost:8000`)

- `GET /api/v1/conversations` - List conversations
- `GET /api/v1/conversations/{id}` - Get conversation details
- `GET /api/v1/conversations/{id}/turns` - Get conversation turns
- `GET /api/v1/actions` - List actions
- `GET /api/v1/actions/{id}` - Get action details
- `GET /api/v1/rooms` - List active rooms
- `WebSocket /ws` - Real-time dashboard updates

## 🛠️ Development

### Project Structure

```
clerk_backend/
├── shared/                 # Shared modules
│   ├── schemas.py         # Pydantic models
│   └── config.py          # Configuration
├── services/
│   ├── rt-gateway/        # Real-time processing
│   ├── api/               # REST API & WebSocket
│   └── workers/           # Background processing
├── infra/                 # Infrastructure
├── tests/                 # Test suites
└── requirements.txt       # Dependencies
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# All tests
pytest
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint code
flake8 .
mypy .
```

## 🔌 External Integrations

### Calendar Integration
- **Google Calendar**: Create, update, delete events
- **Microsoft Outlook**: Full calendar management
- **Meeting scheduling**: Automatic conflict detection

### Communication Tools
- **Slack**: Send messages, get user info, list channels
- **Email**: Send emails via SMTP
- **SMS**: Send SMS via Twilio

### CRM Integration
- **HubSpot**: Contact management
- **Salesforce**: Lead tracking
- **Mock CRM**: Development and testing

### Knowledge Base
- **RAG System**: Semantic search over documents
- **FAQ Management**: Dynamic knowledge updates
- **Context-aware retrieval**: Conversation-aware responses

## 📊 Monitoring

### Health Checks
- `GET /health` - Service health status
- WebSocket connection monitoring
- Queue processing status

### Logging
- Structured JSON logging
- Request/response tracking
- Error monitoring and alerting

### Metrics
- Conversation throughput
- Response times
- Error rates
- Resource utilization

## 🚀 Deployment

### Docker

```bash
# Build images
docker build -f infra/docker/rt-gateway.Dockerfile -t clerk-rt-gateway .
docker build -f infra/docker/api.Dockerfile -t clerk-api .
docker build -f infra/docker/workers.Dockerfile -t clerk-workers .

# Run with docker-compose
docker-compose up -d
```

### AWS ECS

```bash
# Deploy infrastructure
cd infra/terraform
terraform init
terraform plan
terraform apply

# Deploy services
aws ecs update-service --cluster clerk-cluster --service rt-gateway --force-new-deployment
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the implementation plan

## 🔮 Roadmap

- [ ] Advanced voice cloning
- [ ] Multi-language support
- [ ] Video call integration
- [ ] Advanced analytics dashboard
- [ ] Mobile app support
- [ ] Enterprise SSO integration
