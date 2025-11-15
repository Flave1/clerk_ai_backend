"""
Unified API and RT Gateway service.
Combines both services into a single FastAPI application on one port.
"""
import logging
import sys
from contextlib import asynccontextmanager

# Python 3.12 compatibility fix for pydantic v1 ForwardRef._evaluate()
if sys.version_info >= (3, 12):
    import pydantic.v1.typing as pydantic_v1_typing
    original_evaluate = pydantic_v1_typing.evaluate_forwardref
    def patched_evaluate_forwardref(type_, globalns, localns):
        return type_._evaluate(globalns, localns, set(), recursive_guard=set())
    pydantic_v1_typing.evaluate_forwardref = patched_evaluate_forwardref

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from shared.config import get_settings

# API imports
from services.api.dao import DynamoDBDAO, set_dao_instance
from services.api.routes import actions, auth, rooms, meetings, api_keys, webhooks, integrations, meeting_contexts, newsletter
from services.api.ws import ConnectionManager

# RT Gateway imports
from services.rt_gateway.services import (
    active_bot_sessions,
    active_conversations,
    event_publisher,
    external_turn_manager,
    llm_service,
    stt_service,
    tts_service,
    turn_manager,
)

from services.rt_gateway.routes import (
    bot,
    conversations,
    llm,
    stt,
    tts,
)

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S.%f'
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP client logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

settings = get_settings()

# API global services
connection_manager = ConnectionManager()
dao = DynamoDBDAO()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Unified application lifespan manager."""
    global turn_manager, external_turn_manager, stt_service, tts_service
    global llm_service, event_publisher, connection_manager, dao

    logger.info("Starting unified API and RT Gateway services...")

    try:
        # Initialize API services first (lighter)
        logger.info("Initializing API services...")
        await dao.initialize()
        await connection_manager.initialize()
        set_dao_instance(dao)
        logger.info("✅ API services initialized")

        # Initialize RT Gateway services (heavier)
        logger.info("Initializing RT Gateway services...")
        
        # Import here to avoid circular imports
        from services.rt_gateway.events import EventPublisher
        from services.rt_gateway.external_turn_manager import ExternalTurnManager
        from services.rt_gateway.llm import LLMService
        from services.rt_gateway.stt import STTService
        from services.rt_gateway.tts import TTSService
        from services.rt_gateway.turn_manager import TurnManager
        from services.rt_gateway.state_manager import get_state_manager
        from services.rt_gateway import services as rt_services
        
        # Initialize state manager (Redis-based distributed state)
        logger.info("Initializing state manager...")
        state_manager = await get_state_manager()
        rt_services.state_manager = state_manager
        logger.info("✅ State manager initialized")
        
        event_publisher = EventPublisher()
        await event_publisher.initialize()
        rt_services.event_publisher = event_publisher

        # Initialize STT service (gracefully handle failures)
        try:
            logger.info("Initializing STT service...")
            stt_service = STTService()
            rt_services.stt_service = stt_service
            logger.info(f"STT service object created: {stt_service}")
            # Check if any STT clients were initialized
            has_provider = (stt_service.deepgram_client or stt_service.openai_realtime_client or 
                           stt_service.aws_transcribe or stt_service.elevenlabs_client)
            if has_provider:
                logger.info("✅ STT service initialized with at least one provider")
            else:
                logger.warning("⚠️ STT service initialized but no providers available (no API keys configured)")
                logger.warning("   STT service object exists but detect_speech_activity will still work")
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize STT service: {e}", exc_info=True)
            logger.warning("STT service will not be available")
            rt_services.stt_service = None
            stt_service = None  # Ensure local variable is also None
        
        # Initialize TTS service (gracefully handle failures)
        tts_service = None
        try:
            tts_service = TTSService()
            rt_services.tts_service = tts_service
            logger.info("✅ TTS service initialized")
            # Check if any providers are available
            has_provider = (
                tts_service.elevenlabs_client is not None or
                tts_service.aws_polly is not None or
                settings.openai_api_key is not None
            )
            if has_provider:
                logger.info("✅ TTS service has at least one provider available")
            else:
                logger.warning("⚠️ TTS service object created but no providers available (OpenAI/ElevenLabs/AWS keys may be missing)")
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize TTS service: {e}", exc_info=True)
            logger.warning("Creating fallback TTS service object...")
            # Create a minimal service object even if initialization fails
            try:
                tts_service = TTSService.__new__(TTSService)
                tts_service.elevenlabs_client = None
                tts_service.aws_polly = None
                rt_services.tts_service = tts_service
                logger.warning("⚠️ TTS service fallback object created (will use OpenAI if key is available)")
            except Exception as fallback_error:
                logger.error(f"❌ Failed to create fallback TTS service: {fallback_error}")
                rt_services.tts_service = None
        
        # Initialize LLM service (gracefully handle failures)
        llm_service = None
        try:
            # Pass initialized DAO to LLMService
            llm_service = LLMService(dao=dao)
            rt_services.llm_service = llm_service
            logger.info("✅ LLM service initialized")
            # Log service state for debugging
            if llm_service.llm is None:
                logger.warning("⚠️ LLM service object created but LLM client is None (OpenAI API key may be missing)")
            else:
                logger.info("✅ LLM client initialized successfully")
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize LLM service: {e}", exc_info=True)
            logger.warning("Creating fallback LLM service object...")
            # Create a minimal service object even if initialization fails
            try:
                llm_service = LLMService.__new__(LLMService)
                llm_service.llm = None
                llm_service.tools = []
                llm_service.graph = None
                rt_services.llm_service = llm_service
                logger.warning("⚠️ LLM service fallback object created (limited functionality)")
            except Exception as fallback_error:
                logger.error(f"❌ Failed to create fallback LLM service: {fallback_error}")
                rt_services.llm_service = None
        
        try:
            # Pass initialized DAO to TurnManager
            turn_manager = TurnManager(llm_service, event_publisher, tts_service, dao=dao)
            rt_services.turn_manager = turn_manager
            logger.info("✅ Turn manager initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize TurnManager: {e}", exc_info=True)
            rt_services.turn_manager = None
            raise

        if settings.use_external_turn_manager:
            external_turn_manager = ExternalTurnManager(llm_service)
            await external_turn_manager.initialize()
            rt_services.external_turn_manager = external_turn_manager
            logger.info("✅ External turn manager initialized")
        else:
            logger.info("ℹ️ External turn manager disabled")

        logger.info("✅ RT Gateway services initialized")
        logger.info("✅ Unified service initialized successfully")
        yield

    except Exception as e:
        logger.error(f"Failed to initialize unified service: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down unified services...")
        if external_turn_manager:
            await external_turn_manager.cleanup()
        # Cleanup state manager
        if rt_services.state_manager:
            await rt_services.state_manager.cleanup()
        logger.info("Shutdown complete")

app = FastAPI(
    title="Aurray Unified Service",
    description="Combined REST API, WebSocket, and Real-time Gateway service",
    version=settings.app_version,
    lifespan=lifespan,
)

# ✅ Get CORS origins from settings
cors_origins = settings.get_cors_origins()
logger.info(f"CORS allowed origins: {cors_origins}")

# ✅ Add proxy header middleware to handle HTTPS behind CloudFront
class ProxyHeaderMiddleware(BaseHTTPMiddleware):
    """Middleware to handle X-Forwarded-Proto header for HTTPS redirects."""
    async def dispatch(self, request: Request, call_next):
        # Check if we're behind a proxy (CloudFront)
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
        if forwarded_proto == "https":
            # Force HTTPS scheme for redirects
            request.scope["scheme"] = "https"
        elif request.headers.get("X-Forwarded-For"):
            # If we have X-Forwarded-For, assume HTTPS (CloudFront always uses HTTPS)
            request.scope["scheme"] = "https"
        
        response = await call_next(request)
        return response

app.add_middleware(ProxyHeaderMiddleware)

# ✅ Add CORS middleware
# Note: CORSMiddleware automatically handles OPTIONS preflight requests
# Ensure localhost:3000 is always included for local development
final_cors_origins = list(cors_origins)  # Make a copy
if "http://localhost:3000" not in final_cors_origins:
    final_cors_origins.append("http://localhost:3000")
if "http://127.0.0.1:3000" not in final_cors_origins:
    final_cors_origins.append("http://127.0.0.1:3000")

logger.info(f"Final CORS allowed origins: {final_cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=final_cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Accept",
        "Origin",
        "X-Requested-With",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
    ],
    expose_headers=["*"],
)

# ============================================================================
# UNIFIED HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Unified health check endpoint."""
    rt_gateway_status = "healthy"
    return {
        "status": "healthy",
        "service": "unified",
        "version": settings.app_version,
        "api": {
            "status": "healthy",
            "active_connections": len(connection_manager.active_connections),
        },
        "rt_gateway": {
            "status": rt_gateway_status,
            "active_conversations": len(active_conversations),
            "active_bot_sessions": len(active_bot_sessions),
        }
    }

# ============================================================================
# API ROUTES
# ============================================================================

app.include_router(
    auth.router,
    prefix=f"{settings.api_prefix}/auth",
    tags=["authentication"],
)

app.include_router(
    actions.router,
    prefix=f"{settings.api_prefix}/actions",
    tags=["actions"],
)

app.include_router(
    rooms.router,
    prefix=f"{settings.api_prefix}/rooms",
    tags=["rooms"],
)

app.include_router(
    meetings.router,
    prefix=f"{settings.api_prefix}/meetings",
    tags=["meetings"],
)

app.include_router(
    api_keys.router,
    prefix=f"{settings.api_prefix}/api-keys",
    tags=["api-keys"],
)

app.include_router(
    webhooks.router,
    prefix=f"{settings.api_prefix}/ws",
    tags=["webhooks"],
)

app.include_router(
    integrations.router,
    prefix=f"{settings.api_prefix}/integrations",
    tags=["integrations"],
)

app.include_router(
    meeting_contexts.router,
    prefix=f"{settings.api_prefix}/meeting-contexts",
    tags=["meeting-contexts"],
)

app.include_router(
    newsletter.router,
    prefix=f"{settings.api_prefix}/newsletter",
    tags=["newsletter"],
)

# ============================================================================
# RT GATEWAY ROUTES
# ============================================================================

app.include_router(
    conversations.router,
    prefix="/conversations",
    tags=["conversations"],
)

app.include_router(
    stt.router,
    prefix="/stt",
    tags=["stt"],
)

app.include_router(
    tts.router,
    prefix="/tts",
    tags=["tts"],
)

app.include_router(
    llm.router,
    prefix="/llm",
    tags=["llm"],
)

# Bot HTTP routes
app.include_router(
    bot.router,
    prefix="/bot",
    tags=["bot"],
)

# WebSocket routes (conversation WebSocket, bot audio streams)
app.include_router(
    bot.ws_router,
    prefix="/ws",
    tags=["websocket"],
)


# ============================================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.unified_app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
