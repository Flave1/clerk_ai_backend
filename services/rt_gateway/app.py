"""
Real-time Gateway FastAPI application.
Handles STT, TTS, and LLM orchestration.

This file now uses modular routes from services.rt_gateway.routes.
For unified deployment, use services.unified_app instead.
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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.config import get_settings

from .routes import bot, conversations, llm, stt, tts
from .services import (
    active_bot_sessions,
    active_conversations,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP client logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Import here to avoid circular imports
    from .events import EventPublisher
    from .external_turn_manager import ExternalTurnManager
    from .llm import LLMService
    from .stt import STTService
    from .tts import TTSService
    from .turn_manager import TurnManager
    from . import services as rt_services
    
    logger.info("Starting RT Gateway services...")

    try:
        # Initialize services
        event_publisher = EventPublisher()
        await event_publisher.initialize()
        rt_services.event_publisher = event_publisher

        stt_service = STTService()
        rt_services.stt_service = stt_service
        
        tts_service = TTSService()
        rt_services.tts_service = tts_service
        
        llm_service = LLMService()
        rt_services.llm_service = llm_service
        
        turn_manager = TurnManager(llm_service, event_publisher, tts_service)
        rt_services.turn_manager = turn_manager

        if settings.use_external_turn_manager:
            external_turn_manager = ExternalTurnManager(llm_service)
            await external_turn_manager.initialize()
            rt_services.external_turn_manager = external_turn_manager
            logger.info("✅ External turn manager initialized")
        else:
            logger.info("ℹ️ External turn manager disabled")

        logger.info("RT Gateway services initialized successfully")
        yield

    except Exception as e:
        logger.error(f"Failed to initialize RT Gateway services: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down RT Gateway services...")
        if rt_services.external_turn_manager:
            await rt_services.external_turn_manager.cleanup()
        logger.info("Shutdown complete")


app = FastAPI(
    title="Aurray (CLERK) RT Gateway",
    description="Real-time audio/video processing and AI orchestration",
    version=settings.app_version,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "rt_gateway",
        "version": settings.app_version,
        "active_conversations": len(active_conversations),
        "active_bot_sessions": len(active_bot_sessions),
    }

# Include all route modules
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

app.include_router(
    bot.ws_router,
    prefix="/ws",
    tags=["websocket"],
)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.rt_gateway.app:app",
        host=settings.rt_gateway_host,
        port=settings.rt_gateway_port,
        reload=settings.debug,
    )
