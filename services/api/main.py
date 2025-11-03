"""
Main API service for the AI Receptionist system.
Provides REST endpoints and WebSocket connections for the dashboard.
"""
import logging
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from shared.config import get_settings

from .dao import DynamoDBDAO, set_dao_instance
# from .routes import actions, conversations, rooms, meetings
# from .ws import ConnectionManager
from services.api.routes import actions, auth, rooms, meetings, api_keys, webhooks
from services.api.ws import ConnectionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Global services
connection_manager = ConnectionManager()
dao = DynamoDBDAO()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting API service...")

    try:
        # Initialize services
        await dao.initialize()
        await connection_manager.initialize()
        
        # Set the DAO instance for dependency injection
        set_dao_instance(dao)

        logger.info("API service initialized successfully")
        yield

    except Exception as e:
        logger.error(f"Failed to initialize API service: {e}")
        raise
    finally:
        logger.info("Shutting down API service...")


app = FastAPI(
    title="AI Receptionist API",
    description="REST API and WebSocket service for the AI Receptionist dashboard",
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

# Include routers
app.include_router(
    auth.router,
    prefix=f"{settings.api_prefix}/auth",
    tags=["authentication"],
)

app.include_router(
    actions.router, prefix=f"{settings.api_prefix}/actions", tags=["actions"]
)

app.include_router(rooms.router, prefix=f"{settings.api_prefix}/rooms", tags=["rooms"])

app.include_router(meetings.router, prefix=f"{settings.api_prefix}/meetings", tags=["meetings"])

app.include_router(
    api_keys.router,
    prefix=f"{settings.api_prefix}/api-keys",
    tags=["api-keys"],
)

app.include_router(
    webhooks.router,
    prefix="/v1/api.auray.net",
    tags=["webhooks"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "api",
        "version": settings.app_version,
        "active_connections": len(connection_manager.active_connections),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
