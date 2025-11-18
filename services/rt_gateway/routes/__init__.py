"""RT Gateway routes package."""
from .bot import router as bot_router, ws_router as bot_ws_router
from .conversations import router as conversations_router
# from .llm import router as llm_router  # REMOVED: LLM service removed
from .stt import router as stt_router
from .tts import router as tts_router

# Export routers for unified_app
__all__ = [
    "bot",
    "conversations",
    # "llm",  # REMOVED: LLM service removed
    "stt",
    "tts",
]

# Export as module-level variables (for cleaner imports)
bot = type('bot', (), {
    'router': bot_router,
    'ws_router': bot_ws_router,
})()

conversations = type('conversations', (), {
    'router': conversations_router,
})()

# llm = type('llm', (), {
#     'router': llm_router,
# })()  # REMOVED: LLM service removed

stt = type('stt', (), {
    'router': stt_router,
})()

tts = type('tts', (), {
    'router': tts_router,
})()
