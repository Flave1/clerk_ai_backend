"""
Text-to-Speech routes for RT Gateway.
"""
import logging
from fastapi import APIRouter, HTTPException

from shared.schemas import TTSRequest, TTSResponse

from ..services import tts_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest) -> TTSResponse:
    """Synthesize text to speech."""
    try:
        response = await tts_service.synthesize(request)
        return response
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

