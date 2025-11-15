"""
Speech-to-Text routes for RT Gateway.
"""
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from shared.schemas import STTRequest

from ..services import stt_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    conversation_id: str = Form(...),
    turn_number: int = Form(default=1)
):
    """Transcribe audio to text."""
    try:
        audio_data = await audio.read()
        
        request = STTRequest(
            audio_data=audio_data,
            conversation_id=conversation_id,
            turn_number=turn_number
        )
        
        response = await stt_service.transcribe(request)
        return response
    except Exception as e:
        logger.error(f"STT transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

