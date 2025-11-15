"""
LLM routes for RT Gateway.
"""
import logging
from fastapi import APIRouter, HTTPException

from shared.schemas import LLMRequest, LLMResponse

from ..services import llm_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate", response_model=LLMResponse)
async def generate_response(request: LLMRequest) -> LLMResponse:
    """Generate LLM response."""
    try:
        response = await llm_service.generate_response(request)
        return response
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

