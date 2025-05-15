from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
from loguru import logger

from app.services.typescript_verifier.agent import TypeScriptVerifierAgent
from app.models.schemas import VerificationRequest, VerificationResponse

router = APIRouter()

@router.post("/verify/typescript", response_model=VerificationResponse)
async def verify_typescript_code(
    request: VerificationRequest,
    background_tasks: BackgroundTasks
) -> VerificationResponse:
    """
    Verify TypeScript code using LLM agent
    """
    try:
        agent = TypeScriptVerifierAgent()
        result = await agent.verify_code(request.code)
        
        return VerificationResponse(
            success=True,
            message="Verification completed successfully",
            details=result
        )
    except Exception as e:
        logger.error(f"Error in TypeScript verification: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        ) 