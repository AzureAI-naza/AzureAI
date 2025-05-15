from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
from loguru import logger

from app.services.solana_program_verifier.agent import SolanaProgramVerifierAgent
from app.models.schemas import VerificationRequest, VerificationResponse

router = APIRouter()

@router.post("/verify/solana/program", response_model=VerificationResponse)
async def verify_solana_program_code(
    request: VerificationRequest,
    background_tasks: BackgroundTasks
) -> VerificationResponse:
    """
    Verify Solana smart contract code using LLM agent
    """
    try:
        agent = SolanaProgramVerifierAgent()
        result = await agent.verify_code(request.code)
        
        return VerificationResponse(
            success=True,
            message="Verification completed successfully",
            details=result
        )
    except Exception as e:
        logger.error(f"Error in Solana program verification: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        ) 
