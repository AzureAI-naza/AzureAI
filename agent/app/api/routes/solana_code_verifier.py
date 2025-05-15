from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
from loguru import logger

from app.services.solana_code_verifier.agent import SolanaCodeVerifierAgent
from app.models.schemas import VerificationRequest, VerificationResponse

router = APIRouter()

@router.post("/verify/solana/code", response_model=VerificationResponse)
async def verify_solana_web3_code(
    request: VerificationRequest,
    background_tasks: BackgroundTasks
) -> VerificationResponse:
    """
    Verify Solana web3.js code using RAG and LLM
    """
    try:
        agent = SolanaCodeVerifierAgent()
        result = await agent.verify_code(request.code)
        
        return VerificationResponse(
            success=True,
            message="Success",
            details=result
        )
    except Exception as e:
        logger.error(f"Error in Solana code verification: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed: {str(e)}"
        ) 
