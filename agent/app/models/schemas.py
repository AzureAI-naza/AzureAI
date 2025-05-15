from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class VerificationRequest(BaseModel):
    code: str
    options: Optional[Dict[str, Any]] = None

class VerificationResponse(BaseModel):
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None

class AgentWorkflowStep(BaseModel):
    name: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AgentWorkflow(BaseModel):
    id: str
    steps: List[AgentWorkflowStep]
    status: str
    created_at: str
    completed_at: Optional[str] = None 