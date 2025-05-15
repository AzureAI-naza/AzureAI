from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.api.routes import solana_program_verifier, solana_code_verifier, typescript_verifier

app = FastAPI(
    title="LLM Agent Service",
    description="A service that provides various LLM agent capabilities",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings  .CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(solana_program_verifier.router, prefix="/api/v1", tags=["solana-program-verifier"])
app.include_router(solana_code_verifier.router, prefix="/api/v1", tags=["solana-code-verifier"])
app.include_router(typescript_verifier.router, prefix="/api/v1", tags=["typescript-verifier"])

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up AzureAI Backend Service...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AzureAI Backend Service...")

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 