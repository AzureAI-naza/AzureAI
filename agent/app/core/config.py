from typing import List
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LLM Agent Service"
    
    # CORS Configuration
    CORS_ORIGINS: List[AnyHttpUrl] = ["http://localhost:3000"]
    
    # OpenAI API Configuration
    OPENAI_API_URL: str = "https://api.openai.com"
    OPENAI_API_KEY: str = ""
    
    # Anthropic API Configuration
    ANTHROPIC_API_URL: str = "https://api.anthropic.com"
    ANTHROPIC_API_KEY: str = ""
    
    # Google Gemini API Configuration
    GEMINI_API_URL: str = "https://generativelanguage.googleapis.com"
    GEMINI_API_KEY: str = "AIzaSyDXGe11cKy6J42xhHv5Tm0rGHQHLhanmrc"
    
    # RAG System Configuration
    RAG_API_URL: str = "http://127.0.0.1:6000"
    RAG_API_KEY: str = "your-api-key"
    
    # Common LLM API Configuration
    LLM_API_TIMEOUT: int = 60
    LLM_API_MAX_RETRIES: int = 3
    LLM_API_RETRY_DELAY: int = 1
    
    # Service Configuration
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 