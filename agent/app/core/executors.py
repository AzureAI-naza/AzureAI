from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
import httpx
from loguru import logger
import json
import asyncio
import subprocess
import os
import tempfile
import shutil
from pathlib import Path
from app.core.config import settings
import traceback
from urllib.parse import urljoin

class ExecutorConfig(BaseModel):
    """Base configuration for all executors"""
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 1

class ExecutorResult(BaseModel):
    """Base result model for all executors"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseExecutor(ABC):
    """Base abstract class for all executors"""
    def __init__(self, config: Optional[ExecutorConfig] = None):
        self.config = config or ExecutorConfig()

    @abstractmethod
    async def execute(self, **kwargs) -> ExecutorResult:
        """Execute the executor's main logic"""
        pass

class LLMProvider(ABC):
    """Base class for LLM providers"""
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = self._create_client(**kwargs)

    @abstractmethod
    def _create_client(self, **kwargs) -> httpx.AsyncClient:
        """Create HTTP client for the provider"""
        pass

    @abstractmethod
    async def format_request(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Format request payload for the provider"""
        pass

    @abstractmethod
    async def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse response from the provider"""
        pass
    
    @abstractmethod
    def get_api_endpoint(self) -> str:
        """Get the API endpoint for the provider"""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    def _create_client(self, **kwargs) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=settings.OPENAI_API_URL,
            timeout=settings.LLM_API_TIMEOUT,
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
        )
        
    def get_api_endpoint(self) -> str:
        """Get the API endpoint for OpenAI"""
        return "/v1/chat/completions"

    async def format_request(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }

        # Add any additional parameters from kwargs
        for key, value in kwargs.items():
            if key not in ["temperature", "max_tokens"]:
                payload[key] = value

        return payload

    async def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "response": response["choices"][0]["message"]["content"],
            "usage": response.get("usage", {}),
            "finish_reason": response["choices"][0].get("finish_reason")
        }

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""
    def _create_client(self, **kwargs) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=settings.ANTHROPIC_API_URL,
            timeout=settings.LLM_API_TIMEOUT,
            headers={
                "x-api-key": settings.ANTHROPIC_API_KEY,
                "Content-Type": "application/json"
            }
        )
        
    def get_api_endpoint(self) -> str:
        """Get the API endpoint for Anthropic"""
        return "/v1/messages"

    async def format_request(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }

        return payload

    async def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "response": response["content"][0]["text"],
            "usage": response.get("usage", {}),
            "finish_reason": response.get("stop_reason")
        }

class GeminiProvider(LLMProvider):
    """Google Gemini API provider"""
    def _create_client(self, **kwargs) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=settings.GEMINI_API_URL,
            timeout=settings.LLM_API_TIMEOUT,
            headers={
                "Content-Type": "application/json"
            }
        )
        
    def get_api_endpoint(self) -> str:
        """Get the API endpoint for Gemini"""
        return f"/v1beta/models/{self.model}:generateContent?key={settings.GEMINI_API_KEY}"

    async def format_request(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # Create basic payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.temperature),
                "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens)
            }
        }
        
        # If system prompt is provided, add system_instruction
        if system_prompt:
            payload["system_instruction"] = {
                "parts": [
                    {"text": system_prompt}
                ]
            }

        return payload

    async def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "response": response["candidates"][0]["content"]["parts"][0]["text"],
            "usage": response.get("usageMetadata", {}),
            "finish_reason": response["candidates"][0].get("finishReason")
        }

class LLMProviderFactory:
    """Factory class for creating LLM providers"""
    @staticmethod
    def create_provider(model: str, **kwargs) -> LLMProvider:
        if model.startswith("gpt-"):
            return OpenAIProvider(model, **kwargs)
        elif model.startswith("claude-"):
            return AnthropicProvider(model, **kwargs)
        elif model.startswith("gemini-"):
            return GeminiProvider(model, **kwargs)
        else:
            raise ValueError(f"Unsupported model: {model}")

class LLMExecutor(BaseExecutor):
    """Executor for LLM-based operations supporting multiple providers"""
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        # Extract BaseExecutor config if provided
        config = kwargs.pop("config", None)
        # Extract default system_prompt from step config
        self.default_system_prompt = kwargs.pop("system_prompt", None)
        super().__init__(config=config)
        # Initialize provider without executor-specific kwargs
        self.provider = LLMProviderFactory.create_provider(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    async def execute(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ExecutorResult:
        """
        Execute LLM call using the appropriate provider
        
        Args:
            prompt: The main prompt to send to the LLM
            system_prompt: Optional system prompt to set the context
            **kwargs: Additional parameters to override defaults
        """
        try:
            # Implement retry logic
            for attempt in range(settings.LLM_API_MAX_RETRIES):
                try:
                    # Use provided or default system prompt
                    actual_system_prompt = system_prompt or self.default_system_prompt
                    payload = await self.provider.format_request(prompt, actual_system_prompt, **kwargs)
                    
                    # Check if base_url already contains API path
                    # If base_url already contains complete path, do not add additional path
                    endpoint = ""
                    
                    # Get current base_url
                    base_url = str(self.provider.client.base_url)
                    
                    # Use provider's get_api_endpoint method to get API path
                    api_endpoint = self.provider.get_api_endpoint()
                    if api_endpoint not in base_url:
                        endpoint = api_endpoint
                        
                    response = await self.provider.client.post(endpoint, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    
                    parsed_result = await self.provider.parse_response(result)
                    return ExecutorResult(
                        success=True,
                        data=parsed_result,
                        metadata={
                            "model": self.provider.model,
                            "attempt": attempt + 1
                        }
                    )
                except httpx.HTTPError as e:
                    if attempt == settings.LLM_API_MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(settings.LLM_API_RETRY_DELAY * (attempt + 1))
                    continue

        except Exception as e:
            logger.error(f"LLM API call failed: {str(e)}")
            logger.error(traceback.format_exc())
            return ExecutorResult(
                success=False,
                error=str(e),
                metadata={"model": self.provider.model}
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.provider.__aexit__(exc_type, exc_val, exc_tb)

class PythonCodeExecutor(BaseExecutor):
    """Executor for running Python code"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def execute(self, code: str, **kwargs) -> ExecutorResult:
        try:
            # TODO: Implement safe Python code execution
            # This is a placeholder for the actual implementation
            return ExecutorResult(
                success=True,
                data={"output": "Python execution result placeholder"}
            )
        except Exception as e:
            logger.error(f"Python code execution failed: {str(e)}")
            return ExecutorResult(success=False, error=str(e))

class HTTPExecutor(BaseExecutor):
    """Executor for making HTTP requests"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = httpx.AsyncClient(timeout=self.config.timeout)

    async def execute(self, url: str, method: str = "GET", **kwargs) -> ExecutorResult:
        try:
            response = await self.client.request(method, url, **kwargs)
            return ExecutorResult(
                success=True,
                data={
                    "status_code": response.status_code,
                    "content": response.text,
                    "headers": dict(response.headers)
                }
            )
        except Exception as e:
            logger.error(f"HTTP request failed: {str(e)}")
            return ExecutorResult(success=False, error=str(e))

class WebSearchExecutor(BaseExecutor):
    """Executor for web search operations"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def execute(self, query: str, **kwargs) -> ExecutorResult:
        try:
            # TODO: Implement web search functionality
            # This is a placeholder for the actual implementation
            return ExecutorResult(
                success=True,
                data={"search_results": "Search results placeholder"}
            )
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return ExecutorResult(success=False, error=str(e))

class BashExecutor(BaseExecutor):
    """Executor for running bash commands with stdin/stdout support"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.working_dir = kwargs.get("working_dir", os.getcwd())

    async def execute(
        self,
        command: str,
        stdin: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ExecutorResult:
        """
        Execute a bash command with optional stdin input
        
        Args:
            command: The bash command to execute
            stdin: Optional input to pass to the command via stdin
            env: Optional environment variables
            **kwargs: Additional parameters
        """
        try:
            # Prepare environment
            process_env = os.environ.copy()
            if env:
                process_env.update(env)

            # Create process
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE if stdin else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
                cwd=self.working_dir
            )

            # Execute with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(stdin.encode() if stdin else None),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return ExecutorResult(
                    success=False,
                    error="Command execution timed out",
                    metadata={"command": command}
                )

            # Check return code
            if process.returncode != 0:
                return ExecutorResult(
                    success=False,
                    error=stderr.decode(),
                    data={"return_code": process.returncode},
                    metadata={"command": command}
                )

            return ExecutorResult(
                success=True,
                data={
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "return_code": process.returncode
                },
                metadata={"command": command}
            )

        except Exception as e:
            logger.error(f"Bash execution failed: {str(e)}")
            return ExecutorResult(
                success=False,
                error=str(e),
                metadata={"command": command}
            )

class FileSystemExecutor(BaseExecutor):
    """Executor for file system operations"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_path = kwargs.get("base_path", os.getcwd())

    async def execute(
        self,
        operation: str,
        path: str,
        content: Optional[Union[str, bytes]] = None,
        **kwargs
    ) -> ExecutorResult:
        """
        Execute file system operations
        
        Args:
            operation: One of 'read', 'write', 'delete', 'list', 'exists'
            path: File or directory path
            content: Content to write (for write operation)
            **kwargs: Additional parameters
        """
        try:
            full_path = os.path.join(self.base_path, path)

            if operation == "read":
                mode = "rb" if kwargs.get("binary", False) else "r"
                with open(full_path, mode) as f:
                    data = f.read()
                return ExecutorResult(
                    success=True,
                    data={"content": data}
                )

            elif operation == "write":
                mode = "wb" if isinstance(content, bytes) else "w"
                with open(full_path, mode) as f:
                    f.write(content)
                return ExecutorResult(
                    success=True,
                    data={"path": full_path}
                )

            elif operation == "delete":
                if os.path.isfile(full_path):
                    os.remove(full_path)
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                return ExecutorResult(
                    success=True,
                    data={"path": full_path}
                )

            elif operation == "list":
                items = os.listdir(full_path)
                return ExecutorResult(
                    success=True,
                    data={"items": items}
                )

            elif operation == "exists":
                return ExecutorResult(
                    success=True,
                    data={"exists": os.path.exists(full_path)}
                )

            else:
                raise ValueError(f"Unknown operation: {operation}")

        except Exception as e:
            logger.error(f"File system operation failed: {str(e)}")
            return ExecutorResult(
                success=False,
                error=str(e),
                metadata={"operation": operation, "path": path}
            )

class DockerExecutor(BaseExecutor):
    """Executor for Docker operations"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.docker_command = kwargs.get("docker_command", "docker")

    async def execute(
        self,
        operation: str,
        image: Optional[str] = None,
        command: Optional[str] = None,
        **kwargs
    ) -> ExecutorResult:
        """
        Execute Docker operations
        
        Args:
            operation: One of 'run', 'build', 'pull', 'push', 'inspect'
            image: Docker image name
            command: Command to run in container
            **kwargs: Additional parameters
        """
        try:
            if operation == "run":
                cmd = f"{self.docker_command} run {kwargs.get('options', '')} {image} {command or ''}"
            elif operation == "build":
                cmd = f"{self.docker_command} build {kwargs.get('options', '')} -t {image} {kwargs.get('context', '.')}"
            elif operation == "pull":
                cmd = f"{self.docker_command} pull {image}"
            elif operation == "push":
                cmd = f"{self.docker_command} push {image}"
            elif operation == "inspect":
                cmd = f"{self.docker_command} inspect {image}"
            else:
                raise ValueError(f"Unknown operation: {operation}")

            # Use BashExecutor to run the command
            bash_executor = BashExecutor()
            result = await bash_executor.execute(cmd)

            if not result.success:
                return result

            return ExecutorResult(
                success=True,
                data={"output": result.data["stdout"]},
                metadata={"operation": operation, "image": image}
            )

        except Exception as e:
            logger.error(f"Docker operation failed: {str(e)}")
            return ExecutorResult(
                success=False,
                error=str(e),
                metadata={"operation": operation, "image": image}
            )

class GitExecutor(BaseExecutor):
    """Executor for Git operations"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.working_dir = kwargs.get("working_dir", os.getcwd())

    async def execute(
        self,
        operation: str,
        **kwargs
    ) -> ExecutorResult:
        """
        Execute Git operations
        
        Args:
            operation: One of 'clone', 'pull', 'push', 'commit', 'checkout'
            **kwargs: Additional parameters
        """
        try:
            if operation == "clone":
                cmd = f"git clone {kwargs.get('url')} {kwargs.get('path', '.')}"
            elif operation == "pull":
                cmd = f"git pull {kwargs.get('remote', 'origin')} {kwargs.get('branch', '')}"
            elif operation == "push":
                cmd = f"git push {kwargs.get('remote', 'origin')} {kwargs.get('branch', '')}"
            elif operation == "commit":
                cmd = f"git commit -m '{kwargs.get('message', '')}'" 
            elif operation == "checkout":
                cmd = f"git checkout {kwargs.get('branch')}"
            else:
                raise ValueError(f"Unknown operation: {operation}")

            # Use BashExecutor to run the command
            bash_executor = BashExecutor(working_dir=self.working_dir)
            result = await bash_executor.execute(cmd)

            if not result.success:
                return result

            return ExecutorResult(
                success=True,
                data={"output": result.data["stdout"]},
                metadata={"operation": operation}
            )

        except Exception as e:
            logger.error(f"Git operation failed: {str(e)}")
            return ExecutorResult(
                success=False,
                error=str(e),
                metadata={"operation": operation}
            ) 


class RAGExecutor(BaseExecutor):
    """Executor for RAG (Retrieval Augmented Generation) operations"""
    def __init__(self, **kwargs):
        # Extract custom parameters from kwargs
        api_key = kwargs.pop("api_key", "")
        knowledge_id = kwargs.pop("knowledge_id", "default")
        top_k = kwargs.pop("top_k", 5)
        score_threshold = kwargs.pop("score_threshold", 0.5)
        
        # Pass remaining parameters to parent class
        super().__init__(**kwargs)
        
        # Initialize custom attributes
        self.api_url = settings.RAG_API_URL
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        
        # Only add Authorization header if API key is not empty
        if api_key:
            print('API key: ', api_key)
            headers["Authorization"] = f"Bearer {api_key}"
        
        self.client = httpx.AsyncClient(
            timeout=kwargs.get("timeout", settings.LLM_API_TIMEOUT),
            headers=headers
        )
        self.knowledge_id = knowledge_id
        self.top_k = top_k
        self.score_threshold = score_threshold

    async def execute(
        self,
        query: str,
        **kwargs
    ) -> ExecutorResult:
        """
        Execute RAG query to retrieve relevant information
        
        Args:
            query: The query to send to the RAG system
            **kwargs: Additional parameters including:
                - knowledge_id: Override the default knowledge ID
                - top_k: Maximum number of results to return (default: 5)
                - score_threshold: Minimum relevance score (0-1) for results (default: 0.5)
                - metadata_condition: Optional filtering conditions
        """
        try:
            # Prepare the request payload
            payload = {
                "knowledge_id": kwargs.get("knowledge_id", self.knowledge_id),
                "query": query,
                "retrieval_setting": {
                    "top_k": kwargs.get("top_k", self.top_k),
                    "score_threshold": kwargs.get("score_threshold", self.score_threshold)
                }
            }
            
            # Add metadata conditions if provided
            if "metadata_condition" in kwargs:
                payload["metadata_condition"] = kwargs["metadata_condition"]
            
            # Construct the full URL for the retrieval endpoint
            retrieval_url = urljoin(self.api_url, "/retrieval")
            
            # Send the request to the RAG API
            response = await self.client.post(
                retrieval_url,
                json=payload
            )
            
            # Check for errors
            response.raise_for_status()
            response_data = response.json()
            
            # Return the results
            return ExecutorResult(
                success=True,
                data={
                    "records": response_data.get("records", []),
                    "total_records": len(response_data.get("records", []))
                },
                metadata={
                    "query": query,
                    "knowledge_id": payload["knowledge_id"],
                    "top_k": payload["retrieval_setting"]["top_k"],
                    "score_threshold": payload["retrieval_setting"]["score_threshold"]
                }
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"RAG API HTTP error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg = f"RAG API error: {error_data.get('error_msg', str(e))}"
            except:
                pass
            
            logger.error(error_msg)
            return ExecutorResult(
                success=False,
                error=error_msg,
                metadata={"query": query}
            )
            
        except Exception as e:
            logger.error(f"RAG operation failed: {str(e)}")
            return ExecutorResult(
                success=False,
                error=str(e),
                metadata={"query": query}
            )