from mcp.server.fastmcp import FastMCP, Context
from typing import Dict, Any, List, Optional
import logging
import os
import uvicorn
import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server.sse import SseServerTransport

# Configure logging
if os.getenv("DEBUG", "0") == "1":
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Agent Service URL
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://agent:8000")

class AzureAIServer:
    def __init__(self):
        self.mcp = FastMCP("AzureAI Solana Code Verifier")
        self.name = "AzureAI Solana Code Verifier"
        self.agent_service_url = AGENT_SERVICE_URL
        self._setup_tools()

    def _setup_tools(self):
        @self.mcp.tool()
        async def verify_solana_contract(ctx: Context, code: str) -> Dict[str, Any]:
            """Verify Solana smart contract code"""
            logger.debug("Starting Solana smart contract code verification")
            
            try:
                # Call Agent Service's Solana program verification API
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.agent_service_url}/api/v1/verify/solana-program",
                        json={"code": code}
                    )
                    
                    if response.status_code != 200:
                        logger.error(f"Agent Service returned error: {response.status_code} - {response.text}")
                        return {
                            "status": "error",
                            "message": f"Agent Service returned error: {response.status_code}"
                        }
                    
                    result = response.json()
                    logger.debug(f"Solana smart contract verification result: {result}")
                    return result
            except Exception as e:
                logger.error(f"Error occurred while verifying Solana smart contract: {str(e)}")
                return {
                    "status": "error",
                    "message": str(e)
                }
        
        @self.mcp.tool()
        async def verify_solana_web3_code(ctx: Context, code: str) -> Dict[str, Any]:
            """Verify Solana Web3.js client code"""
            logger.debug("Starting Solana Web3.js code verification")
            
            try:
                # Call Agent Service's Solana code verification API
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.agent_service_url}/api/v1/verify/solana-code",
                        json={"code": code}
                    )
                    
                    if response.status_code != 200:
                        logger.error(f"Agent Service returned error: {response.status_code} - {response.text}")
                        return {
                            "status": "error",
                            "message": f"Agent Service returned error: {response.status_code}"
                        }
                    
                    result = response.json()
                    logger.debug(f"Solana Web3.js code verification result: {result}")
                    return result
            except Exception as e:
                logger.error(f"Error occurred while verifying Solana Web3.js code: {str(e)}")
                return {
                    "status": "error",
                    "message": str(e)
                }
        
        @self.mcp.tool()
        async def verify_typescript_code(ctx: Context, code: str) -> Dict[str, Any]:
            """Verify TypeScript code"""
            logger.debug("Starting TypeScript code verification")
            
            try:
                # Call Agent Service's TypeScript verification API
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.agent_service_url}/api/v1/verify/typescript",
                        json={"code": code}
                    )
                    
                    if response.status_code != 200:
                        logger.error(f"Agent Service returned error: {response.status_code} - {response.text}")
                        return {
                            "status": "error",
                            "message": f"Agent Service returned error: {response.status_code}"
                        }
                    
                    result = response.json()
                    logger.debug(f"TypeScript code verification result: {result}")
                    return result
            except Exception as e:
                logger.error(f"Error occurred while verifying TypeScript code: {str(e)}")
                return {
                    "status": "error",
                    "message": str(e)
                }
        
        @self.mcp.tool()
        async def analyze_code(ctx: Context, code: str, code_type: str) -> Dict[str, Any]:
            """Analyze code based on code type"""
            logger.debug(f"Starting code analysis, type: {code_type}")
            
            try:
                if code_type.lower() in ["solana", "rust", "contract", "smart contract"]:
                    return await verify_solana_contract(ctx, code)
                elif code_type.lower() in ["web3", "javascript", "js"]:
                    return await verify_solana_web3_code(ctx, code)
                elif code_type.lower() in ["typescript", "ts"]:
                    return await verify_typescript_code(ctx, code)
                else:
                    return {
                        "status": "error",
                        "message": f"Unsupported code type: {code_type}, please use 'solana'/'rust', 'web3'/'javascript', or 'typescript'"
                    }
            except Exception as e:
                logger.error(f"Error occurred while analyzing code: {str(e)}")
                return {
                    "status": "error",
                    "message": str(e)
                }

    def create_starlette_app(self, debug: bool = False) -> Starlette:
        """Create a Starlette application that provides MCP service via SSE"""
        sse = SseServerTransport("/messages/")
        mcp_server = self.mcp._mcp_server

        async def handle_sse(request: Request) -> None:
            async with sse.connect_sse(
                    request.scope,
                    request.receive,
                    request._send,  # noqa: SLF001
            ) as (read_stream, write_stream):
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                )

        return Starlette(
            debug=debug,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

    def run(self, host: str = "0.0.0.0", port: int = 8080, debug: bool = False):
        """Run MCP server as HTTP SSE server"""
        starlette_app = self.create_starlette_app(debug=debug)
        logger.info(f"Starting AzureAI Code Verifier MCP SSE server at: {host}:{port}")
        logger.info(f"Connecting to Agent Service: {self.agent_service_url}")
        uvicorn.run(starlette_app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run AzureAI Code Verifier MCP SSE server')
    parser.add_argument('--host', default='0.0.0.0', help='Service binding address')
    parser.add_argument('--port', type=int, default=8080, help='Service listening port')
    parser.add_argument('--agent-url', default=AGENT_SERVICE_URL, help='Agent Service URL')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    app = AzureAIServer()
    if args.agent_url != AGENT_SERVICE_URL:
        app.agent_service_url = args.agent_url
    app.run(host=args.host, port=args.port, debug=args.debug)