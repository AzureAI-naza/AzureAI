from typing import Dict, Any, List
from loguru import logger
from app.core.workflow import BaseWorkflow, WorkflowDefinition, WorkflowStep
from app.core.executors import RAGExecutor, LLMExecutor
from app.core.config import settings

class SolanaCodeVerifierWorkflow(BaseWorkflow):
    """Workflow for Solana code verification using RAG and LLM"""

    def __init__(self):
        super().__init__()
        # Register RAG executor
        self.engine.register_handler("rag", lambda config: RAGExecutor(**config.get("kwargs", {})))

    def get_definition(self) -> WorkflowDefinition:
        return WorkflowDefinition(
            id="solana_code_verifier",
            name="Solana Code Verifier",
            description="Verify Solana web3.js code using RAG and LLM",
            steps=[
                WorkflowStep(
                    id="knowledge_retrieval",
                    type="rag",
                    config={
                        "kwargs": {
                            "api_key": settings.RAG_API_KEY,
                            "knowledge_id": "solana",
                            "top_k": 5,
                            "score_threshold": 0.5
                        }
                    }
                ),
                WorkflowStep(
                    id="code_verification",
                    type="llm",
                    config={
                        "model": "gemini-2.5-pro-preview-05-06",
                        "temperature": 0.2,
                        "max_tokens": 2000,
                        "system_prompt_template": """
Use the following context as your learned knowledge, inside <context></context> XML tags.

<context>
{context}
</context>

When answer to user:
- If you don't know, just say that you don't know.
- If you don't know when you are not sure, ask for clarification.
Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.

<role>
You are a helper to review Solana web3.js code, helping users review the API usage in their code to see if it matches the documentation definition, outputting the API calls that do not match the definition and the correct API signature, and outputting the API calls that do not exist and clearly indicating that they do not exist.
</role>
<rules>
* If the definition is correct, do not output, **only output the incorrect definition**, the incorrect definition content is wrapped in `<items></items>`, each item's format is `<apiName>API_NAME</apiName><existed>True</existed><correctSignature>CORRECT_SIG</correctSignature><userArguments>USER_ARGS</userArguments><sigDiff>SIGNATURE_DIFF</sigDiff>`, if the API does not exist, then it is `<apiName>API_NAME</apiName><existed>False</existed>`, and finally output `<r>FAILED</r>`
* If all definitions are correct, just reply `<items></items><r>PASS</r>`
* If the user outputs other content, just reply `<items></items><r>NOT_HANDLED</r>`
</rules>
"""
                    },
                    depends_on=["knowledge_retrieval"]
                )
            ]
        )

class SolanaCodeVerifierAgent:
    def __init__(self):
        logger.info("初始化 Solana Code Verifier Agent")
        self.workflow = SolanaCodeVerifierWorkflow()

    async def verify_code(self, code: str) -> Dict[str, Any]:
        """
        Verify Solana web3.js code using RAG and LLM
        
        Args:
            code: The Solana web3.js code to verify
            
        Returns:
            Dict containing the verification results
        """
        try:
            # Build input data
            inputs = {
                "query": code,  # Used for RAG query
                "prompt": code   # Used for LLM prompt
            }

            # Execute workflow
            result = await self.workflow.execute(**inputs)

            if result["status"] == "error":
                return {
                    "status": "error",
                    "message": result["error"]
                }

            # Process workflow output
            outputs = result["outputs"]
            
            # Get RAG retrieval result
            rag_result = outputs["knowledge_retrieval"]
            
            # Prepare LLM input, inject RAG result into system prompt
            knowledge_context = ""
            if rag_result.success and "records" in rag_result.data:
                for record in rag_result.data["records"]:
                    knowledge_context += f"{record.get('content', '')}\n"
            
            # Get LLM verification result
            llm_result = outputs["code_verification"]
            
            return {
                "status": "success",
                "results": {
                    "verification": llm_result.data["response"],
                    "knowledge_retrieved": len(rag_result.data.get("records", [])) if rag_result.success else 0
                }
            }

        except Exception as e:
            logger.error(f"Error in Solana code verification: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
