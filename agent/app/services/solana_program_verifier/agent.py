from typing import Dict, Any
from loguru import logger
from app.core.workflow import BaseWorkflow, WorkflowDefinition, WorkflowStep

class SolanaProgramVerificationWorkflow(BaseWorkflow):
    """Workflow for Solana smart contract verification"""

    def get_definition(self) -> WorkflowDefinition:
        return WorkflowDefinition(
            id="solana_program_verifier",
            name="Solana Program Verifier",
            description="Verify Solana smart contract code for correctness and security",
            steps=[
                WorkflowStep(
                    id="syntax_validation",
                    type="llm",
                    config={
                        "model": "gpt-4o",
                        "system_prompt": (
                            "You are a Solana smart contract code syntax verification expert. Please check the syntax correctness of the following code:\n"
                            "* If the code syntax is correct, return `<r>True</r>`\n"
                            "* If there are syntax errors, return `<r>False</r><reasons><item>error1</item></reasons>`"
                        )
                    }
                ),
                WorkflowStep(
                    id="security_analysis",
                    type="llm",
                    config={
                        "model": "gpt-4o",
                        "system_prompt": (
                            "You are a Solana smart contract security analysis expert. Please check the security of the following code:\n"
                            "* If no security issues are found, return `<r>True</r>`\n"
                            "* If security issues are found, return `<r>False</r><issues><item>issue1 description</item></issues><suggestions><item>suggestion1</item></suggestions>`"
                        )
                    },
                    depends_on=["syntax_validation"]
                ),
                WorkflowStep(
                    id="best_practices",
                    type="llm",
                    config={
                        "model": "gpt-4o",
                        "system_prompt": (
                            "You are a Solana smart contract best practices expert. Please check if the following code follows best practices:\n"
                            "* If the code follows best practices, return `<r>True</r>`\n"
                            "* If there is room for improvement, return `<r>False</r><recommendations><item>recommendation1</item></recommendations>`"
                        )
                    },
                    depends_on=["syntax_validation"]
                )
            ]
        )

class SolanaProgramVerifierAgent:
    def __init__(self):
        logger.info("Initialize Solana Program Verifier Agent")
        self.workflow = SolanaProgramVerificationWorkflow()

    async def verify_code(self, code: str) -> Dict[str, Any]:
        """
        Verify Solana smart contract code using a series of LLM-powered steps
        
        Args:
            code: The Solana smart contract code to verify
            
        Returns:
            Dict containing the verification results
        """
        try:
            # Build input data
            inputs = {
                "prompt": code
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
            return {
                "status": "success",
                "results": {
                    "syntax": outputs["syntax_validation"].data["response"],
                    "security": outputs["security_analysis"].data["response"],
                    "best_practices": outputs["best_practices"].data["response"]
                }
            }

        except Exception as e:
            logger.error(f"Error in Solana program verification: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
