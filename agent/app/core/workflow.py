from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
from loguru import logger
import asyncio

class WorkflowStep(BaseModel):
    """Base model for workflow steps"""
    id: str
    type: str
    config: Dict[str, Any] = {}
    depends_on: Optional[List[str]] = None
    retry_count: int = 3
    retry_delay: int = 1

class WorkflowContext(BaseModel):
    """Context for workflow execution"""
    inputs: Dict[str, Any] = {}
    outputs: Dict[str, Any] = {}
    state: Dict[str, Any] = {}

class WorkflowDefinition(BaseModel):
    """Workflow definition model"""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    
class WorkflowEngine:
    """Core workflow engine"""
    def __init__(self):
        self._step_handlers = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default step handlers"""
        from app.core.executors import LLMExecutor

        # Register LLM executor
        self._step_handlers["llm"] = lambda config: LLMExecutor(**config)

    def register_handler(self, step_type: str, handler_factory):
        """Register a new step handler"""
        self._step_handlers[step_type] = handler_factory

    async def execute_step(self, step: WorkflowStep, context: WorkflowContext) -> Dict[str, Any]:
        """Execute a single workflow step"""
        try:
            if step.type not in self._step_handlers:
                raise ValueError(f"No handler registered for step type: {step.type}")

            # Create step handler
            handler = self._step_handlers[step.type](step.config)

            # Process dependencies
            if step.depends_on:
                for dep_id in step.depends_on:
                    if dep_id not in context.outputs:
                        raise ValueError(f"Dependency {dep_id} not satisfied for step {step.id}")

            # Execute step
            for attempt in range(step.retry_count):
                try:
                    result = await handler.execute(**context.inputs)
                    context.outputs[step.id] = result
                    return result
                except Exception as e:
                    if attempt == step.retry_count - 1:
                        raise
                    logger.warning(f"Step {step.id} failed, retrying... ({attempt + 1}/{step.retry_count})")
                    continue

        except Exception as e:
            logger.error(f"Error executing step {step.id}: {str(e)}")
            raise

    async def execute(self, workflow: WorkflowDefinition, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the entire workflow as DAG with parallel steps"""
        context = WorkflowContext(inputs=initial_context or {})
        pending = list(workflow.steps)
        results = {}
        try:
            while pending:
                # Find steps with satisfied dependencies
                ready = [step for step in pending if not step.depends_on or all(dep in results for dep in step.depends_on)]
                if not ready:
                    raise ValueError(f"Circular or unmet dependencies in workflow {workflow.id}")
                # Execute runnable steps concurrently
                tasks = {step.id: asyncio.create_task(self.execute_step(step, context)) for step in ready}
                for step_id, task in tasks.items():
                    res = await task
                    context.outputs[step_id] = res
                    results[step_id] = res
                    # Remove from pending list
                    pending = [s for s in pending if s.id != step_id]
            return {"status":"success","workflow_id":workflow.id,"outputs":context.outputs}
        except Exception as e:
            logger.error(f"Workflow {workflow.id} failed: {str(e)}")
            return {"status":"error","workflow_id":workflow.id,"error":str(e)}

class BaseWorkflow(ABC):
    """Base class for all workflows"""
    def __init__(self):
        self.engine = WorkflowEngine()

    @abstractmethod
    def get_definition(self) -> WorkflowDefinition:
        """Get the workflow definition"""
        pass

    async def execute(self, **inputs) -> Dict[str, Any]:
        """Execute the workflow"""
        definition = self.get_definition()
        return await self.engine.execute(definition, inputs)
