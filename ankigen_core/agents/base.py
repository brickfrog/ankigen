# Base agent wrapper and configuration classes

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pydantic import BaseModel
import asyncio
import time
from openai import AsyncOpenAI
from agents import Agent, Runner

from ankigen_core.logging import logger


@dataclass
class AgentConfig:
    """Configuration for individual agents"""

    name: str
    instructions: str
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: float = 30.0
    retry_attempts: int = 3
    enable_tracing: bool = True
    custom_prompts: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.custom_prompts is None:
            self.custom_prompts = {}


class BaseAgentWrapper:
    """Base wrapper for OpenAI Agents SDK integration"""

    def __init__(self, config: AgentConfig, openai_client: AsyncOpenAI):
        self.config = config
        self.openai_client = openai_client
        self.agent = None
        self.runner = None
        self._performance_metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "average_response_time": 0.0,
            "error_count": 0,
        }

    async def initialize(self):
        """Initialize the OpenAI agent"""
        try:
            self.agent = Agent(
                name=self.config.name,
                instructions=self.config.instructions,
                model=self.config.model,
                temperature=self.config.temperature,
            )

            # Initialize runner with the OpenAI client
            self.runner = Runner(
                agent=self.agent,
                client=self.openai_client,
            )

            logger.info(f"Initialized agent: {self.config.name}")

        except Exception as e:
            logger.error(f"Failed to initialize agent {self.config.name}: {e}")
            raise

    async def execute(self, user_input: str, context: Dict[str, Any] = None) -> Any:
        """Execute the agent with user input and optional context"""
        if not self.runner:
            await self.initialize()

        start_time = time.time()
        self._performance_metrics["total_calls"] += 1

        try:
            # Add context to the user input if provided
            enhanced_input = user_input
            if context is not None:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
                enhanced_input = f"{user_input}\n\nContext:\n{context_str}"

            # Execute the agent
            result = await asyncio.wait_for(
                self._run_agent(enhanced_input), timeout=self.config.timeout
            )

            # Update metrics
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time, success=True)

            logger.debug(
                f"Agent {self.config.name} executed successfully in {response_time:.2f}s"
            )
            return result

        except asyncio.TimeoutError:
            self._performance_metrics["error_count"] += 1
            logger.error(
                f"Agent {self.config.name} timed out after {self.config.timeout}s"
            )
            raise
        except Exception as e:
            self._performance_metrics["error_count"] += 1
            logger.error(f"Agent {self.config.name} execution failed: {e}")
            raise

    async def _run_agent(self, input_text: str) -> Any:
        """Run the agent with retry logic"""
        last_exception = None

        for attempt in range(self.config.retry_attempts):
            try:
                # Create a new run
                run = await self.runner.create_run(
                    messages=[{"role": "user", "content": input_text}]
                )

                # Wait for completion
                while run.status in ["queued", "in_progress"]:
                    await asyncio.sleep(0.1)
                    run = await self.runner.get_run(run.id)

                if run.status == "completed":
                    # Get the final message
                    messages = await self.runner.get_messages(run.thread_id)
                    if messages and messages[-1].role == "assistant":
                        return messages[-1].content
                    else:
                        raise ValueError("No assistant response found")
                else:
                    raise ValueError(f"Run failed with status: {run.status}")

            except Exception as e:
                last_exception = e
                if attempt < self.config.retry_attempts - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Agent {self.config.name} attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Agent {self.config.name} failed after {self.config.retry_attempts} attempts"
                    )

        raise last_exception

    def _update_performance_metrics(self, response_time: float, success: bool):
        """Update performance metrics"""
        if success:
            self._performance_metrics["successful_calls"] += 1

        # Update average response time
        total_successful = self._performance_metrics["successful_calls"]
        if total_successful > 0:
            current_avg = self._performance_metrics["average_response_time"]
            self._performance_metrics["average_response_time"] = (
                current_avg * (total_successful - 1) + response_time
            ) / total_successful

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent"""
        return {
            **self._performance_metrics,
            "success_rate": (
                self._performance_metrics["successful_calls"]
                / max(1, self._performance_metrics["total_calls"])
            ),
            "agent_name": self.config.name,
        }

    async def handoff_to(
        self, target_agent: "BaseAgentWrapper", context: Dict[str, Any]
    ) -> Any:
        """Hand off execution to another agent with context"""
        logger.info(
            f"Handing off from {self.config.name} to {target_agent.config.name}"
        )

        # Prepare handoff context
        handoff_context = {
            "from_agent": self.config.name,
            "handoff_reason": context.get("reason", "Standard workflow handoff"),
            **context,
        }

        # Execute the target agent
        return await target_agent.execute(
            context.get("user_input", "Continue processing"), handoff_context
        )


class AgentResponse(BaseModel):
    """Standard response format for agents"""

    success: bool
    data: Any
    agent_name: str
    execution_time: float
    metadata: Dict[str, Any] = {}
    errors: List[str] = []
