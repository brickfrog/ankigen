# Base agent wrapper and configuration classes

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pydantic import BaseModel
import asyncio
import json
from openai import AsyncOpenAI
from agents import Agent, Runner, ModelSettings

from ankigen_core.logging import logger
from .token_tracker import track_usage_from_agents_sdk


def parse_agent_json_response(response: Any) -> Dict[str, Any]:
    """Parse agent response, handling markdown code blocks if present"""
    if isinstance(response, str):
        # Strip markdown code blocks
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]  # Remove ```json
        if response.startswith("```"):
            response = response[3:]  # Remove ```
        if response.endswith("```"):
            response = response[:-3]  # Remove trailing ```
        response = response.strip()

        return json.loads(response)
    else:
        return response


@dataclass
class AgentConfig:
    """Configuration for individual agents"""

    name: str
    instructions: str
    model: str = "gpt-5.1"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: float = 30.0
    retry_attempts: int = 3
    enable_tracing: bool = True
    custom_prompts: Optional[Dict[str, str]] = None
    output_type: Optional[type] = None  # For structured outputs

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

    async def initialize(self):
        """Initialize the OpenAI agent with structured output support"""
        try:
            # Set the default OpenAI client for the agents SDK
            from agents import set_default_openai_client

            set_default_openai_client(self.openai_client, use_for_tracing=False)

            # Create model settings with temperature and GPT-5.1 reasoning support
            model_settings_kwargs = {"temperature": self.config.temperature}

            # GPT-5.1 (not chat-latest) supports reasoning_effort
            if (
                self.config.model.startswith("gpt-5")
                and "chat-latest" not in self.config.model
            ):
                from openai.types.shared import Reasoning

                model_settings_kwargs["reasoning"] = Reasoning(effort="none")

            model_settings = ModelSettings(**model_settings_kwargs)

            # Use clean instructions without JSON formatting hacks
            clean_instructions = self.config.instructions

            # Create agent with structured output if output_type is provided
            if self.config.output_type:
                self.agent = Agent(
                    name=self.config.name,
                    instructions=clean_instructions,
                    model=self.config.model,
                    model_settings=model_settings,
                    output_type=self.config.output_type,
                )
                logger.info(
                    f"Initialized agent with structured output: {self.config.name} -> {self.config.output_type}"
                )
            else:
                self.agent = Agent(
                    name=self.config.name,
                    instructions=clean_instructions,
                    model=self.config.model,
                    model_settings=model_settings,
                )
                logger.info(
                    f"Initialized agent (no structured output): {self.config.name}"
                )

        except Exception as e:
            logger.error(f"Failed to initialize agent {self.config.name}: {e}")
            raise

    def _enhance_input_with_context(
        self, user_input: str, context: Optional[Dict[str, Any]]
    ) -> str:
        """Add context to user input if provided."""
        if context is None:
            return user_input
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
        return f"{user_input}\n\nContext:\n{context_str}"

    async def _execute_with_retry(self, enhanced_input: str) -> Any:
        """Execute agent with retry logic on timeout."""
        for attempt in range(self.config.retry_attempts):
            try:
                result = await asyncio.wait_for(
                    Runner.run(
                        starting_agent=self.agent,
                        input=enhanced_input,
                    ),
                    timeout=self.config.timeout,
                )
                return result
            except asyncio.TimeoutError:
                if attempt < self.config.retry_attempts - 1:
                    logger.warning(
                        f"Agent {self.config.name} timed out "
                        f"(attempt {attempt + 1}/{self.config.retry_attempts}), retrying..."
                    )
                    continue
                logger.error(
                    f"Agent {self.config.name} timed out after {self.config.retry_attempts} attempts"
                )
                raise
        raise RuntimeError("Retry loop exited without result")

    def _extract_and_track_usage(self, result: Any) -> Dict[str, Any]:
        """Extract usage info from result and track it."""
        total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }

        if hasattr(result, "raw_responses") and result.raw_responses:
            for response in result.raw_responses:
                if hasattr(response, "usage") and response.usage:
                    total_usage["input_tokens"] += response.usage.input_tokens
                    total_usage["output_tokens"] += response.usage.output_tokens
                    total_usage["total_tokens"] += response.usage.total_tokens
                    total_usage["requests"] += response.usage.requests

            track_usage_from_agents_sdk(total_usage, self.config.model)
            logger.info(f"Agent usage: {total_usage}")

        return total_usage

    def _extract_output(self, result: Any) -> Any:
        """Extract final output from agent result."""
        if not (hasattr(result, "new_items") and result.new_items):
            return str(result)

        from agents.items import ItemHelpers

        text_output = ItemHelpers.text_message_outputs(result.new_items)

        if self.config.output_type and self.config.output_type is not str:
            logger.info(
                f"Structured output: {type(text_output)} -> {self.config.output_type}"
            )

        return text_output

    async def execute(
        self, user_input: str, context: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, Dict[str, Any]]:
        """Execute the agent with user input and optional context."""
        if not self.agent:
            await self.initialize()

        if self.agent is None:
            raise ValueError("Agent not initialized")

        enhanced_input = self._enhance_input_with_context(user_input, context)

        logger.info(f"Executing agent: {self.config.name}")
        logger.info(f"Input: {enhanced_input[:200]}...")

        import time

        start_time = time.time()

        try:
            result = await self._execute_with_retry(enhanced_input)
            execution_time = time.time() - start_time
            logger.info(f"Agent {self.config.name} executed in {execution_time:.2f}s")

            total_usage = self._extract_and_track_usage(result)
            output = self._extract_output(result)

            return output, total_usage

        except asyncio.TimeoutError:
            logger.error(
                f"Agent {self.config.name} timed out after {self.config.timeout}s"
            )
            raise
        except Exception as e:
            logger.error(f"Agent {self.config.name} execution failed: {e}")
            raise

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
    metadata: Dict[str, Any] = {}
    errors: List[str] = []
