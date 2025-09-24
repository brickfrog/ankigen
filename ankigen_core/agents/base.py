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
    model: str = "gpt-4.1"
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
            # Create model settings with temperature
            model_settings = ModelSettings(temperature=self.config.temperature)

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

    async def execute(
        self, user_input: str, context: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, Dict[str, Any]]:
        """Execute the agent with user input and optional context"""
        if not self.agent:
            await self.initialize()

        # Add context to the user input if provided
        enhanced_input = user_input
        if context is not None:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            enhanced_input = f"{user_input}\n\nContext:\n{context_str}"

        # Execute the agent using Runner.run() with retry logic
        if self.agent is None:
            raise ValueError("Agent not initialized")

        logger.info(f"🤖 EXECUTING AGENT: {self.config.name}")
        logger.info(f"📝 INPUT: {enhanced_input[:200]}...")

        import time

        start_time = time.time()

        for attempt in range(self.config.retry_attempts):
            try:
                result = await asyncio.wait_for(
                    Runner.run(
                        starting_agent=self.agent,
                        input=enhanced_input,
                    ),
                    timeout=self.config.timeout,
                )
                break
            except asyncio.TimeoutError:
                if attempt < self.config.retry_attempts - 1:
                    logger.warning(
                        f"Agent {self.config.name} timed out (attempt {attempt + 1}/{self.config.retry_attempts}), retrying..."
                    )
                    continue
                else:
                    logger.error(
                        f"Agent {self.config.name} timed out after {self.config.retry_attempts} attempts"
                    )
                    raise

        try:
            execution_time = time.time() - start_time
            logger.info(
                f"Agent {self.config.name} executed successfully in {execution_time:.2f}s"
            )

            # Extract usage information from raw_responses
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

                # Track usage with the token tracker
                track_usage_from_agents_sdk(total_usage, self.config.model)
                logger.info(f"💰 AGENT USAGE: {total_usage}")

            # Extract the final output from the result
            if hasattr(result, "new_items") and result.new_items:
                # Get the last message content
                from agents.items import ItemHelpers

                text_output = ItemHelpers.text_message_outputs(result.new_items)

                # If we have structured output, the response should already be parsed
                if self.config.output_type and self.config.output_type is not str:
                    logger.info(
                        f"✅ STRUCTURED OUTPUT: {type(text_output)} -> {self.config.output_type}"
                    )
                    # The agents SDK should return the structured object directly
                    return text_output, total_usage
                else:
                    return text_output, total_usage
            else:
                return str(result), total_usage

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
