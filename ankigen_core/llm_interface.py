# Module for OpenAI client management and API call logic

import asyncio
import time
from typing import Optional, TypeVar

from agents import Agent, ModelSettings, Runner, set_default_openai_client
from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel

from ankigen_core.logging import logger
from ankigen_core.utils import ResponseCache

T = TypeVar("T", bound=BaseModel)


class OpenAIClientManager:
    """Manages the AsyncOpenAI client instance."""

    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None
        self._api_key: Optional[str] = None

    async def initialize_client(self, api_key: str):
        """Initializes the AsyncOpenAI client with the given API key."""
        if not api_key or not api_key.startswith("sk-"):
            logger.error("Invalid OpenAI API key provided for client initialization.")
            raise ValueError("Invalid OpenAI API key format.")
        self._api_key = api_key
        try:
            self._client = AsyncOpenAI(api_key=self._api_key)
            logger.info("AsyncOpenAI client initialized successfully.")
        except OpenAIError as e:  # Catch specific OpenAI errors
            logger.error(f"Failed to initialize AsyncOpenAI client: {e}", exc_info=True)
            self._client = None  # Ensure client is None on failure
            raise  # Re-raise the OpenAIError to be caught by UI
        except Exception as e:  # Catch any other unexpected errors
            logger.error(
                f"An unexpected error occurred during AsyncOpenAI client initialization: {e}",
                exc_info=True,
            )
            self._client = None
            raise RuntimeError("Unexpected error initializing AsyncOpenAI client.")

    def get_client(self) -> AsyncOpenAI:
        """Returns the initialized AsyncOpenAI client. Raises error if not initialized."""
        if self._client is None:
            logger.error(
                "AsyncOpenAI client accessed before initialization or after a failed initialization."
            )
            raise RuntimeError(
                "AsyncOpenAI client is not initialized. Please provide a valid API key."
            )
        return self._client

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        await self.aclose()
        return False

    def close(self) -> None:
        """Close the OpenAI client synchronously."""
        if self._client:
            try:
                # OpenAI client has a close method for cleanup
                if hasattr(self._client, "close"):
                    self._client.close()
                logger.debug("OpenAI client closed")
            except Exception as e:
                logger.warning(f"Error closing OpenAI client: {e}")
            finally:
                self._client = None

    async def aclose(self) -> None:
        """Close the OpenAI client asynchronously."""
        if self._client:
            try:
                # OpenAI async client has an aclose method
                if hasattr(self._client, "aclose"):
                    await self._client.aclose()
                elif hasattr(self._client, "close"):
                    self._client.close()
                logger.debug("OpenAI client closed (async)")
            except Exception as e:
                logger.warning(f"Error closing OpenAI client: {e}")
            finally:
                self._client = None


# --- Agents SDK Utility ---


async def structured_agent_call(
    openai_client: AsyncOpenAI,
    model: str,
    instructions: str,
    user_input: str,
    output_type: type[T],
    cache: Optional[ResponseCache] = None,
    cache_key: Optional[str] = None,
    temperature: float = 0.7,
    timeout: float = 120.0,
    retry_attempts: int = 3,
) -> T:
    """
    Make a single-turn structured output call using the agents SDK.

    This is a lightweight wrapper for simple structured output calls,
    not intended for complex multi-agent workflows.

    Args:
        openai_client: AsyncOpenAI client instance
        model: Model name (e.g., "gpt-5.2", "gpt-5.2-chat-latest")
        instructions: System instructions for the agent
        user_input: User prompt/input
        output_type: Pydantic model class for structured output
        cache: Optional ResponseCache instance
        cache_key: Cache key (required if cache is provided)
        temperature: Model temperature (default 0.7)
        timeout: Request timeout in seconds (default 120)
        retry_attempts: Number of retry attempts (default 3)

    Returns:
        Instance of output_type with the structured response
    """
    # 1. Check cache first
    if cache and cache_key:
        cached = cache.get(cache_key, model)
        if cached is not None:
            logger.info(f"Using cached response for model {model}")
            # Reconstruct Pydantic model from cached dict
            if isinstance(cached, dict):
                return output_type.model_validate(cached)
            return cached

    # 2. Set up the OpenAI client for agents SDK
    set_default_openai_client(openai_client, use_for_tracing=False)

    # 3. Build model settings with GPT-5.x reasoning support
    model_settings_kwargs: dict = {"temperature": temperature}

    # GPT-5.x (not chat-latest) supports reasoning_effort
    if model.startswith("gpt-5") and "chat-latest" not in model:
        from openai.types.shared import Reasoning

        model_settings_kwargs["reasoning"] = Reasoning(effort="none")

    model_settings = ModelSettings(**model_settings_kwargs)

    # 4. Create agent with structured output
    agent = Agent(
        name="structured_output_agent",
        instructions=instructions,
        model=model,
        model_settings=model_settings,
        output_type=output_type,
    )

    # 5. Execute with retry and timeout
    last_error: Optional[Exception] = None
    for attempt in range(retry_attempts):
        try:
            result = await asyncio.wait_for(
                Runner.run(agent, user_input),
                timeout=timeout,
            )

            # 6. Extract structured output
            output = result.final_output

            # 7. Cache successful result (as dict for serialization)
            if cache and cache_key and output is not None:
                if isinstance(output, BaseModel):
                    cache.set(cache_key, model, output.model_dump())
                else:
                    cache.set(cache_key, model, output)

            logger.debug(f"Successfully received response from model {model}")
            return output

        except asyncio.TimeoutError as e:
            last_error = e
            if attempt < retry_attempts - 1:
                wait_time = 4 * (2**attempt)  # Exponential backoff
                logger.warning(
                    f"Agent timed out (attempt {attempt + 1}/{retry_attempts}), "
                    f"retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
                continue
            logger.error(f"Agent timed out after {retry_attempts} attempts")
            raise
        except Exception as e:
            last_error = e
            if attempt < retry_attempts - 1:
                wait_time = 4 * (2**attempt)
                logger.warning(
                    f"Agent failed (attempt {attempt + 1}/{retry_attempts}): {e}, "
                    f"retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
                continue
            logger.error(f"Agent failed after {retry_attempts} attempts: {e}")
            raise

    raise RuntimeError(f"Retry loop exited without result: {last_error}")


# Generic schema for arbitrary JSON structured outputs
class GenericJsonOutput(BaseModel):
    """Generic container for JSON output - allows any structure."""

    model_config = {"extra": "allow"}  # Allow arbitrary fields


async def structured_output_completion(
    openai_client: AsyncOpenAI,
    model: str,
    response_format: dict,  # Legacy parameter - kept for API compatibility
    system_prompt: str,
    user_prompt: str,
    cache: ResponseCache,
) -> Optional[dict]:
    """
    Makes an API call with structured output using agents SDK.

    Note: response_format parameter is ignored - the agents SDK handles
    JSON parsing automatically. For typed outputs, use structured_agent_call() directly.
    """
    cache_key = f"{system_prompt}:{user_prompt}"

    # Ensure system_prompt includes JSON instruction
    effective_system_prompt = system_prompt
    if "JSON object matching the specified schema" not in system_prompt:
        effective_system_prompt = f"{system_prompt}\nProvide your response as a JSON object matching the specified schema."

    try:
        result = await structured_agent_call(
            openai_client=openai_client,
            model=model,
            instructions=effective_system_prompt.strip(),
            user_input=user_prompt.strip(),
            output_type=GenericJsonOutput,
            cache=cache,
            cache_key=cache_key,
            temperature=0.7,
        )

        # Convert Pydantic model back to dict for backward compatibility
        if isinstance(result, BaseModel):
            return result.model_dump()
        return result

    except Exception as e:
        logger.error(
            f"structured_output_completion failed for model {model}: {e}",
            exc_info=True,
        )
        raise  # Re-raise unexpected errors


# Specific OpenAI exceptions to retry on
RETRYABLE_OPENAI_ERRORS = (
    APIConnectionError,
    RateLimitError,
    APIStatusError,  # Typically for 5xx server errors
)

# --- New OpenAIRateLimiter Class (Subtask 9.2) ---


class OpenAIRateLimiter:
    """Manages token usage to proactively stay within (estimated) OpenAI rate limits."""

    def __init__(self, tokens_per_minute: int = 60000):  # Default, can be configured
        self.tokens_per_minute_limit: int = tokens_per_minute
        self.tokens_used_current_window: int = 0
        self.current_window_start_time: float = time.monotonic()

    async def wait_if_needed(self, estimated_tokens_for_request: int):
        """Waits if adding the estimated tokens would exceed the rate limit for the current window."""
        current_time = time.monotonic()

        # Check if the 60-second window has passed
        if current_time - self.current_window_start_time >= 60.0:
            # Reset window and token count
            self.current_window_start_time = current_time
            self.tokens_used_current_window = 0
            logger.debug("OpenAIRateLimiter: Window reset.")

        # Check if the request would exceed the limit in the current window
        if (
            self.tokens_used_current_window + estimated_tokens_for_request
            > self.tokens_per_minute_limit
        ):
            time_to_wait = (self.current_window_start_time + 60.0) - current_time
            if time_to_wait > 0:
                logger.info(
                    f"OpenAIRateLimiter: Approaching token limit. Waiting for {time_to_wait:.2f} seconds to reset window."
                )
                await asyncio.sleep(time_to_wait)
            # After waiting for the window to reset, reset counters
            self.current_window_start_time = time.monotonic()  # New window starts now
            self.tokens_used_current_window = 0
            logger.debug("OpenAIRateLimiter: Window reset after waiting.")

        # If we are here, it's safe to proceed (or we've waited and reset)
        # Add tokens for the current request
        self.tokens_used_current_window += estimated_tokens_for_request
        logger.debug(
            f"OpenAIRateLimiter: Tokens used in current window: {self.tokens_used_current_window}/{self.tokens_per_minute_limit}"
        )


# Global instance of the rate limiter
# This assumes a single rate limit bucket for all calls from this application instance.
# More sophisticated scenarios might need per-model or per-key limiters.
openai_rate_limiter = OpenAIRateLimiter()  # Using default 60k TPM for now
