# Module for OpenAI client management and API call logic

import asyncio
import time
from typing import Callable, List, Optional, TypeVar

import tiktoken
from agents import Agent, ModelSettings, Runner, set_default_openai_client
from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ankigen_core.logging import logger
from ankigen_core.models import Card, CardBack, CardFront, CrawledPage
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
        model: Model name (e.g., "gpt-5.1", "gpt-5.1-chat-latest")
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

    # 3. Build model settings with GPT-5.1 reasoning support
    model_settings_kwargs: dict = {"temperature": temperature}

    # GPT-5.1 (not chat-latest) supports reasoning_effort
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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RETRYABLE_OPENAI_ERRORS),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying OpenAI call (attempt {retry_state.attempt_number}) for process_crawled_page due to {retry_state.outcome.exception() if retry_state.outcome else 'unknown reason'}"
    ),
)
async def process_crawled_page(
    openai_client: AsyncOpenAI,
    page: CrawledPage,
    model: str = "gpt-4o",
    custom_system_prompt: Optional[str] = None,
    custom_user_prompt_template: Optional[str] = None,
    max_prompt_content_tokens: int = 6000,
    cache: Optional[ResponseCache] = None,
) -> List[Card]:
    """Process a crawled page and extract structured Card objects using OpenAI.

    Args:
        openai_client: The OpenAI client instance
        page: The crawled page to process
        model: The model to use for generation
        custom_system_prompt: Optional custom system prompt
        custom_user_prompt_template: Optional custom user prompt template
        max_prompt_content_tokens: Maximum tokens for content
        cache: Optional ResponseCache for page-level caching

    Returns:
        List of generated Card objects
    """
    # Check page-level cache first
    if cache:
        cache_key = f"{page.url}:{model}"
        cached_cards = cache.get(cache_key, "page_cache")
        if cached_cards is not None:
            logger.info(f"Using cached cards for page: {page.url}")
            return cached_cards

    logger.info(
        f"Processing page: {page.url} with model {model}, max_prompt_content_tokens: {max_prompt_content_tokens}"
    )

    if not page.text_content or not page.text_content.strip():
        logger.info(f"Skipping page {page.url} as it has empty text content.")
        return []

    system_prompt = (
        custom_system_prompt
        if custom_system_prompt and custom_system_prompt.strip()
        else """
You are an expert Anki card creator. Your task is to generate Anki flashcards from the provided web page content.
For each card, provide:
- "front": A dictionary with a "question" field.
- "back": A dictionary with "answer", "explanation", and "example" fields.
- "tags": A list of relevant keywords (optional).
- "source_url": The URL of the page the content was extracted from (this will be provided by the system).
- "note_type": Specify "Basic" for question/answer cards or "Cloze" for cloze deletion cards. (This will be mapped to "card_type").
- "metadata": An optional dictionary for additional structured information such as:
    - "prerequisites": ["list", "of", "prerequisites"]
    - "learning_outcomes": ["list", "of", "learning", "outcomes"]
    - "common_misconceptions": ["list", "of", "common", "misconceptions"]
    - "difficulty": "beginner" | "intermediate" | "advanced"
    - "topic": "The main topic this card relates to, derived from the content"

Focus on creating clear, concise, and accurate cards that are useful for learning.
If generating cloze cards, ensure the "front.question" field uses Anki's cloze syntax, e.g., "The capital of {{c1::France}} is Paris."
Ensure the entire response is a valid JSON object following this structure:
{
  "cards": [
    {
      "front": {"question": "..."},
      "back": {"answer": "...", "explanation": "...", "example": "..."},
      "tags": ["...", "..."],
      "card_type": "Basic",
      "metadata": {"difficulty": "beginner", "prerequisites": [], "topic": "..."}
    },
    // ... more cards
  ]
}
"""
    )

    # User Prompt
    default_user_prompt_template = """
Please generate Anki cards based on the following content from the URL: {url}

Content:
{content}

Generate a few high-quality Anki cards from this content.
"""
    user_prompt: str
    if custom_user_prompt_template and custom_user_prompt_template.strip():
        try:
            user_prompt = custom_user_prompt_template.format(
                url=page.url, content=page.text_content
            )
        except KeyError as e:
            logger.warning(
                f"Custom user prompt template for {page.url} is malformed (missing key {e}). Falling back to default."
            )
            user_prompt = default_user_prompt_template.format(
                url=page.url, content=page.text_content
            )
    else:
        user_prompt = default_user_prompt_template.format(
            url=page.url, content=page.text_content
        )
    # --- End Prompt Definition ---

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(
            f"Tiktoken model {model} not found, using cl100k_base for token estimation and truncation."
        )
        encoding = tiktoken.get_encoding("cl100k_base")

    prompt_structure_tokens = len(encoding.encode(system_prompt + user_prompt))
    available_tokens_for_content = max_prompt_content_tokens - prompt_structure_tokens
    if available_tokens_for_content <= 0:
        logger.error(
            f"Max prompt tokens ({max_prompt_content_tokens}) too small for prompt structure for page {page.url}. Cannot process."
        )
        return []

    page_content_for_prompt = page.text_content or ""
    content_tokens = encoding.encode(page_content_for_prompt)
    if len(content_tokens) > available_tokens_for_content:
        truncated_content_tokens = content_tokens[:available_tokens_for_content]
        page_content_for_prompt = encoding.decode(truncated_content_tokens)
        logger.warning(
            f"Content for page {page.url} was truncated from {len(content_tokens)} tokens "
            f"to {len(truncated_content_tokens)} tokens to fit model's context window (limit: {max_prompt_content_tokens} for content portion)."
        )

    estimated_request_tokens = prompt_structure_tokens + len(
        encoding.encode(page_content_for_prompt)
    )
    await openai_rate_limiter.wait_if_needed(estimated_request_tokens)

    try:
        logger.debug(
            f"Attempting to generate cards for {page.url} using model {model}."
        )

        # Use agents SDK for structured output
        result = await structured_agent_call(
            openai_client=openai_client,
            model=model,
            instructions=system_prompt,
            user_input=user_prompt,
            output_type=GenericJsonOutput,  # Flexible schema for card generation
            temperature=0.5,
            timeout=120.0,
        )

        if result is None:
            logger.error(f"Invalid or empty response from agent for page {page.url}.")
            return []

        # Convert Pydantic model to dict for processing
        parsed_cards = result.model_dump() if isinstance(result, BaseModel) else result

        validated_cards: List[Card] = []

        cards_list_from_json = []
        if (
            isinstance(parsed_cards, dict)
            and "cards" in parsed_cards
            and isinstance(parsed_cards["cards"], list)
        ):
            cards_list_from_json = parsed_cards["cards"]
            logger.info(
                f"Found 'cards' key in response from {page.url} with {len(cards_list_from_json)} cards"
            )
        elif isinstance(parsed_cards, list):
            cards_list_from_json = parsed_cards
        else:
            logger.error(
                f"LLM response for {page.url} was not a list or valid dict. Response: {str(parsed_cards)[:200]}..."
            )
            return []

        for card_dict in cards_list_from_json:
            if not isinstance(card_dict, dict):
                logger.warning(
                    f"Skipping non-dict card item for {page.url}: {card_dict}"
                )
                continue

            try:
                front_data = card_dict.get("front")
                back_data = card_dict.get("back")

                if not isinstance(front_data, dict) or "question" not in front_data:
                    logger.warning(
                        f"Malformed 'front' data in card_dict for {page.url}: {front_data}. Skipping card."
                    )
                    continue
                if not isinstance(back_data, dict) or "answer" not in back_data:
                    logger.warning(
                        f"Malformed 'back' data in card_dict for {page.url}: {back_data}. Skipping card."
                    )
                    continue

                metadata_payload = card_dict.get("metadata", {})
                if not isinstance(metadata_payload, dict):
                    metadata_payload = {}
                metadata_payload["source_url"] = page.url
                if page.title and "topic" not in metadata_payload:
                    metadata_payload["topic"] = page.title

                tags = card_dict.get("tags", [])
                if not isinstance(tags, list) or not all(
                    isinstance(t, str) for t in tags
                ):
                    tags = []

                if tags:
                    metadata_payload["tags"] = tags

                card_obj = Card(
                    front=CardFront(question=str(front_data["question"])),
                    back=CardBack(
                        answer=str(back_data["answer"]),
                        explanation=str(back_data.get("explanation", "")),
                        example=str(back_data.get("example", "")),
                    ),
                    card_type=str(card_dict.get("card_type", "Basic")),
                    metadata=metadata_payload,
                )
                validated_cards.append(card_obj)
            except Exception as e:
                logger.error(
                    f"Error creating Card object for {page.url} from dict: {card_dict}. Error: {e}",
                    exc_info=True,
                )

        if not validated_cards:
            logger.info(
                f"No valid Cards generated or parsed from {page.url} after LLM processing."
            )
        else:
            logger.info(
                f"Successfully generated {len(validated_cards)} Cards from {page.url}."
            )
            # Cache successful results for page-level caching
            if cache:
                cache_key = f"{page.url}:{model}"
                cache.set(cache_key, "page_cache", validated_cards)
                logger.debug(f"Cached {len(validated_cards)} cards for {page.url}")

        return validated_cards

    except Exception as e:
        logger.error(
            f"Error processing page {page.url} with agents SDK: {e}", exc_info=True
        )
        return []


async def process_crawled_pages(
    openai_client: AsyncOpenAI,
    pages: List[CrawledPage],
    model: str = "gpt-4o",
    max_prompt_content_tokens: int = 6000,
    max_concurrent_requests: int = 5,
    custom_system_prompt: Optional[str] = None,
    custom_user_prompt_template: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cache: Optional[ResponseCache] = None,
) -> List[Card]:
    if not pages:
        logger.info("No pages provided to process_crawled_pages.")
        return []

    logger.info(
        f"Starting batch processing of {len(pages)} pages with model {model}. Max concurrent requests: {max_concurrent_requests}."
    )

    semaphore = asyncio.Semaphore(max_concurrent_requests)
    tasks = []
    processed_count = 0

    async def process_with_semaphore(page: CrawledPage):
        nonlocal processed_count
        async with semaphore:
            logger.debug(
                f"Submitting task for page: {page.url} (Semaphore count: {semaphore._value})"
            )
            try:
                page_cards = await process_crawled_page(
                    openai_client=openai_client,
                    page=page,
                    model=model,
                    custom_system_prompt=custom_system_prompt,
                    custom_user_prompt_template=custom_user_prompt_template,
                    max_prompt_content_tokens=max_prompt_content_tokens,
                    cache=cache,
                )
                if page_cards is None:
                    logger.warning(
                        f"process_crawled_page returned None for {page.url}, expected list. Defaulting to empty list."
                    )
                    page_cards = []

                logger.info(
                    f"Completed processing for page: {page.url}. Generated {len(page_cards)} cards."
                )
                return page_cards
            except Exception as e:
                logger.error(
                    f"Error in process_with_semaphore for page {page.url}: {e}",
                    exc_info=True,
                )
                return []
            finally:
                processed_count += 1
                if progress_callback:
                    progress_callback(processed_count, len(pages))

    for page_to_process in pages:
        tasks.append(asyncio.create_task(process_with_semaphore(page_to_process)))

    results_from_tasks: List[List[Card]] = []
    for i, future in enumerate(asyncio.as_completed(tasks)):
        try:
            result_list = await future
            if result_list:
                results_from_tasks.append(result_list)
        except Exception as e:
            logger.error(
                f"Unhandled error gathering result for a page task: {e}", exc_info=True
            )

    all_cards: List[Card] = []
    for card_list in results_from_tasks:
        all_cards.extend(card_list)

    logger.info(
        f"Finished processing all {len(pages)} pages. Generated {len(all_cards)} Cards in total."
    )
    return all_cards
