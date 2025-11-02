# Module for OpenAI client management and API call logic

from openai import (
    AsyncOpenAI,
    OpenAIError,
    APIConnectionError,  # For more specific retry
    RateLimitError,  # For more specific retry
    APIStatusError,  # For retry on 5xx errors
)  # Added OpenAIError for specific exception handling
import json
import time  # Added for process_crawled_pages later, but good to have
from typing import List, Optional, Callable  # Added List, Optional, Callable
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import asyncio  # Import asyncio for gather
import tiktoken  # Added tiktoken

# Imports from our new core modules
from ankigen_core.logging import logger  # Updated to use the new logger
from ankigen_core.utils import ResponseCache  # Removed get_logger
from ankigen_core.models import (
    CrawledPage,
    Card,
    CardFront,
    CardBack,
)  # Added CrawledPage, Card, CardFront, CardBack
# We will need Pydantic models if response_format is a Pydantic model,
# but for now, it's a dict like {"type": "json_object"}.
# from ankigen_core.models import ... # Placeholder if needed later

# logger = get_logger() # Removed, using imported logger


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


# Retry decorator for API calls - kept similar to original
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        Exception
    ),  # Consider refining this to specific network/API errors
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying structured_output_completion (attempt {retry_state.attempt_number}) due to {retry_state.outcome.exception() if retry_state.outcome else 'unknown reason'}"
    ),
)
async def structured_output_completion(
    openai_client: AsyncOpenAI,  # Expecting an initialized AsyncOpenAI client instance
    model: str,
    response_format: dict,  # e.g., {"type": "json_object"}
    system_prompt: str,
    user_prompt: str,
    cache: ResponseCache,  # Expecting a ResponseCache instance
):
    """Makes an API call to OpenAI with structured output, retry logic, and caching."""

    # Use the passed-in cache instance
    cached_response = cache.get(f"{system_prompt}:{user_prompt}", model)
    if cached_response is not None:
        logger.info(f"Using cached response for model {model}")
        return cached_response  # Return cached value directly, not as a coroutine

    try:
        logger.debug(f"Making API call to OpenAI model {model}")

        # Ensure system_prompt includes JSON instruction if response_format is json_object
        # This was previously done before calling this function, but good to ensure here too.
        effective_system_prompt = system_prompt
        if (
            response_format.get("type") == "json_object"
            and "JSON object matching the specified schema" not in system_prompt
        ):
            effective_system_prompt = f"{system_prompt}\nProvide your response as a JSON object matching the specified schema."

        # Security: Add timeout to prevent indefinite hanging
        completion = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": effective_system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            response_format=response_format,  # Pass the dict directly
            temperature=0.7,  # Consider making this configurable
            timeout=120.0,  # 120 second timeout
        )

        if not hasattr(completion, "choices") or not completion.choices:
            logger.warning(
                f"No choices returned in OpenAI completion for model {model}."
            )
            return None  # Or raise an error

        first_choice = completion.choices[0]
        if (
            not hasattr(first_choice, "message")
            or first_choice.message is None
            or first_choice.message.content is None
        ):
            logger.warning(
                f"No message content in the first choice for OpenAI model {model}."
            )
            return None  # Or raise an error

        # Parse the JSON response
        result = json.loads(first_choice.message.content)

        # Cache the successful response using the passed-in cache instance
        cache.set(f"{system_prompt}:{user_prompt}", model, result)
        logger.debug(f"Successfully received and parsed response from model {model}")
        return result

    except OpenAIError as e:  # More specific error handling
        logger.error(f"OpenAI API call failed for model {model}: {e}", exc_info=True)
        raise  # Re-raise to be handled by the calling function, potentially as gr.Error
    except json.JSONDecodeError as e:
        # Accessing first_choice might be an issue if completion itself failed before choices
        # However, structure assumes choices are checked before this json.loads typically
        # For safety, check if first_choice.message.content is available
        response_content_for_log = "<unavailable>"
        if (
            "first_choice" in locals()
            and first_choice.message
            and first_choice.message.content
        ):
            response_content_for_log = first_choice.message.content[:500]
        logger.error(
            f"Failed to parse JSON response from model {model}: {e}. Response: {response_content_for_log}",
            exc_info=True,
        )
        raise ValueError(
            f"Invalid JSON response from AI model {model}."
        )  # Raise specific error
    except Exception as e:
        logger.error(
            f"Unexpected error during structured_output_completion for model {model}: {e}",
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
        response_format_param = {"type": "json_object"}
        # Security: Add timeout to prevent indefinite hanging
        response_data = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format_param,
            temperature=0.5,
            timeout=120.0,  # 120 second timeout
        )

        if (
            not response_data.choices
            or not response_data.choices[0].message
            or not response_data.choices[0].message.content
        ):
            logger.error(f"Invalid or empty response from OpenAI for page {page.url}.")
            return []

        cards_json_str = response_data.choices[0].message.content
        parsed_cards = json.loads(cards_json_str)

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
                f"LLM response for {page.url} was not a list or valid dict. Response: {cards_json_str[:200]}..."
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

    except json.JSONDecodeError as e:
        # cards_json_str might not be defined if json.loads fails early, or if response_data was bad
        raw_response_content = "<response_content_unavailable>"
        if "cards_json_str" in locals() and cards_json_str:
            raw_response_content = cards_json_str[:500]
        elif (
            "response_data" in locals()
            and response_data
            and response_data.choices
            and len(response_data.choices) > 0
            and response_data.choices[0].message
            and response_data.choices[0].message.content
        ):
            raw_response_content = response_data.choices[0].message.content[:500]

        logger.error(
            f"Failed to decode JSON response from OpenAI for page {page.url}: {e}. Response: {raw_response_content}...",
            exc_info=True,
        )
        return []
    except OpenAIError as e:
        logger.error(
            f"OpenAI API error while processing page {page.url}: {e}", exc_info=True
        )
        return []
    except Exception as e:
        logger.error(
            f"Unexpected error processing page {page.url} with LLM: {e}", exc_info=True
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
