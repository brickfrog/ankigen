# Module for OpenAI client management and API call logic

from openai import (
    OpenAI,
    OpenAIError,
)  # Added OpenAIError for specific exception handling
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Imports from our new core modules
from ankigen_core.utils import get_logger, ResponseCache
# We will need Pydantic models if response_format is a Pydantic model,
# but for now, it's a dict like {"type": "json_object"}.
# from ankigen_core.models import ... # Placeholder if needed later

logger = get_logger()


class OpenAIClientManager:
    """Manages the OpenAI client instance."""

    def __init__(self):
        self._client = None
        self._api_key = None

    def initialize_client(self, api_key: str):
        """Initializes the OpenAI client with the given API key."""
        if not api_key or not api_key.startswith("sk-"):
            logger.error("Invalid OpenAI API key provided for client initialization.")
            # Decide if this should raise an error or just log and leave client as None
            raise ValueError("Invalid OpenAI API key format.")
        self._api_key = api_key
        try:
            self._client = OpenAI(api_key=self._api_key)
            logger.info("OpenAI client initialized successfully.")
        except OpenAIError as e:  # Catch specific OpenAI errors
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            self._client = None  # Ensure client is None on failure
            raise  # Re-raise the OpenAIError to be caught by UI
        except Exception as e:  # Catch any other unexpected errors
            logger.error(
                f"An unexpected error occurred during OpenAI client initialization: {e}",
                exc_info=True,
            )
            self._client = None
            raise RuntimeError("Unexpected error initializing OpenAI client.")

    def get_client(self):
        """Returns the initialized OpenAI client. Raises error if not initialized."""
        if self._client is None:
            logger.error(
                "OpenAI client accessed before initialization or after a failed initialization."
            )
            raise RuntimeError(
                "OpenAI client is not initialized. Please provide a valid API key."
            )
        return self._client


# Retry decorator for API calls - kept similar to original
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        Exception
    ),  # Consider refining this to specific network/API errors
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying structured_output_completion (attempt {retry_state.attempt_number}) due to {retry_state.outcome.exception()}"
    ),
)
def structured_output_completion(
    openai_client: OpenAI,  # Expecting an initialized OpenAI client instance
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
        return cached_response

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

        completion = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": effective_system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            response_format=response_format,  # Pass the dict directly
            temperature=0.7,  # Consider making this configurable
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
        logger.error(
            f"Failed to parse JSON response from model {model}: {e}. Response: {first_choice.message.content[:500]}",
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
