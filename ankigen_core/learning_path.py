# Module for the 'Analyze Learning Path' feature

import pandas as pd
import gradio as gr  # For gr.Error
from openai import OpenAIError  # For specific error handling

# Imports from our core modules
from ankigen_core.utils import get_logger, ResponseCache
from ankigen_core.llm_interface import OpenAIClientManager, structured_output_completion
# Assuming no specific models needed here unless prompts change
# from ankigen_core.models import ...

logger = get_logger()


def analyze_learning_path(
    client_manager: OpenAIClientManager,  # Expect the manager
    cache: ResponseCache,  # Expect the cache instance
    # --- UI Inputs ---
    api_key: str,
    description: str,
    model: str,
):
    """Analyze a job description or learning goal to create a structured learning path."""
    logger.info(
        f"Starting learning path analysis for description (length: {len(description)}) using model {model}"
    )

    # --- Initialization and Validation ---
    if not api_key:
        logger.warning("No API key provided for learning path analysis")
        raise gr.Error("OpenAI API key is required")

    try:
        # Ensure client is initialized (using the passed manager)
        client_manager.initialize_client(api_key)
        openai_client = client_manager.get_client()
    except (ValueError, RuntimeError, OpenAIError, Exception) as e:
        logger.error(f"Client initialization failed in learning path analysis: {e}")
        raise gr.Error(f"OpenAI Client Error: {e}")

    # --- Prompt Preparation ---
    system_prompt = """You are an expert curriculum designer and educational consultant.
    Your task is to analyze learning goals and create structured, achievable learning paths.
    Break down complex topics into manageable subjects, identify prerequisites,
    and suggest practical projects that reinforce learning.
    Focus on creating a logical progression that builds upon previous knowledge.
    Ensure the output strictly follows the requested JSON format.
    """

    path_prompt = f"""
    Analyze this description and create a structured learning path.
    Return your analysis as a JSON object with the following structure:
    {{
        "subjects": [
            {{
                "Subject": "name of the subject",
                "Prerequisites": "required prior knowledge (list or text)",
                "Time Estimate": "estimated time to learn (e.g., '2 weeks', '10 hours')"
            }}
            // ... more subjects
        ],
        "learning_order": "recommended sequence of study (text description)",
        "projects": "suggested practical projects (list or text description)"
    }}

    Description to analyze:
    --- START DESCRIPTION ---
    {description}
    --- END DESCRIPTION ---
    """

    # --- API Call ---
    try:
        logger.debug("Calling LLM for learning path analysis...")
        response = structured_output_completion(
            openai_client=openai_client,
            model=model,
            response_format={"type": "json_object"},
            system_prompt=system_prompt,
            user_prompt=path_prompt,
            cache=cache,
        )

        # --- Response Processing ---
        if (
            not response
            or not isinstance(response, dict)  # Basic type check
            or "subjects" not in response
            or "learning_order" not in response
            or "projects" not in response
            or not isinstance(response["subjects"], list)  # Check if subjects is a list
        ):
            logger.error(
                f"Invalid or incomplete response format received from API for learning path. Response: {str(response)[:500]}"
            )
            raise gr.Error(
                "Failed to analyze learning path due to invalid API response format. Please try again."
            )

        # Validate subject structure before creating DataFrame
        validated_subjects = []
        for subj in response["subjects"]:
            if (
                isinstance(subj, dict)
                and "Subject" in subj
                and "Prerequisites" in subj
                and "Time Estimate" in subj
            ):
                validated_subjects.append(subj)
            else:
                logger.warning(
                    f"Skipping invalid subject entry in learning path response: {subj}"
                )

        if not validated_subjects:
            logger.error(
                "No valid subjects found in the API response for learning path."
            )
            raise gr.Error("API returned no valid subjects for the learning path.")

        subjects_df = pd.DataFrame(validated_subjects)
        # Ensure required columns exist, add empty if missing to prevent errors downstream
        for col in ["Subject", "Prerequisites", "Time Estimate"]:
            if col not in subjects_df.columns:
                subjects_df[col] = ""  # Add empty column if missing
                logger.warning(f"Added missing column '{col}' to subjects DataFrame.")

        # Format markdown outputs
        learning_order_text = (
            f"### Recommended Learning Order\n{response['learning_order']}"
        )
        projects_text = f"### Suggested Projects\n{response['projects']}"

        logger.info("Successfully analyzed learning path.")
        return subjects_df, learning_order_text, projects_text

    except (ValueError, OpenAIError, RuntimeError, gr.Error) as e:
        # Catch errors raised by structured_output_completion or processing
        logger.error(f"Error during learning path analysis: {str(e)}", exc_info=True)
        # Re-raise Gradio errors, wrap others
        if isinstance(e, gr.Error):
            raise e
        else:
            raise gr.Error(f"Failed to analyze learning path: {str(e)}")
    except Exception as e:
        logger.error(
            f"Unexpected error during learning path analysis: {str(e)}", exc_info=True
        )
        raise gr.Error("An unexpected error occurred during learning path analysis.")
