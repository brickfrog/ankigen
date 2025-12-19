# Module for core card generation logic

import gradio as gr
import pandas as pd
from typing import List, Dict, Any

# Imports from our core modules
from ankigen_core.utils import (
    get_logger,
    ResponseCache,
    strip_html_tags,
)
from ankigen_core.llm_interface import OpenAIClientManager
from ankigen_core.models import (
    Card,
)  # Import necessary Pydantic models

# Import agent system - required
from ankigen_core.agents.integration import AgentOrchestrator
from agents import set_tracing_disabled

logger = get_logger()

# Disable tracing to prevent metrics persistence issues
set_tracing_disabled(True)

AGENTS_AVAILABLE = True
logger.info("Agent system loaded successfully")

# --- Constants --- (Moved from app.py)
AVAILABLE_MODELS = [
    {
        "value": "gpt-5.1",
        "label": "GPT-5.1 (Best Quality)",
        "description": "Latest model with adaptive reasoning, 400K context",
    },
    {
        "value": "gpt-4.1",
        "label": "GPT-4.1 (Legacy)",
        "description": "Previous generation, large context window",
    },
    {
        "value": "gpt-4.1-nano",
        "label": "GPT-4.1 Nano (Legacy Fast)",
        "description": "Previous generation, ultra-fast",
    },
]

GENERATION_MODES = [
    {
        "value": "subject",
        "label": "Single Subject",
        "description": "Generate cards for a specific topic",
    },
]

# --- Core Functions --- (Moved and adapted from app.py)


# Legacy functions removed - all card generation now handled by agent system


def _map_generation_mode_to_subject(generation_mode: str, subject: str) -> str:
    """Map UI generation mode to agent subject."""
    if generation_mode == "subject":
        return subject if subject else "general"
    elif generation_mode == "path":
        return "curriculum_design"
    elif generation_mode == "text":
        return "content_analysis"
    return "general"


def _build_generation_context(generation_mode: str, source_text: str) -> Dict[str, Any]:
    """Build context dict for card generation."""
    context: Dict[str, Any] = {}
    if generation_mode == "text" and source_text:
        context["source_text"] = source_text
    return context


def _get_token_usage_html(token_tracker) -> str:
    """Extract token usage and format as HTML."""
    try:
        if hasattr(token_tracker, "get_session_summary"):
            token_usage = token_tracker.get_session_summary()
        elif hasattr(token_tracker, "get_session_usage"):
            token_usage = token_tracker.get_session_usage()
        else:
            raise AttributeError("TokenTracker has no session summary method")

        return f"<div style='margin-top: 8px;'><b>Token Usage:</b> {token_usage['total_tokens']} tokens</div>"
    except Exception as e:
        logger.error(f"Token usage collection failed: {e}")
        return "<div style='margin-top: 8px;'><b>Token Usage:</b> No usage data</div>"


def _format_cards_to_dataframe(
    agent_cards: List[Card], subject: str
) -> tuple[pd.DataFrame, str]:
    """Format agent cards to DataFrame and generate message."""
    formatted_cards = format_cards_for_dataframe(
        agent_cards,
        topic_name=subject if subject else "General",
        start_index=1,
    )
    output_df = pd.DataFrame(formatted_cards, columns=get_dataframe_columns())
    total_cards_message = f"<div><b>Cards Generated:</b> <span id='total-cards-count'>{len(output_df)}</span></div>"
    return output_df, total_cards_message


async def orchestrate_card_generation(
    client_manager: OpenAIClientManager,
    cache: ResponseCache,
    api_key_input: str,
    subject: str,
    generation_mode: str,
    source_text: str,
    url_input: str,
    model_name: str,
    topic_number: int,
    cards_per_topic: int,
    preference_prompt: str,
    generate_cloze: bool,
    use_llm_judge: bool = False,
    library_name: str = None,
    library_topic: str = None,
    topics_list: List[str] = None,
):
    """Orchestrates the card generation process based on UI inputs."""
    logger.info(f"Starting card generation orchestration in {generation_mode} mode")
    logger.debug(
        f"Parameters: mode={generation_mode}, topics={topic_number}, "
        f"cards_per_topic={cards_per_topic}, cloze={generate_cloze}"
    )

    if not AGENTS_AVAILABLE:
        logger.error("Agent system is required but not available")
        gr.Error("Agent system is required but not available")
        return pd.DataFrame(columns=get_dataframe_columns()), "Agent system error", ""

    try:
        from ankigen_core.agents.token_tracker import get_token_tracker

        token_tracker = get_token_tracker()
        orchestrator = AgentOrchestrator(client_manager)

        logger.info(f"Using {model_name} for SubjectExpertAgent")
        await orchestrator.initialize(api_key_input, {"subject_expert": model_name})

        agent_subject = _map_generation_mode_to_subject(generation_mode, subject)
        context = _build_generation_context(generation_mode, source_text)
        if preference_prompt:
            context["learning_preferences"] = preference_prompt
        total_cards_needed = topic_number * cards_per_topic

        agent_cards, agent_metadata = await orchestrator.generate_cards_with_agents(
            topic=subject if subject else "Mixed Topics",
            subject=agent_subject,
            num_cards=total_cards_needed,
            difficulty="intermediate",
            context=context,
            library_name=library_name,
            library_topic=library_topic,
            generate_cloze=generate_cloze,
            topics_list=topics_list,
            cards_per_topic=cards_per_topic,
        )

        token_usage_html = _get_token_usage_html(token_tracker)

        if agent_cards:
            output_df, total_cards_message = _format_cards_to_dataframe(
                agent_cards, subject
            )
            logger.info(f"Agent system generated {len(output_df)} cards successfully")
            return output_df, total_cards_message, token_usage_html

        logger.error("Agent system returned no cards")
        gr.Error("Agent system returned no cards")
        return (
            pd.DataFrame(columns=get_dataframe_columns()),
            "Agent system returned no cards.",
            "",
        )

    except Exception as e:
        logger.error(f"Agent system failed: {e}")
        gr.Error(f"Agent system error: {str(e)}")
        return (
            pd.DataFrame(columns=get_dataframe_columns()),
            f"Agent system error: {str(e)}",
            "",
        )


# Legacy helper functions removed - all processing now handled by agent system


# --- Formatting and Utility Functions --- (Moved and adapted)
def format_cards_for_dataframe(
    cards: list[Card], topic_name: str, topic_index: int = 0, start_index: int = 1
) -> list:
    """Formats a list of Card objects into a list of dictionaries for DataFrame display.
    Ensures all data is plain text.
    """
    formatted_cards = []
    for i, card_obj in enumerate(cards):
        actual_index = start_index + i
        card_type = card_obj.card_type or "basic"
        question = card_obj.front.question or ""
        answer = card_obj.back.answer or ""
        explanation = card_obj.back.explanation or ""
        example = card_obj.back.example or ""

        # Metadata processing
        metadata = card_obj.metadata or {}
        prerequisites = metadata.get("prerequisites", [])
        learning_outcomes = metadata.get("learning_outcomes", [])
        difficulty = metadata.get("difficulty", "N/A")
        # Ensure list-based metadata are joined as plain strings for DataFrame
        prerequisites_str = strip_html_tags(
            ", ".join(prerequisites)
            if isinstance(prerequisites, list)
            else str(prerequisites)
        )
        learning_outcomes_str = strip_html_tags(
            ", ".join(learning_outcomes)
            if isinstance(learning_outcomes, list)
            else str(learning_outcomes)
        )
        difficulty_str = strip_html_tags(str(difficulty))

        formatted_card = {
            "Index": (
                f"{topic_index}.{actual_index}"
                if topic_index > 0
                else str(actual_index)
            ),
            "Topic": strip_html_tags(topic_name),  # Ensure topic is also plain
            "Card_Type": strip_html_tags(card_type),
            "Question": question,  # Already stripped during Card object creation
            "Answer": answer,  # Already stripped
            "Explanation": explanation,  # Already stripped
            "Example": example,  # Already stripped
            "Prerequisites": prerequisites_str,
            "Learning_Outcomes": learning_outcomes_str,
            "Difficulty": difficulty_str,  # Ensure difficulty is plain text
            "Source_URL": strip_html_tags(
                metadata.get("source_url", "")
            ),  # Ensure Source_URL is plain
        }
        formatted_cards.append(formatted_card)
    return formatted_cards


def get_dataframe_columns() -> list[str]:
    """Returns the standard list of columns for the Anki card DataFrame."""
    return [
        "Index",
        "Topic",
        "Card_Type",
        "Question",
        "Answer",
        "Explanation",
        "Example",
        "Prerequisites",
        "Learning_Outcomes",
        "Difficulty",
        "Source_URL",
    ]


def generate_token_usage_html(token_usage=None):
    """Generate HTML for token usage display"""
    if token_usage and isinstance(token_usage, dict):
        total_tokens = token_usage.get("total_tokens", 0)
        return f"<div style='margin-top: 8px;'><b>Token Usage:</b> {total_tokens} tokens</div>"
    else:
        return "<div style='margin-top: 8px;'><b>Token Usage:</b> No usage data</div>"
