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
        "value": "gpt-4.1",
        "label": "GPT-4.1 (Best Quality)",
        "description": "Highest quality, large context window",
    },
    {
        "value": "gpt-4.1-nano",
        "label": "GPT-4.1 Nano (Ultra Fast)",
        "description": "Ultra-fast and cost-effective",
    },
]

GENERATION_MODES = [
    {
        "value": "subject",
        "label": "Single Subject",
        "description": "Generate cards for a specific topic",
    },
    {
        "value": "path",
        "label": "Learning Path",
        "description": "Break down a job description or learning goal into subjects",
    },
    {
        "value": "text",
        "label": "From Text",
        "description": "Generate cards from provided text",
    },
    {
        "value": "web",
        "label": "From Web",
        "description": "Generate cards from a web page URL",
    },
]

# --- Core Functions --- (Moved and adapted from app.py)


# Legacy functions removed - all card generation now handled by agent system


async def orchestrate_card_generation(  # MODIFIED: Added async
    client_manager: OpenAIClientManager,  # Expect the manager
    cache: ResponseCache,  # Expect the cache instance
    # --- UI Inputs --- (These will be passed from app.py handler)
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
):
    """Orchestrates the card generation process based on UI inputs."""

    logger.info(f"Starting card generation orchestration in {generation_mode} mode")
    logger.debug(
        f"Parameters: mode={generation_mode}, topics={topic_number}, cards_per_topic={cards_per_topic}, cloze={generate_cloze}"
    )

    # --- AGENT SYSTEM INTEGRATION ---
    if AGENTS_AVAILABLE:
        logger.info("🤖 Using agent system for card generation")
        try:
            from ankigen_core.agents.token_tracker import get_token_tracker

            token_tracker = get_token_tracker()

            orchestrator = AgentOrchestrator(client_manager)

            logger.info(f"Using {model_name} for SubjectExpertAgent")
            await orchestrator.initialize(api_key_input, {"subject_expert": model_name})

            # Map generation mode to subject
            agent_subject = "general"
            if generation_mode == "subject":
                agent_subject = subject if subject else "general"
            elif generation_mode == "path":
                agent_subject = "curriculum_design"
            elif generation_mode == "text":
                agent_subject = "content_analysis"

            total_cards_needed = topic_number * cards_per_topic

            context = {}
            if generation_mode == "text" and source_text:
                context["source_text"] = source_text

            agent_cards, agent_metadata = await orchestrator.generate_cards_with_agents(
                topic=subject if subject else "Mixed Topics",
                subject=agent_subject,
                num_cards=total_cards_needed,
                difficulty="intermediate",
                context=context,
                library_name=library_name,
                library_topic=library_topic,
            )

            # Get token usage from session
            try:
                # Try both method names for compatibility
                if hasattr(token_tracker, "get_session_summary"):
                    token_usage = token_tracker.get_session_summary()
                elif hasattr(token_tracker, "get_session_usage"):
                    token_usage = token_tracker.get_session_usage()
                else:
                    raise AttributeError("TokenTracker has no session summary method")

                token_usage_html = f"<div style='margin-top: 8px;'><b>Token Usage:</b> {token_usage['total_tokens']} tokens</div>"
            except Exception as e:
                logger.error(f"Token usage collection failed: {e}")
                token_usage_html = "<div style='margin-top: 8px;'><b>Token Usage:</b> No usage data</div>"

            # Convert agent cards to dataframe format
            if agent_cards:
                formatted_cards = format_cards_for_dataframe(
                    agent_cards,
                    topic_name=subject if subject else "General",
                    start_index=1,
                )

                output_df = pd.DataFrame(
                    formatted_cards, columns=get_dataframe_columns()
                )
                total_cards_message = f"<div><b>Cards Generated:</b> <span id='total-cards-count'>{len(output_df)}</span></div>"

                logger.info(
                    f"Agent system generated {len(output_df)} cards successfully"
                )
                return output_df, total_cards_message, token_usage_html
            else:
                logger.error("Agent system returned no cards")
                gr.Error("🤖 Agent system returned no cards")
                return (
                    pd.DataFrame(columns=get_dataframe_columns()),
                    "Agent system returned no cards.",
                    "",
                )

        except Exception as e:
            logger.error(f"Agent system failed: {e}")
            gr.Error(f"🤖 Agent system error: {str(e)}")
            return (
                pd.DataFrame(columns=get_dataframe_columns()),
                f"Agent system error: {str(e)}",
                "",
            )

    # Agent system is required and should never fail to be available
    logger.error("Agent system failed but is required - this should not happen")
    gr.Error("Agent system is required but not available")
    return (
        pd.DataFrame(columns=get_dataframe_columns()),
        "Agent system error",
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


# This function might be specific to the old crawler flow if AnkiCardData is only from there.
# If orchestrate_card_generation now also produces something convertible to AnkiCardData, it might be useful.
# For now, it's used by generate_cards_from_crawled_content.
def deduplicate_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates a list of card dictionaries based on the 'Question' field."""
    seen_questions = set()
    unique_cards = []
    for card_dict in cards:
        question = card_dict.get("Question")
        if question is None:  # Should not happen if cards are well-formed
            logger.warning(f"Card dictionary missing 'Question' key: {card_dict}")
            unique_cards.append(card_dict)  # Keep it if no question to dedupe on
            continue

        # Normalize whitespace and case for deduplication
        normalized_question = " ".join(str(question).strip().lower().split())
        if normalized_question not in seen_questions:
            seen_questions.add(normalized_question)
            unique_cards.append(card_dict)
        else:
            logger.info(f"Deduplicated card with question: {question}")
    return unique_cards


# --- Modification for generate_cards_from_crawled_content ---


def generate_cards_from_crawled_content(
    all_cards: List[Card],
) -> List[Dict[str, Any]]:  # Changed AnkiCardData to Card
    """
    Processes a list of Card objects (expected to have plain text fields after generate_cards_batch)
    and formats them into a list of dictionaries suitable for the DataFrame.
    """
    if not all_cards:
        return []

    data_for_dataframe = []
    for i, card_obj in enumerate(all_cards):
        # Extract data, assuming it's already plain text from Card object creation
        topic = (
            card_obj.metadata.get("topic", f"Crawled Content - Card {i+1}")
            if card_obj.metadata
            else f"Crawled Content - Card {i+1}"
        )

        # Ensure list-based metadata are joined as plain strings for DataFrame
        prerequisites = (
            card_obj.metadata.get("prerequisites", []) if card_obj.metadata else []
        )
        learning_outcomes = (
            card_obj.metadata.get("learning_outcomes", []) if card_obj.metadata else []
        )

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
        difficulty_str = strip_html_tags(
            str(
                card_obj.metadata.get("difficulty", "N/A")
                if card_obj.metadata
                else "N/A"
            )
        )

        card_dict = {
            "Index": str(i + 1),
            "Topic": strip_html_tags(topic),
            "Card_Type": strip_html_tags(card_obj.card_type or "basic"),
            "Question": card_obj.front.question or "",  # Should be plain
            "Answer": card_obj.back.answer or "",  # Should be plain
            "Explanation": card_obj.back.explanation or "",  # Should be plain
            "Example": card_obj.back.example or "",  # Should be plain
            "Prerequisites": prerequisites_str,
            "Learning_Outcomes": learning_outcomes_str,
            "Difficulty": difficulty_str,
            "Source_URL": strip_html_tags(
                card_obj.metadata.get("source_url", "") if card_obj.metadata else ""
            ),
        }
        data_for_dataframe.append(card_dict)
    return data_for_dataframe


def generate_token_usage_html(token_usage=None):
    """Generate HTML for token usage display"""
    if token_usage and isinstance(token_usage, dict):
        total_tokens = token_usage.get("total_tokens", 0)
        return f"<div style='margin-top: 8px;'><b>Token Usage:</b> {total_tokens} tokens</div>"
    else:
        return "<div style='margin-top: 8px;'><b>Token Usage:</b> No usage data</div>"
