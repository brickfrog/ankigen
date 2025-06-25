# Module for core card generation logic

import gradio as gr
import pandas as pd
from typing import List, Dict, Any
import asyncio
from urllib.parse import urlparse

# Imports from our core modules
from ankigen_core.utils import (
    get_logger,
    ResponseCache,
    fetch_webpage_text,
    strip_html_tags,
)
from ankigen_core.llm_interface import OpenAIClientManager, structured_output_completion
from ankigen_core.models import (
    Card,
    CardFront,
    CardBack,
)  # Import necessary Pydantic models

logger = get_logger()

# --- Constants --- (Moved from app.py)
AVAILABLE_MODELS = [
    {
        "value": "gpt-4.1",
        "label": "gpt-4.1 (Best Quality)",
        "description": "Highest quality, slower generation",
    },
    {
        "value": "gpt-4.1-nano",
        "label": "gpt-4.1 Nano (Fast & Efficient)",
        "description": "Optimized for speed and lower cost",
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


async def generate_cards_batch(
    openai_client,  # Renamed from client to openai_client for clarity
    cache: ResponseCache,  # Added cache parameter
    model: str,
    topic: str,
    num_cards: int,
    system_prompt: str,
    generate_cloze: bool = False,
    batch_size: int = 3,  # Keep batch_size, though not explicitly used in this version
):
    """Generate a batch of cards for a topic, potentially including cloze deletions"""

    cloze_instruction = ""
    if generate_cloze:
        cloze_instruction = """
        Where appropriate, generate Cloze deletion cards.
        - For Cloze cards, set "card_type" to "cloze".
        - Format the question field using Anki's cloze syntax (e.g., "The capital of France is {{c1::Paris}}.").
        - The "answer" field should contain the full, non-cloze text or specific context for the cloze.
        - For standard question/answer cards, set "card_type" to "basic".
        """

    cards_prompt = f"""
    Generate {num_cards} flashcards for the topic: {topic}
    {cloze_instruction}
    Return your response as a JSON object with the following structure:
    {{
        "cards": [
            {{
                "card_type": "basic or cloze",
                "front": {{
                    "question": "question text (potentially with {{{{c1::cloze syntax}}}})"
                }},
                "back": {{
                    "answer": "concise answer or full text for cloze",
                    "explanation": "detailed explanation",
                    "example": "practical example"
                }},
                "metadata": {{
                    "prerequisites": ["list", "of", "prerequisites"],
                    "learning_outcomes": ["list", "of", "outcomes"],
                    "misconceptions": ["list", "of", "misconceptions"],
                    "difficulty": "beginner/intermediate/advanced"
                }}
            }}
            // ... more cards
        ]
    }}
    """

    try:
        logger.info(
            f"Generating card batch for {topic}, Cloze enabled: {generate_cloze}"
        )
        # Call the imported structured_output_completion, passing client and cache
        response = await structured_output_completion(
            openai_client=openai_client,
            model=model,
            response_format={"type": "json_object"},
            system_prompt=system_prompt,
            user_prompt=cards_prompt,
            cache=cache,  # Pass the cache instance
        )

        if not response or "cards" not in response:
            logger.error("Invalid cards response format")
            raise ValueError("Failed to generate cards. Please try again.")

        cards_list = []
        for card_data in response["cards"]:
            if "front" not in card_data or "back" not in card_data:
                logger.warning(
                    f"Skipping card due to missing front/back data: {card_data}"
                )
                continue
            if "question" not in card_data["front"]:
                logger.warning(f"Skipping card due to missing question: {card_data}")
                continue
            if (
                "answer" not in card_data["back"]
                or "explanation" not in card_data["back"]
                or "example" not in card_data["back"]
            ):
                logger.warning(
                    f"Skipping card due to missing answer/explanation/example: {card_data}"
                )
                continue

            # Use imported Pydantic models
            card = Card(
                card_type=card_data.get("card_type", "basic"),
                front=CardFront(
                    question=strip_html_tags(card_data["front"].get("question", ""))
                ),
                back=CardBack(
                    answer=strip_html_tags(card_data["back"].get("answer", "")),
                    explanation=strip_html_tags(
                        card_data["back"].get("explanation", "")
                    ),
                    example=strip_html_tags(card_data["back"].get("example", "")),
                ),
                metadata=card_data.get("metadata", {}),
            )
            cards_list.append(card)

        return cards_list

    except Exception as e:
        logger.error(
            f"Failed to generate cards batch for {topic}: {str(e)}", exc_info=True
        )
        raise  # Re-raise for the main function to handle


async def judge_card(
    openai_client,
    cache: ResponseCache,
    model: str,
    card: Card,
) -> bool:
    """Use an LLM to validate a single card."""
    system_prompt = (
        "You review flashcards and decide if the question is clear and useful. "
        'Respond with a JSON object like {"is_valid": true}.'
    )
    user_prompt = f"Question: {card.front.question}\nAnswer: {card.back.answer}"
    try:
        result = await structured_output_completion(
            openai_client=openai_client,
            model=model,
            response_format={"type": "json_object"},
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cache=cache,
        )
        if isinstance(result, dict):
            return bool(result.get("is_valid", True))
    except Exception as e:  # pragma: no cover - network or parse errors
        logger.warning(f"LLM judge failed for card '{card.front.question}': {e}")
    return True


async def judge_cards(
    openai_client,
    cache: ResponseCache,
    model: str,
    cards: List[Card],
) -> List[Card]:
    """Filter cards using the LLM judge."""
    validated: List[Card] = []
    for card in cards:
        if await judge_card(openai_client, cache, model, card):
            validated.append(card)
        else:
            logger.info(f"Card rejected by judge: {card.front.question}")
    return validated


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
):
    """Orchestrates the card generation process based on UI inputs."""

    logger.info(f"Starting card generation orchestration in {generation_mode} mode")
    logger.debug(
        f"Parameters: mode={generation_mode}, topics={topic_number}, cards_per_topic={cards_per_topic}, cloze={generate_cloze}"
    )

    # --- Initialization and Validation ---
    if not api_key_input:
        logger.warning("No API key provided to orchestrator")
        gr.Error("OpenAI API key is required")
        return pd.DataFrame(columns=get_dataframe_columns()), "API key is required.", 0
    # Re-initialize client via manager if API key changes or not initialized
    # This logic might need refinement depending on how API key state is managed in UI
    try:
        # Attempt to initialize (will raise error if key is invalid)
        await client_manager.initialize_client(api_key_input)
        openai_client = client_manager.get_client()
    except (ValueError, RuntimeError, Exception) as e:
        logger.error(f"Client initialization failed in orchestrator: {e}")
        gr.Error(f"OpenAI Client Error: {e}")
        return (
            pd.DataFrame(columns=get_dataframe_columns()),
            f"OpenAI Client Error: {e}",
            0,
        )

    model = model_name
    flattened_data = []
    total_cards_generated = 0
    # Use track_tqdm=True in the calling Gradio handler if desired
    # progress_tracker = gr.Progress(track_tqdm=True)

    # -------------------------------------

    try:
        # page_text_for_generation = "" # No longer needed here

        # --- Web Mode (Crawler) is now handled by crawl_and_generate in ui_logic.py ---
        # The 'web' case for orchestrate_card_generation is removed as it's a separate flow.
        # This function now handles 'subject', 'path', and 'text' (where text can be a URL).

        # --- Subject Mode ---
        if generation_mode == "subject":
            logger.info("Orchestrator: Subject Mode")
            if not subject or not subject.strip():
                gr.Error("Subject is required for 'Single Subject' mode.")
                return (
                    pd.DataFrame(columns=get_dataframe_columns()),
                    "Subject is required.",
                    gr.update(
                        value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
                        visible=False,
                    ),
                )
            system_prompt = f"""You are an expert in {subject} and an experienced educator. {preference_prompt}"""
            # Split subjects if multiple are comma-separated
            individual_subjects = [s.strip() for s in subject.split(",") if s.strip()]
            if (
                not individual_subjects
            ):  # Handle case where subject might be just commas or whitespace
                gr.Error("Valid subject(s) required.")
                return (
                    pd.DataFrame(columns=get_dataframe_columns()),
                    "Valid subject(s) required.",
                    gr.update(
                        value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
                        visible=False,
                    ),
                )

            topics_for_generation = []
            max(1, topic_number // len(individual_subjects))  # Distribute topic_number

            for ind_subject in individual_subjects:
                # For single/multiple subjects, we might generate sub-topics or just use the subject as a topic
                # For simplicity, let's assume each subject passed is a "topic" for now,
                # and cards_per_topic applies to each.
                # Or, if topic_number > 1, we could try to make LLM break down ind_subject into num_topics_per_subject.
                # Current UI has "Number of Topics" and "Cards per Topic".
                # If "Number of Topics" is meant per subject provided, then this logic needs care.
                # Let's assume "Number of Topics" is total, and we divide it.
                # If "Single Subject" mode, topic_number might represent sub-topics of that single subject.

                # For now, let's simplify: treat each provided subject as a high-level topic.
                # And generate 'cards_per_topic' for each. 'topic_number' might be less relevant here or define sub-breakdown.
                # To align with UI (topic_number and cards_per_topic), if multiple subjects,
                # we could make `topic_number` apply to how many sub-topics to generate for EACH subject,
                # and `cards_per_topic` for each of those sub-topics.
                # Or, if len(individual_subjects) > 1, `topic_number` is ignored and we use `cards_per_topic` for each subject.

                # Simpler: if 1 subject, topic_number is subtopics. If multiple, each is a topic.
                if len(individual_subjects) == 1:
                    # If it's a single subject, we might want to break it down into `topic_number` sub-topics.
                    # This would require an LLM call to get sub-topics first.
                    # For now, let's treat the single subject as one topic, and `topic_number` is ignored.
                    # Or, let's assume `topic_number` means we want `topic_number` variations or aspects of this subject.
                    # The prompt for generate_cards_batch takes a "topic".
                    # Let's create `topic_number` "topics" that are just slight variations or aspects of the main subject.
                    if topic_number == 1:
                        topics_for_generation.append(
                            {"name": ind_subject, "num_cards": cards_per_topic}
                        )
                    else:
                        # This is a placeholder for a more sophisticated sub-topic generation
                        # For now, just make `topic_number` distinct calls for the same subject if user wants more "topics"
                        # gr.Info(f"Generating for {topic_number} aspects/sub-sections of '{ind_subject}'.")
                        for i in range(topic_number):
                            topics_for_generation.append(
                                {
                                    "name": f"{ind_subject} - Aspect {i + 1}",
                                    "num_cards": cards_per_topic,
                                }
                            )
                else:  # Multiple subjects provided
                    topics_for_generation.append(
                        {"name": ind_subject, "num_cards": cards_per_topic}
                    )

        # --- Learning Path Mode ---
        elif generation_mode == "path":
            logger.info("Orchestrator: Learning Path Mode")
            # In path mode, 'subject' contains the pre-analyzed subjects, comma-separated.
            # 'description' (the learning goal) was used by analyze_learning_path, not directly here for card gen.
            if (
                not subject or not subject.strip()
            ):  # 'subject' here comes from the anki_cards_data_df after analysis
                gr.Error("No subjects provided from learning path analysis.")
                return (
                    pd.DataFrame(columns=get_dataframe_columns()),
                    "No subjects from path analysis.",
                    gr.update(
                        value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
                        visible=False,
                    ),
                )

            system_prompt = f"""You are an expert in curriculum design and an experienced educator. {preference_prompt}"""
            analyzed_subjects = [s.strip() for s in subject.split(",") if s.strip()]
            if not analyzed_subjects:
                gr.Error("No valid subjects parsed from learning path.")
                return (
                    pd.DataFrame(columns=get_dataframe_columns()),
                    "No valid subjects from path.",
                    gr.update(
                        value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
                        visible=False,
                    ),
                )

            # topic_number might be interpreted as how many cards to generate for EACH analyzed subject,
            # or how many sub-topics to break each analyzed subject into.
            # Given "Cards per Topic" slider, it's more likely each analyzed subject is a "topic".
            topics_for_generation = [
                {"name": subj, "num_cards": cards_per_topic}
                for subj in analyzed_subjects
            ]

        # --- Text Mode / Single Web Page from Text Mode ---
        elif generation_mode == "text":
            logger.info("Orchestrator: Text Mode")
            actual_text_to_process = source_text

            if (
                not actual_text_to_process or not actual_text_to_process.strip()
            ):  # Check after potential fetch
                gr.Error("Text input is empty.")
                return (
                    pd.DataFrame(columns=get_dataframe_columns()),
                    "Text input is empty.",
                    gr.update(
                        value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
                        visible=False,
                    ),
                )

            # Check if source_text is a URL
            # Use a more robust check for URL (e.g., regex or urllib.parse)
            is_url = False
            if isinstance(source_text, str) and source_text.strip().lower().startswith(
                ("http://", "https://")
            ):
                try:
                    # A more robust check could involve trying to parse it
                    result = urlparse(source_text.strip())
                    if all([result.scheme, result.netloc]):
                        is_url = True
                except ImportError:  # Fallback if urlparse not available (should be)
                    pass  # is_url remains False

            if is_url:
                url_to_fetch = source_text.strip()
                logger.info(f"Text mode identified URL: {url_to_fetch}")
                gr.Info(f"🕸️ Fetching content from URL in text field: {url_to_fetch}...")
                try:
                    page_content = await asyncio.to_thread(
                        fetch_webpage_text, url_to_fetch
                    )  # Ensure fetch_webpage_text is thread-safe or run in executor
                    if not page_content or not page_content.strip():
                        gr.Warning(
                            f"Could not extract meaningful text from URL: {url_to_fetch}. Please check the URL or page content."
                        )
                        return (
                            pd.DataFrame(columns=get_dataframe_columns()),
                            "No meaningful text extracted from URL.",
                            gr.update(
                                value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
                                visible=False,
                            ),
                        )
                    actual_text_to_process = page_content
                    source_text_display_name = f"Content from {url_to_fetch}"
                    gr.Info(
                        f"✅ Successfully fetched text from URL (approx. {len(actual_text_to_process)} chars)."
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to fetch or process URL {url_to_fetch} in text mode: {e}",
                        exc_info=True,
                    )
                    gr.Error(f"Failed to fetch content from URL: {str(e)}")
                    return (
                        pd.DataFrame(columns=get_dataframe_columns()),
                        f"URL fetch error: {str(e)}",
                        gr.update(
                            value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
                            visible=False,
                        ),
                    )
            else:  # Not a URL, or failed to parse as one
                if (
                    not source_text or not source_text.strip()
                ):  # Re-check original source_text if not a URL
                    gr.Error("Text input is empty.")
                    return (
                        pd.DataFrame(columns=get_dataframe_columns()),
                        "Text input is empty.",
                        gr.update(
                            value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
                            visible=False,
                        ),
                    )
                actual_text_to_process = source_text  # Use as is
                source_text_display_name = "Content from Provided Text"
                logger.info("Text mode: Processing provided text directly.")

            # For text mode (either direct text or fetched from URL), generate cards from this content.
            # The LLM will need the text. We can pass it via the system prompt or a specialized user prompt.
            # For now, let's use a system prompt that tells it to base cards on the provided text.
            # And we'll create one "topic" for all cards.

            system_prompt = f"""You are an expert in distilling information and creating flashcards from text. {preference_prompt}
            Base your flashcards STRICTLY on the following text content provided by the user in their next message.
            Do not use external knowledge unless explicitly asked to clarify something from the text.
            The user will provide the text content that needs to be turned into flashcards."""  # System prompt now expects text in user prompt.

            # The user_prompt in generate_cards_batch will need to include actual_text_to_process.
            # Let's adapt generate_cards_batch or how it's called for this.
            # For now, let's assume generate_cards_batch's `cards_prompt` will be wrapped or modified
            # to include `actual_text_to_process` when `generation_mode` is "text".

            # This requires a change in how `generate_cards_batch` constructs its `cards_prompt` if text is primary.
            # Alternative: pass `actual_text_to_process` as part of the user_prompt to `structured_output_completion`
            # directly from here, bypassing `generate_cards_batch`'s topic-based prompt for "text" mode.
            # This seems cleaner.

            # Let's make a direct call to structured_output_completion for "text" mode.
            text_mode_user_prompt = f"""
            Please generate {cards_per_topic * topic_number} flashcards based on the following text content.
            I have already provided the text content in the system prompt (or it is implicitly part of this context).
            Ensure the flashcards cover diverse aspects of the text.
            {get_cloze_instruction(generate_cloze)}
            Return your response as a JSON object with the following structure:
            {get_card_json_structure_prompt()}

            Text Content to process:
            ---
            {actual_text_to_process[:15000]}
            ---
            """  # Truncate to avoid excessive length, system prompt already set context.

            gr.Info(f"Generating cards from: {source_text_display_name}...")
            try:
                response = await structured_output_completion(
                    openai_client=openai_client,
                    model=model,
                    response_format={"type": "json_object"},
                    system_prompt=system_prompt,  # System prompt instructs to use text from user prompt
                    user_prompt=text_mode_user_prompt,  # User prompt contains the text
                    cache=cache,
                )
                raw_cards = []  # Default if response is None
                if response:
                    raw_cards = response.get("cards", [])
                else:
                    logger.warning(
                        "structured_output_completion returned None, defaulting to empty card list for text mode."
                    )
                processed_cards = process_raw_cards_data(raw_cards)
                if use_llm_judge and processed_cards:
                    processed_cards = await judge_cards(
                        openai_client, cache, model, processed_cards
                    )
                formatted_cards = format_cards_for_dataframe(
                    processed_cards, topic_name=source_text_display_name, start_index=1
                )
                flattened_data.extend(formatted_cards)
                total_cards_generated += len(formatted_cards)

                # Skip topics_for_generation loop for text mode as cards are generated directly.
                topics_for_generation = []  # Ensure it's empty

            except Exception as e:
                logger.error(
                    f"Error during 'From Text' card generation: {e}", exc_info=True
                )
                gr.Error(f"Error generating cards from text: {str(e)}")
                return (
                    pd.DataFrame(columns=get_dataframe_columns()),
                    f"Text Gen Error: {str(e)}",
                    gr.update(
                        value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
                        visible=False,
                    ),
                )

        else:  # Should not happen if generation_mode is validated, but as a fallback
            logger.error(f"Unknown generation mode: {generation_mode}")
            gr.Error(f"Unknown generation mode: {generation_mode}")
            return (
                pd.DataFrame(columns=get_dataframe_columns()),
                "Unknown mode.",
                gr.update(
                    value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
                    visible=False,
                ),
            )

        # --- Batch Generation Loop (for subject and path modes) ---
        # progress_total_batches = len(topics_for_generation)
        # current_batch_num = 0

        for topic_info in (
            topics_for_generation
        ):  # This loop will be skipped if text_mode populated flattened_data directly
            # current_batch_num += 1
            # progress_tracker.progress(current_batch_num / progress_total_batches, desc=f"Generating for topic: {topic_info['name']}")
            # logger.info(f"Progress: {current_batch_num}/{progress_total_batches} - Topic: {topic_info['name']}")
            gr.Info(
                f"Generating cards for topic: {topic_info['name']}..."
            )  # UI feedback

            try:
                # System prompt is already set based on mode (subject/path)
                # generate_cards_batch will use this system_prompt
                batch_cards = await generate_cards_batch(
                    openai_client,
                    cache,
                    model,
                    topic_info["name"],
                    topic_info["num_cards"],
                    system_prompt,  # System prompt defined above based on mode
                    generate_cloze,
                )
                if use_llm_judge and batch_cards:
                    batch_cards = await judge_cards(
                        openai_client, cache, model, batch_cards
                    )
                # Assign topic name to cards before formatting for DataFrame
                formatted_batch = format_cards_for_dataframe(
                    batch_cards,
                    topic_name=topic_info["name"],
                    start_index=total_cards_generated + 1,
                )
                flattened_data.extend(formatted_batch)
                total_cards_generated += len(formatted_batch)
                logger.info(
                    f"Generated {len(formatted_batch)} cards for topic {topic_info['name']}"
                )

            except Exception as e:
                logger.error(
                    f"Error generating cards for topic {topic_info['name']}: {e}",
                    exc_info=True,
                )
                # Optionally, decide if one topic failing should stop all, or just skip
                gr.Warning(
                    f"Could not generate cards for topic '{topic_info['name']}': {str(e)}. Skipping."
                )
                continue  # Continue to next topic

        # --- Final Processing ---
        if not flattened_data:
            gr.Info(
                "No cards were generated."
            )  # More informative than just empty table
            # Return empty dataframe with correct columns
            return (
                pd.DataFrame(columns=get_dataframe_columns()),
                "No cards generated.",
                gr.update(
                    value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
                    visible=False,
                ),
            )

        # Deduplication (if needed, and if it makes sense across different topics)
        # For now, deduplication logic might be too aggressive if topics are meant to have overlapping concepts from different angles.
        # final_cards_data = deduplicate_cards(flattened_data) # Assuming deduplicate_cards expects list of dicts
        final_cards_data = (
            flattened_data  # Skipping deduplication for now to preserve topic structure
        )

        # Re-index cards if deduplication changed the count or if start_index logic wasn't perfect
        # For now, format_cards_for_dataframe handles indexing.

        output_df = pd.DataFrame(final_cards_data, columns=get_dataframe_columns())

        total_cards_message = f"<div><b>Total Cards Generated:</b> <span id='total-cards-count'>{len(output_df)}</span></div>"

        logger.info(f"Orchestration complete. Total cards: {len(output_df)}")
        return output_df, total_cards_message

    except Exception as e:
        logger.error(
            f"Critical error in orchestrate_card_generation: {e}", exc_info=True
        )
        gr.Error(f"An unexpected error occurred: {str(e)}")
        return (
            pd.DataFrame(columns=get_dataframe_columns()),
            f"Unexpected error: {str(e)}",
            gr.update(
                value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
                visible=False,
            ),
        )
    finally:
        # Placeholder if any cleanup is needed
        pass


# Helper function to get Cloze instruction string
def get_cloze_instruction(generate_cloze: bool) -> str:
    if generate_cloze:
        return """
        Where appropriate, generate Cloze deletion cards.
        - For Cloze cards, set "card_type" to "cloze".
        - Format the question field using Anki's cloze syntax (e.g., "The capital of France is {{c1::Paris}}.").
        - The "answer" field should contain the full, non-cloze text or specific context for the cloze.
        - For standard question/answer cards, set "card_type" to "basic".
        """
    return ""


# Helper function to get JSON structure prompt for cards
def get_card_json_structure_prompt() -> str:
    return """
    {
        "cards": [
            {
                "card_type": "basic or cloze",
                "front": {
                    "question": "question text (potentially with {{{{c1::cloze syntax}}}})"
                },
                "back": {
                    "answer": "concise answer or full text for cloze",
                    "explanation": "detailed explanation",
                    "example": "practical example"
                },
                "metadata": {
                    "prerequisites": ["list", "of", "prerequisites"],
                    "learning_outcomes": ["list", "of", "outcomes"],
                    "misconceptions": ["list", "of", "misconceptions"],
                    "difficulty": "beginner/intermediate/advanced"
                }
            }
            // ... more cards
        ]
    }
    """


# Helper function to process raw card data from LLM into Card Pydantic models
def process_raw_cards_data(cards_data: list) -> list[Card]:
    cards_list = []
    if not isinstance(cards_data, list):
        logger.warning(
            f"Expected a list of cards, got {type(cards_data)}. Raw data: {cards_data}"
        )
        return cards_list

    for card_item in cards_data:
        if not isinstance(card_item, dict):
            logger.warning(
                f"Expected card item to be a dict, got {type(card_item)}. Item: {card_item}"
            )
            continue
        try:
            # Basic validation for essential fields
            if (
                not all(k in card_item for k in ["front", "back"])
                or not isinstance(card_item["front"], dict)
                or not isinstance(card_item["back"], dict)
                or "question" not in card_item["front"]
                or "answer" not in card_item["back"]
            ):
                logger.warning(
                    f"Skipping card due to missing essential fields: {card_item}"
                )
                continue

            card = Card(
                card_type=card_item.get("card_type", "basic"),
                front=CardFront(
                    question=strip_html_tags(card_item["front"].get("question", ""))
                ),
                back=CardBack(
                    answer=strip_html_tags(card_item["back"].get("answer", "")),
                    explanation=strip_html_tags(
                        card_item["back"].get("explanation", "")
                    ),
                    example=strip_html_tags(card_item["back"].get("example", "")),
                ),
                metadata=card_item.get("metadata", {}),
            )
            cards_list.append(card)
        except Exception as e:  # Catch Pydantic validation errors or others
            logger.error(
                f"Error processing card data item: {card_item}. Error: {e}",
                exc_info=True,
            )
    return cards_list


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
        common_misconceptions = metadata.get("misconceptions", [])
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
        common_misconceptions_str = strip_html_tags(
            ", ".join(common_misconceptions)
            if isinstance(common_misconceptions, list)
            else str(common_misconceptions)
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
            "Common_Misconceptions": common_misconceptions_str,
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
        "Common_Misconceptions",
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
        common_misconceptions = (
            card_obj.metadata.get("common_misconceptions", [])
            if card_obj.metadata
            else []
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
        common_misconceptions_str = strip_html_tags(
            ", ".join(common_misconceptions)
            if isinstance(common_misconceptions, list)
            else str(common_misconceptions)
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
            "Common_Misconceptions": common_misconceptions_str,
            "Difficulty": difficulty_str,
            "Source_URL": strip_html_tags(
                card_obj.metadata.get("source_url", "") if card_obj.metadata else ""
            ),
        }
        data_for_dataframe.append(card_dict)
    return data_for_dataframe
