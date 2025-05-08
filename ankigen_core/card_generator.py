# Module for core card generation logic

import gradio as gr
import pandas as pd

# Imports from our core modules
from ankigen_core.utils import get_logger, ResponseCache, fetch_webpage_text
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


def generate_cards_batch(
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
        response = structured_output_completion(
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
                front=CardFront(**card_data["front"]),
                back=CardBack(**card_data["back"]),
                metadata=card_data.get("metadata", {}),
            )
            cards_list.append(card)

        return cards_list

    except Exception as e:
        logger.error(
            f"Failed to generate cards batch for {topic}: {str(e)}", exc_info=True
        )
        raise  # Re-raise for the main function to handle


def orchestrate_card_generation(  # Renamed from generate_cards
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
        client_manager.initialize_client(api_key_input)
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
        page_text_for_generation = ""

        # --- Web Mode ---
        if generation_mode == "web":
            logger.info("Orchestrator: Web Mode")
            if not url_input or not url_input.strip():
                gr.Error("URL is required for 'From Web' mode.")
                return (
                    pd.DataFrame(columns=get_dataframe_columns()),
                    "URL is required.",
                    0,
                )

            # Use imported fetch_webpage_text
            gr.Info(f"🕸️ Fetching content from {url_input}...")
            try:
                page_text_for_generation = fetch_webpage_text(url_input)
                if (
                    not page_text_for_generation
                ):  # Handle case where fetch is successful but returns no text
                    gr.Warning(
                        f"Could not extract meaningful text content from {url_input}. Please check the page or try another URL."
                    )
                    # Return empty results gracefully
                    return (
                        pd.DataFrame(columns=get_dataframe_columns()),
                        "No meaningful text extracted from URL.",
                        0,
                    )

                gr.Info(
                    f"✅ Successfully fetched text (approx. {len(page_text_for_generation)} chars). Starting AI generation..."
                )
            except (ConnectionError, ValueError, RuntimeError) as e:
                logger.error(f"Failed to fetch or process URL {url_input}: {e}")
                gr.Error(f"Failed to get content from URL: {e}")
                return (
                    pd.DataFrame(columns=get_dataframe_columns()),
                    "Failed to get content from URL.",
                    0,
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error fetching URL {url_input}: {e}", exc_info=True
                )
                gr.Error("An unexpected error occurred fetching the URL.")
                return (
                    pd.DataFrame(columns=get_dataframe_columns()),
                    "Unexpected error fetching URL.",
                    0,
                )

        # --- Text Mode ---
        elif generation_mode == "text":
            logger.info("Orchestrator: Text Input Mode")
            if not source_text or not source_text.strip():
                gr.Error("Source text is required for 'From Text' mode.")
                return (
                    pd.DataFrame(columns=get_dataframe_columns()),
                    "Source text is required.",
                    0,
                )
            page_text_for_generation = source_text
            gr.Info("🚀 Starting card generation from text...")

        # --- Generation from Text/Web Content --- (Common Logic)
        if generation_mode == "text" or generation_mode == "web":
            topic_name = (
                "From Web Content" if generation_mode == "web" else "From Text Input"
            )
            logger.info(f"Generating cards directly from content: {topic_name}")

            # Prepare prompts (Consider moving prompt templates to a constants file or dedicated module later)
            text_system_prompt = f"""
            You are an expert educator creating flashcards from provided text.
            Generate {cards_per_topic} clear, concise flashcards based *only* on the text given.
            Focus on key concepts, definitions, facts, or processes.
            Adhere to the user's learning preferences: {preference_prompt}
            Use the specified JSON output format.
            Format code examples with triple backticks (```).
            """
            json_structure_prompt = get_card_json_structure_prompt()
            cloze_instruction = get_cloze_instruction(generate_cloze)

            text_user_prompt = f"""
            Generate {cards_per_topic} flashcards based *only* on the following text:
            --- TEXT START ---
            {page_text_for_generation}
            --- TEXT END ---
            {cloze_instruction}
            {json_structure_prompt}
            """

            # Call LLM interface
            response = structured_output_completion(
                openai_client=openai_client,
                model=model,
                response_format={"type": "json_object"},
                system_prompt=text_system_prompt,
                user_prompt=text_user_prompt,
                cache=cache,
            )

            if not response or "cards" not in response:
                logger.error("Invalid cards response format from text/web generation.")
                gr.Error("Failed to generate cards from content. Please try again.")
                return (
                    pd.DataFrame(columns=get_dataframe_columns()),
                    "Failed to generate cards from content.",
                    0,
                )

            cards_data = response["cards"]
            card_list = process_raw_cards_data(cards_data)

            flattened_data.extend(
                format_cards_for_dataframe(card_list, topic_name, start_index=1)
            )
            total_cards_generated = len(flattened_data)
            gr.Info(
                f"✅ Generated {total_cards_generated} cards from the provided content."
            )

        # --- Subject Mode ---
        elif generation_mode == "subject":
            logger.info(f"Orchestrator: Subject Mode for {subject}")
            if not subject or not subject.strip():
                gr.Error("Subject is required for 'Single Subject' mode.")
                return (
                    pd.DataFrame(columns=get_dataframe_columns()),
                    "Subject is required.",
                    0,
                )

            gr.Info("🚀 Starting card generation for subject...")

            system_prompt = f"""
            You are an expert educator in {subject}. Create an optimized learning sequence.
            Break down {subject} into {topic_number} logical concepts/topics, ordered by difficulty.
            Keep in mind the user's preferences: {preference_prompt}
            """
            topic_prompt = f"""
            Generate the top {topic_number} important subjects/topics to know about {subject} 
            ordered by ascending difficulty (beginner to advanced).
            Return your response as a JSON object: {{"topics": [{{"name": "topic name", "difficulty": "beginner/intermediate/advanced", "description": "brief description"}}]}}
            """

            logger.info("Generating topics...")
            topics_response = structured_output_completion(
                openai_client=openai_client,
                model=model,
                response_format={"type": "json_object"},
                system_prompt=system_prompt,
                user_prompt=topic_prompt,
                cache=cache,
            )

            if not topics_response or "topics" not in topics_response:
                logger.error("Invalid topics response format")
                gr.Error("Failed to generate topics. Please try again.")
                return (
                    pd.DataFrame(columns=get_dataframe_columns()),
                    "Failed to generate topics.",
                    0,
                )

            topics = topics_response["topics"]
            gr.Info(
                f"✨ Generated {len(topics)} topics successfully! Now generating cards..."
            )

            # System prompt for card generation (reused for each batch)
            card_system_prompt = f"""
            You are an expert educator in {subject}, creating flashcards for specific topics.            
            Focus on clarity, accuracy, and adherence to the user's preferences: {preference_prompt}
            Format code examples with triple backticks (```).
            Use the specified JSON output format.
            """

            # Generate cards for each topic - Consider parallelization later if needed
            for i, topic_info in enumerate(topics):  # Use enumerate for proper indexing
                topic_name = topic_info.get("name", f"Topic {i + 1}")
                logger.info(f"Generating cards for topic: {topic_name}")
                try:
                    cards = generate_cards_batch(
                        openai_client=openai_client,
                        cache=cache,
                        model=model,
                        topic=topic_name,
                        num_cards=cards_per_topic,
                        system_prompt=card_system_prompt,
                        generate_cloze=generate_cloze,
                    )

                    if cards:
                        flattened_data.extend(
                            format_cards_for_dataframe(cards, topic_name, topic_index=i)
                        )
                        total_cards_generated += len(cards)
                        gr.Info(
                            f"✅ Generated {len(cards)} cards for {topic_name} (Total: {total_cards_generated})"
                        )
                    else:
                        gr.Warning(
                            f"⚠️ No cards generated for topic '{topic_name}' (API might have returned empty list)."
                        )

                except Exception as e:
                    logger.error(
                        f"Failed during card generation for topic {topic_name}: {e}",
                        exc_info=True,
                    )
                    gr.Warning(
                        f"Failed to generate cards for '{topic_name}'. Skipping."
                    )
                    continue  # Continue to the next topic
        else:
            logger.error(f"Invalid generation mode received: {generation_mode}")
            gr.Error(f"Unsupported generation mode selected: {generation_mode}")
            return pd.DataFrame(columns=get_dataframe_columns()), "Unsupported mode.", 0

        # --- Common Completion Logic ---
        logger.info(
            f"Card generation orchestration complete. Total cards: {total_cards_generated}"
        )
        final_html = f"""
        <div style="text-align: center">
            <p>✅ Generation complete!</p>
            <p>Total cards generated: {total_cards_generated}</p>
        </div>
        """

        # Create DataFrame
        df = pd.DataFrame(flattened_data, columns=get_dataframe_columns())
        return df, final_html, total_cards_generated

    except gr.Error as e:
        logger.warning(f"A Gradio error was raised and caught: {e}")
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during card generation orchestration: {e}", exc_info=True
        )
        gr.Error(f"An unexpected error occurred: {e}")
        return pd.DataFrame(columns=get_dataframe_columns()), "Unexpected error.", 0


# --- Helper Functions --- (Could be moved to utils or stay here if specific)


def get_cloze_instruction(generate_cloze: bool) -> str:
    if not generate_cloze:
        return ""
    return """
    Where appropriate, generate Cloze deletion cards.
    - For Cloze cards, set "card_type" to "cloze".
    - Format the question field using Anki's cloze syntax (e.g., "The capital of France is {{c1::Paris}}.").
    - The "answer" field should contain the full, non-cloze text or specific context for the cloze.
    - For standard question/answer cards, set "card_type" to "basic".
    """


def get_card_json_structure_prompt() -> str:
    return """
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


def process_raw_cards_data(cards_data: list) -> list[Card]:
    """Processes raw card data dicts into a list of Card Pydantic models."""
    cards_list = []
    for card_data in cards_data:
        # Basic validation (can be enhanced)
        if (
            not isinstance(card_data, dict)
            or "front" not in card_data
            or "back" not in card_data
        ):
            logger.warning(f"Skipping malformed card data: {card_data}")
            continue
        try:
            card = Card(
                card_type=card_data.get("card_type", "basic"),
                front=CardFront(**card_data["front"]),
                back=CardBack(**card_data["back"]),
                metadata=card_data.get("metadata", {}),
            )
            cards_list.append(card)
        except Exception as e:
            logger.warning(
                f"Skipping card due to Pydantic validation error: {e} | Data: {card_data}"
            )
    return cards_list


def format_cards_for_dataframe(
    cards: list[Card], topic_name: str, topic_index: int = 0, start_index: int = 1
) -> list:
    """Formats a list of Card objects into a list of lists for the DataFrame."""
    formatted_rows = []
    for card_idx, card in enumerate(cards, start=start_index):
        index_str = (
            f"{topic_index + 1}.{card_idx}" if topic_index >= 0 else f"{card_idx}"
        )
        metadata = card.metadata or {}
        row = [
            index_str,
            topic_name,
            card.card_type,
            card.front.question,
            card.back.answer,
            card.back.explanation,
            card.back.example,
            metadata.get("prerequisites", []),
            metadata.get("learning_outcomes", []),
            metadata.get("misconceptions", []),
            metadata.get("difficulty", "beginner"),
        ]
        formatted_rows.append(row)
    return formatted_rows


def get_dataframe_columns() -> list[str]:
    """Returns the standard list of columns for the results DataFrame."""
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
    ]
