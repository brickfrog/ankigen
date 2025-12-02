# Standard library imports
import asyncio
import os
import re
from datetime import datetime

import gradio as gr
import pandas as pd

from ankigen_core.card_generator import (
    AVAILABLE_MODELS,
    orchestrate_card_generation,
)  # GENERATION_MODES is internal to card_generator
from ankigen_core.exporters import (
    export_dataframe_to_apkg,
    export_dataframe_to_csv,
)  # Anki models (BASIC_MODEL, CLOZE_MODEL) are internal to exporters
from ankigen_core.llm_interface import (
    OpenAIClientManager,
)  # structured_output_completion is internal to core modules
from ankigen_core.ui_logic import update_mode_visibility
from ankigen_core.utils import (
    ResponseCache,
    get_logger,
)  # fetch_webpage_text is used by card_generator
from ankigen_core.auto_config import AutoConfigService

# --- Initialization ---
logger = get_logger()
response_cache = ResponseCache()  # Initialize cache
client_manager = OpenAIClientManager()  # Initialize client manager

# Agent system is required

AGENTS_AVAILABLE_APP = True
logger.info("Agent system is available")

js_storage = """
async () => {
    const loadDecks = () => {
        const decks = localStorage.getItem('ankigen_decks');
        return decks ? JSON.parse(decks) : [];
    };
    const saveDecks = (decks) => {
        localStorage.setItem('ankigen_decks', JSON.stringify(decks));
    };
    window.loadStoredDecks = loadDecks;
    window.saveStoredDecks = saveDecks;
    return loadDecks();
}
"""

try:
    custom_theme = gr.themes.Soft().set(  # type: ignore
        body_background_fill="*background_fill_secondary",
        block_background_fill="*background_fill_primary",
        block_border_width="0",
        button_primary_background_fill="*primary_500",
        button_primary_text_color="white",
    )
except (AttributeError, ImportError):
    # Fallback for older gradio versions or when themes are not available
    custom_theme = None

# CSS for the interface (moved to module level for Gradio 6 compatibility)
custom_css = """
    #footer {display:none !important}
    .tall-dataframe {min-height: 500px !important}
    .contain {max-width: 100% !important; margin: auto;}
    .output-cards {border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);}
    .hint-text {font-size: 0.9em; color: #666; margin-top: 4px;}
    .export-group > .gradio-group { margin-bottom: 0 !important; padding-bottom: 5px !important; }
"""

# --- Example Data for Initialization ---
example_data = pd.DataFrame(
    [
        [
            "1.1",
            "SQL Basics",
            "basic",
            "What is a SELECT statement used for?",
            "Retrieving data from one or more database tables.",
            "The SELECT statement is the most common command in SQL...",
            "```sql\nSELECT column1, column2 FROM my_table WHERE condition;\n```",
            ["Understanding of database tables"],
            ["Retrieve specific data"],
            "beginner",
        ],
        [
            "2.1",
            "Python Fundamentals",
            "cloze",
            "The primary keyword to define a function in Python is {{c1::def}}.",
            "def",
            "Functions are defined using the `def` keyword...",
            """```python
def greet(name):
    print(f"Hello, {name}!")
```""",
            ["Basic programming concepts"],
            ["Define reusable blocks of code"],
            "beginner",
        ],
    ],
    columns=[
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
    ],
)
# -------------------------------------


# --- Helper function for log viewing (Subtask 15.5) ---
def get_recent_logs(logger_name="ankigen") -> str:
    """Fetches the most recent log entries from the current day's log file."""
    try:
        log_dir = os.path.join(os.path.expanduser("~"), ".ankigen", "logs")
        timestamp = datetime.now().strftime("%Y%m%d")
        # Use the logger_name parameter to construct the log file name
        log_file = os.path.join(log_dir, f"{logger_name}_{timestamp}.log")

        if os.path.exists(log_file):
            with open(log_file) as f:
                lines = f.readlines()
                # Display last N lines, e.g., 100
                return "\n".join(lines[-100:])  # Ensured this is standard newline
        return f"Log file for today ({log_file}) not found or is empty."
    except Exception as e:
        # Use the main app logger to log this error, but don't let it crash the UI
        # function
        logger.error(f"Error reading logs: {e}", exc_info=True)
        return f"Error reading logs: {e!s}"


def create_ankigen_interface():
    logger.info("Creating AnkiGen Gradio interface...")
    # Note: theme, css, and js moved to .launch() for Gradio 6 compatibility
    with gr.Blocks(title="AnkiGen") as ankigen:
        with gr.Column(elem_classes="contain"):
            gr.Markdown("# 📚 AnkiGen - Anki Card Generator")
            gr.Markdown("#### Generate Anki flashcards using AI.")

            with gr.Accordion("Configuration Settings", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        generation_mode = gr.Radio(
                            choices=[
                                ("Single Subject", "subject"),
                            ],
                            value="subject",
                            label="Generation Mode",
                            info="Choose how you want to generate content",
                            visible=False,  # Hidden since only one mode exists
                        )
                        with gr.Group() as subject_mode:
                            subject = gr.Textbox(
                                label="Subject",
                                placeholder="e.g., 'Basic SQL Concepts'",
                            )
                            auto_fill_btn = gr.Button(
                                "Auto-fill",
                                variant="secondary",
                            )
                        api_key_input = gr.Textbox(
                            label="OpenAI API Key",
                            type="password",
                            placeholder="Enter your OpenAI API key (sk-...)",
                            value=os.getenv("OPENAI_API_KEY", ""),
                            info="Your key is used solely for processing your requests.",
                            elem_id="api-key-textbox",
                        )

                        # Context7 Library Documentation
                        library_accordion = gr.Accordion(
                            "Library Documentation (optional)", open=False
                        )
                        with library_accordion:
                            library_name_input = gr.Textbox(
                                label="Library Name",
                                placeholder="e.g., 'react', 'tensorflow', 'pandas'",
                                info="Fetch up-to-date documentation for this library",
                            )
                            library_topic_input = gr.Textbox(
                                label="Documentation Focus (optional)",
                                placeholder="e.g., 'hooks', 'data loading', 'transforms'",
                                info="Specific topic within the library to focus on",
                            )
                    with gr.Column(scale=1):
                        with gr.Accordion("Advanced Settings", open=False):
                            model_choices_ui = [
                                (m["label"], m["value"]) for m in AVAILABLE_MODELS
                            ]
                            default_model_value = next(
                                (
                                    m["value"]
                                    for m in AVAILABLE_MODELS
                                    if "nano" in m["value"].lower()
                                ),
                                AVAILABLE_MODELS[0]["value"],
                            )
                            model_choice = gr.Dropdown(
                                choices=model_choices_ui,
                                value=default_model_value,
                                label="Model Selection",
                                info="Select AI model for generation",
                                allow_custom_value=True,
                            )
                            _model_info = gr.Markdown(
                                "**gpt-4.1**: Best quality | **gpt-4.1-nano**: Faster/Cheaper",
                            )
                            topic_number = gr.Slider(
                                label="Number of Topics",
                                minimum=2,
                                maximum=20,
                                step=1,
                                value=2,
                            )
                            cards_per_topic = gr.Slider(
                                label="Cards per Topic",
                                minimum=2,
                                maximum=30,
                                step=1,
                                value=3,
                            )
                            preference_prompt = gr.Textbox(
                                label="Learning Preferences",
                                placeholder="e.g., 'Beginner focus'",
                                lines=3,
                            )
                            generate_cloze_checkbox = gr.Checkbox(
                                label="Generate Cloze Cards (Experimental)",
                                value=False,
                            )
                            gr.Markdown(
                                "*Cards are generated by the subject expert agent with a quick self-review to catch obvious gaps.*"
                            )

            generate_button = gr.Button("Generate Cards", variant="primary")

            with gr.Group() as cards_output:
                gr.Markdown("### Generated Cards")
                with gr.Accordion("Output Format", open=False):
                    gr.Markdown(
                        "Cards: Index, Topic, Type, Q, A, Explanation, Example, Prerequisites, Outcomes, Difficulty. Export: CSV, .apkg",
                    )
                    with gr.Accordion("Example Card Format", open=False):
                        gr.Code(
                            label="Example Card",
                            value='{"front": ..., "back": ..., "metadata": ...}',
                            language="json",
                        )
                output = gr.DataFrame(
                    value=example_data,
                    headers=[
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
                    ],
                    datatype=[
                        "number",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                    ],
                    interactive=True,
                    elem_classes="tall-dataframe",
                    wrap=True,
                    column_widths=[
                        50,
                        100,
                        80,
                        200,
                        200,
                        250,
                        200,
                        150,
                        150,
                        100,
                    ],
                )
                total_cards_html = gr.HTML(
                    value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
                    visible=False,
                )

                # Token usage display
                token_usage_html = gr.HTML(
                    value="<div style='margin-top: 8px;'><b>Token Usage:</b> <span id='token-usage-display'>No usage data</span></div>",
                    visible=True,
                )

                # Export buttons
                with gr.Row(elem_classes="export-group"):
                    export_csv_button = gr.Button("Export to CSV")
                    export_apkg_button = gr.Button("Export to .apkg")
                download_file_output = gr.File(label="Download Deck", visible=False)

            # --- Event Handlers --- (Updated to use functions from ankigen_core)
            generation_mode.change(
                fn=update_mode_visibility,
                inputs=[
                    generation_mode,
                    subject,
                ],
                outputs=[
                    subject_mode,
                    cards_output,
                    subject,
                    output,
                    total_cards_html,
                ],
            )

            # Define an async wrapper for the orchestrate_card_generation
            async def handle_generate_click(
                api_key_input_val,
                subject_val,
                generation_mode_val,
                model_choice_val,
                topic_number_val,
                cards_per_topic_val,
                preference_prompt_val,
                generate_cloze_checkbox_val,
                library_name_val,
                library_topic_val,
                progress=gr.Progress(track_tqdm=True),
            ):
                return await orchestrate_card_generation(
                    client_manager,
                    response_cache,
                    api_key_input_val,
                    subject_val,
                    generation_mode_val,
                    "",  # source_text - deprecated
                    "",  # url_input - deprecated
                    model_choice_val,
                    topic_number_val,
                    cards_per_topic_val,
                    preference_prompt_val,
                    generate_cloze_checkbox_val,
                    library_name=library_name_val if library_name_val else None,
                    library_topic=library_topic_val if library_topic_val else None,
                )

            generate_button.click(
                fn=handle_generate_click,
                inputs=[
                    api_key_input,
                    subject,
                    generation_mode,
                    model_choice,
                    topic_number,
                    cards_per_topic,
                    preference_prompt,
                    generate_cloze_checkbox,
                    library_name_input,
                    library_topic_input,
                ],
                outputs=[output, total_cards_html, token_usage_html],
                show_progress="full",
            )

            # Define handler for CSV export (similar to APKG)
            async def handle_export_dataframe_to_csv_click(df: pd.DataFrame):
                if df is None or df.empty:
                    gr.Warning("No cards generated to export to CSV.")
                    return gr.update(value=None, visible=False)

                try:
                    # export_dataframe_to_csv from exporters.py returns a relative path
                    # or a filename if no path was part of its input.
                    # It already handles None input for filename_suggestion.
                    exported_path_relative = await asyncio.to_thread(
                        export_dataframe_to_csv,
                        df,
                        filename_suggestion="ankigen_cards.csv",
                    )

                    if exported_path_relative:
                        exported_path_absolute = os.path.abspath(exported_path_relative)
                        gr.Info(
                            f"CSV ready for download: {os.path.basename(exported_path_absolute)}",
                        )
                        return gr.update(value=exported_path_absolute, visible=True)
                    # This case might happen if export_dataframe_to_csv itself had an internal issue
                    # and returned None, though it typically raises an error or returns path.
                    gr.Warning("CSV export failed or returned no path.")
                    return gr.update(value=None, visible=False)
                except Exception as e:
                    logger.error(
                        f"Error exporting DataFrame to CSV: {e}",
                        exc_info=True,
                    )
                    gr.Error(f"Failed to export to CSV: {e!s}")
                    return gr.update(value=None, visible=False)

            export_csv_button.click(
                fn=handle_export_dataframe_to_csv_click,  # Use the new handler
                inputs=[output],
                outputs=[download_file_output],
                api_name="export_main_to_csv",
            )

            # Define handler for APKG export from DataFrame (Item 5)
            async def handle_export_dataframe_to_apkg_click(
                df: pd.DataFrame,
                subject_for_deck_name: str,
            ):
                if df is None or df.empty:
                    gr.Warning("No cards generated to export.")
                    return gr.update(value=None, visible=False)

                timestamp_for_name = datetime.now().strftime("%Y%m%d_%H%M%S")

                deck_name_inside_anki = (
                    "AnkiGen Exported Deck"  # Default name inside Anki
                )
                if subject_for_deck_name and subject_for_deck_name.strip():
                    clean_subject = re.sub(
                        r"[^a-zA-Z0-9\s_.-]",
                        "",
                        subject_for_deck_name.strip(),
                    )
                    deck_name_inside_anki = f"AnkiGen - {clean_subject}"
                elif not df.empty and "Topic" in df.columns and df["Topic"].iloc[0]:
                    first_topic = df["Topic"].iloc[0]
                    clean_first_topic = re.sub(
                        r"[^a-zA-Z0-9\s_.-]",
                        "",
                        str(first_topic).strip(),
                    )
                    deck_name_inside_anki = f"AnkiGen - {clean_first_topic}"
                else:
                    deck_name_inside_anki = f"AnkiGen Deck - {timestamp_for_name}"  # Fallback with timestamp

                # Construct the output filename and path
                # Use the deck_name_inside_anki for the base of the filename for consistency
                base_filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", deck_name_inside_anki)
                output_filename = f"{base_filename}_{timestamp_for_name}.apkg"

                output_dir = "output_decks"  # As defined in export_dataframe_to_apkg
                os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
                full_output_path = os.path.join(output_dir, output_filename)

                try:
                    # Call export_dataframe_to_apkg with correct arguments:
                    # 1. df (DataFrame)
                    # 2. output_path (full path for the .apkg file)
                    # 3. deck_name (name of the deck inside Anki)
                    exported_path_relative = await asyncio.to_thread(
                        export_dataframe_to_apkg,
                        df,
                        full_output_path,  # Pass the constructed full output path
                        deck_name_inside_anki,  # This is the name for the deck inside the .apkg file
                    )

                    # export_dataframe_to_apkg returns the actual path it used, which should match full_output_path
                    exported_path_absolute = os.path.abspath(exported_path_relative)

                    gr.Info(
                        f"Successfully exported deck '{deck_name_inside_anki}' to {exported_path_absolute}",
                    )
                    return gr.update(value=exported_path_absolute, visible=True)
                except Exception as e:
                    logger.error(
                        f"Error exporting DataFrame to APKG: {e}",
                        exc_info=True,
                    )
                    gr.Error(f"Failed to export to APKG: {e!s}")
                    return gr.update(value=None, visible=False)

            # Wire button to handler (Item 6)
            export_apkg_button.click(
                fn=handle_export_dataframe_to_apkg_click,
                inputs=[output, subject],  # Added subject as input
                outputs=[download_file_output],
                api_name="export_main_to_apkg",
            )

            # Auto-fill handler
            async def handle_auto_fill_click(
                subject_text: str,
                api_key: str,
                progress=gr.Progress(track_tqdm=True),
            ):
                """Handle auto-fill button click to populate all settings"""
                if not subject_text or not subject_text.strip():
                    gr.Warning("Please enter a subject first")
                    return [gr.update()] * 8  # Return no updates for all outputs

                if not api_key:
                    gr.Warning("OpenAI API key is required for auto-configuration")
                    return [gr.update()] * 8

                try:
                    progress(0, desc="Analyzing subject...")

                    # Initialize OpenAI client
                    await client_manager.initialize_client(api_key)
                    openai_client = client_manager.get_client()

                    # Get auto-configuration
                    auto_config_service = AutoConfigService()
                    config = await auto_config_service.auto_configure(
                        subject_text, openai_client
                    )

                    if not config:
                        gr.Warning("Could not generate configuration")
                        return [gr.update()] * 8

                    # Return updates for all relevant UI components
                    return (
                        gr.update(
                            value=config.get("library_name", "")
                        ),  # library_name_input
                        gr.update(
                            value=config.get("library_topic", "")
                        ),  # library_topic_input
                        gr.update(value=config.get("topic_number", 3)),  # topic_number
                        gr.update(
                            value=config.get("cards_per_topic", 5)
                        ),  # cards_per_topic
                        gr.update(
                            value=config.get("preference_prompt", "")
                        ),  # preference_prompt
                        gr.update(
                            value=config.get("generate_cloze_checkbox", False)
                        ),  # generate_cloze_checkbox
                        gr.update(
                            value=config.get("model_choice", "gpt-4.1-nano")
                        ),  # model_choice
                        gr.update(
                            open=True
                        ),  # Open the Library Documentation accordion
                    )

                except Exception as e:
                    logger.error(f"Auto-configuration failed: {e}", exc_info=True)
                    gr.Error(f"Auto-configuration failed: {str(e)}")
                    return [gr.update()] * 8

            auto_fill_btn.click(
                fn=handle_auto_fill_click,
                inputs=[subject, api_key_input],
                outputs=[
                    library_name_input,
                    library_topic_input,
                    topic_number,
                    cards_per_topic,
                    preference_prompt,
                    generate_cloze_checkbox,
                    model_choice,
                    library_accordion,
                ],
            )

    logger.info("AnkiGen Gradio interface creation complete.")
    return ankigen


# --- Main Execution --- (Runs if script is executed directly)
if __name__ == "__main__":
    import os

    try:
        ankigen_interface = create_ankigen_interface()
        logger.info("Launching AnkiGen Gradio interface...")

        # Configure for HuggingFace Spaces vs local development
        # Note: theme, css, js moved to launch() for Gradio 6 compatibility
        launch_kwargs = {
            "theme": custom_theme,
            "css": custom_css,
            "js": js_storage,
        }
        if os.environ.get("SPACE_ID"):  # On HuggingFace Spaces
            # Let HuggingFace handle all the configuration
            ankigen_interface.queue(default_concurrency_limit=2, max_size=10).launch(
                **launch_kwargs
            )
        else:  # Local development
            # Use auto port finding for local dev
            ankigen_interface.queue(default_concurrency_limit=2, max_size=10).launch(
                server_name="127.0.0.1", share=False, **launch_kwargs
            )
    except Exception as e:
        logger.critical(f"Failed to launch Gradio interface: {e}", exc_info=True)
