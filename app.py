# Standard library imports
import os
from pathlib import Path  # Potentially for favicon_path
from functools import partial  # Moved to utils

import gradio as gr
import pandas as pd

from ankigen_core.utils import (
    get_logger,
    ResponseCache,
)  # fetch_webpage_text is used by card_generator

from ankigen_core.llm_interface import (
    OpenAIClientManager,
)  # structured_output_completion is internal to core modules
from ankigen_core.card_generator import (
    orchestrate_card_generation,
    AVAILABLE_MODELS,
)  # GENERATION_MODES is internal to card_generator
from ankigen_core.learning_path import analyze_learning_path
from ankigen_core.exporters import (
    export_csv,
    export_deck,
)  # Anki models (BASIC_MODEL, CLOZE_MODEL) are internal to exporters
from ankigen_core.ui_logic import update_mode_visibility, use_selected_subjects

# --- Initialization ---
logger = get_logger()
response_cache = ResponseCache()  # Initialize cache
client_manager = OpenAIClientManager()  # Initialize client manager

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

custom_theme = gr.themes.Soft().set(
    body_background_fill="*background_fill_secondary",
    block_background_fill="*background_fill_primary",
    block_border_width="0",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
)

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
            ["❌ SELECT * is always efficient (Reality: Can be slow for large tables)"],
            "beginner",
        ],
        [
            "2.1",
            "Python Fundamentals",
            "cloze",
            "The primary keyword to define a function in Python is {{c1::def}}.",
            "def",
            "Functions are defined using the `def` keyword...",
            r"""```python
def greet(name):
    print(f"Hello, {name}!")
```""",
            ["Basic programming concepts"],
            ["Define reusable blocks of code"],
            ["❌ Forgetting the colon (:) after the definition"],
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
        "Common_Misconceptions",
        "Difficulty",
    ],
)
# -------------------------------------


def create_ankigen_interface():
    logger.info("Creating AnkiGen Gradio interface...")
    with gr.Blocks(
        theme=custom_theme,
        title="AnkiGen",
        css="""
            #footer {display:none !important}
            .tall-dataframe {min-height: 500px !important}
            .contain {max-width: 100% !important; margin: auto;}
            .output-cards {border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);}
            .hint-text {font-size: 0.9em; color: #666; margin-top: 4px;}
            .export-group > .gradio-group { margin-bottom: 0 !important; padding-bottom: 5px !important; }
        """,
        js=js_storage,
    ) as ankigen:
        with gr.Column(elem_classes="contain"):
            gr.Markdown("# 📚 AnkiGen - Advanced Anki Card Generator")
            gr.Markdown("#### Generate comprehensive Anki flashcards using AI.")

            with gr.Accordion("Configuration Settings", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        generation_mode = gr.Radio(
                            choices=[
                                ("Single Subject", "subject"),
                                ("Learning Path", "path"),
                                ("From Text", "text"),
                                ("From Web", "web"),
                            ],
                            value="subject",
                            label="Generation Mode",
                            info="Choose how you want to generate content",
                        )
                        with gr.Group() as subject_mode:
                            subject = gr.Textbox(
                                label="Subject",
                                placeholder="e.g., 'Basic SQL Concepts'",
                            )
                        with gr.Group(visible=False) as path_mode:
                            description = gr.Textbox(
                                label="Learning Goal",
                                placeholder="Paste a job description...",
                                lines=5,
                            )
                            analyze_button = gr.Button(
                                "Analyze & Break Down", variant="secondary"
                            )
                        with gr.Group(visible=False) as text_mode:
                            source_text = gr.Textbox(
                                label="Source Text",
                                placeholder="Paste text here...",
                                lines=15,
                            )
                        with gr.Group(visible=False) as web_mode:
                            url_input = gr.Textbox(
                                label="Web Page URL", placeholder="Paste URL here..."
                            )
                        api_key_input = gr.Textbox(
                            label="OpenAI API Key",
                            type="password",
                            placeholder="Enter your OpenAI API key (sk-...)",
                            value=os.getenv("OPENAI_API_KEY", ""),
                            info="Your key is used solely for processing your requests.",
                            elem_id="api-key-textbox",
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
                            )
                            _model_info = gr.Markdown(
                                "**gpt-4.1**: Best quality | **gpt-4.1-nano**: Faster/Cheaper"
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
                                label="Generate Cloze Cards (Experimental)", value=False
                            )

            generate_button = gr.Button("Generate Cards", variant="primary")

            with gr.Group(visible=False) as path_results:
                gr.Markdown("### Learning Path Analysis")
                subjects_list = gr.Dataframe(
                    headers=["Subject", "Prerequisites", "Time Estimate"],
                    label="Recommended Subjects",
                    interactive=False,
                )
                learning_order = gr.Markdown("### Recommended Learning Order")
                projects = gr.Markdown("### Suggested Projects")
                use_subjects = gr.Button("Use These Subjects ℹ️", variant="primary")
                gr.Markdown(
                    "*Click to copy subjects to main input*", elem_classes="hint-text"
                )

            with gr.Group() as cards_output:
                gr.Markdown("### Generated Cards")
                with gr.Accordion("Output Format", open=False):
                    gr.Markdown(
                        "Cards: Index, Topic, Type, Q, A, Explanation, Example, Prerequisites, Outcomes, Misconceptions, Difficulty. Export: CSV, .apkg"
                    )
                    with gr.Accordion("Example Card Format", open=False):
                        gr.Code(
                            label="Example Card",
                            value='{"front": ..., "back": ..., "metadata": ...}',
                            language="json",
                        )
                output = gr.Dataframe(
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
                        "Common_Misconceptions",
                        "Difficulty",
                    ],
                    interactive=True,
                    elem_classes="tall-dataframe",
                    wrap=True,
                    column_widths=[50, 100, 80, 200, 200, 250, 200, 150, 150, 150, 100],
                )
                with gr.Group(elem_classes="export-group"):
                    gr.Markdown("#### Export Generated Cards")
                    with gr.Row():
                        export_csv_button = gr.Button(
                            "Export to CSV", variant="secondary"
                        )
                        export_anki_button = gr.Button(
                            "Export to Anki Deck (.apkg)", variant="secondary"
                        )
                    with gr.Row():
                        download_csv = gr.File(label="Download CSV", interactive=False)
                        download_anki = gr.File(
                            label="Download Anki Deck", interactive=False
                        )

            with gr.Row():
                progress = gr.HTML(visible=False)
                total_cards = gr.Number(
                    label="Total Cards Generated", value=0, visible=False
                )

            # --- Event Handlers --- (Updated to use functions from ankigen_core)
            generation_mode.change(
                fn=update_mode_visibility,
                inputs=[generation_mode, subject, description, source_text, url_input],
                outputs=[
                    subject_mode,
                    path_mode,
                    text_mode,
                    web_mode,
                    path_results,
                    cards_output,
                    subject,
                    description,
                    source_text,
                    url_input,
                    output,
                    subjects_list,
                    learning_order,
                    projects,
                    progress,
                    total_cards,
                ],
            )

            analyze_button.click(
                fn=partial(analyze_learning_path, client_manager, response_cache),
                inputs=[
                    api_key_input,
                    description,
                    model_choice,
                ],
                outputs=[subjects_list, learning_order, projects],
            )

            use_subjects.click(
                fn=use_selected_subjects,
                inputs=[subjects_list],
                outputs=[
                    generation_mode,
                    subject_mode,
                    path_mode,
                    text_mode,
                    web_mode,
                    path_results,
                    cards_output,
                    subject,
                    description,
                    source_text,
                    url_input,
                    topic_number,
                    preference_prompt,
                    output,
                    subjects_list,
                    learning_order,
                    projects,
                    progress,
                    total_cards,
                ],
            )

            generate_button.click(
                fn=partial(orchestrate_card_generation, client_manager, response_cache),
                inputs=[
                    api_key_input,
                    subject,
                    generation_mode,
                    source_text,
                    url_input,
                    model_choice,
                    topic_number,
                    cards_per_topic,
                    preference_prompt,
                    generate_cloze_checkbox,
                ],
                outputs=[output, progress, total_cards],
                show_progress="full",
            )

            export_csv_button.click(
                fn=export_csv,
                inputs=[output],
                outputs=download_csv,
                show_progress="full",
            )

            export_anki_button.click(
                fn=export_deck,
                inputs=[output, subject],
                outputs=download_anki,
                show_progress="full",
            )

    logger.info("Gradio interface created.")
    return ankigen


# --- Main Execution --- (Runs if script is executed directly)
if __name__ == "__main__":
    try:
        ankigen_interface = create_ankigen_interface()
        logger.info("Launching AnkiGen Gradio interface...")
        # Ensure favicon.ico is in the same directory as app.py or provide correct path
        favicon_path = Path(__file__).parent / "favicon.ico"
        if favicon_path.exists():
            ankigen_interface.launch(share=False, favicon_path=str(favicon_path))
        else:
            logger.warning(
                f"Favicon not found at {favicon_path}, launching without it."
            )
            ankigen_interface.launch(share=False)
    except Exception as e:
        logger.critical(f"Failed to launch Gradio interface: {e}", exc_info=True)
