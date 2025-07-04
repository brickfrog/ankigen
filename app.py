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
from ankigen_core.learning_path import analyze_learning_path
from ankigen_core.llm_interface import (
    OpenAIClientManager,
)  # structured_output_completion is internal to core modules
from ankigen_core.ui_logic import (
    crawl_and_generate,
    create_crawler_main_mode_elements,
    update_mode_visibility,
    use_selected_subjects,
)
from ankigen_core.utils import (
    ResponseCache,
    get_logger,
)  # fetch_webpage_text is used by card_generator

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
            """```python
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

            /* REMOVING CSS previously intended for DataFrame readability to ensure plain text */
            /*
            .explanation-text {
                background: #f0fdf4;
                border-left: 3px solid #4ade80;
                padding: 0.5em;
                margin-bottom: 0.5em;
                border-radius: 4px;
            }
            .example-text-plain {
                background: #fff7ed;
                border-left: 3px solid #f97316;
                padding: 0.5em;
                margin-bottom: 0.5em;
                border-radius: 4px;
            }
            pre code {
                display: block;
                padding: 0.8em;
                background: #1e293b;
                color: #e2e8f0;
                border-radius: 4px;
                overflow-x: auto;
                font-family: 'Fira Code', 'Consolas', monospace;
                font-size: 0.9em;
                margin-bottom: 0.5em;
            }
            */
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
                                "Analyze & Break Down",
                                variant="secondary",
                            )
                        with gr.Group(visible=False) as text_mode:
                            source_text = gr.Textbox(
                                label="Source Text",
                                placeholder="Paste text here...",
                                lines=15,
                            )
                        with gr.Group(visible=False) as web_mode:
                            # --- BEGIN INTEGRATED CRAWLER UI (Task 16) ---
                            logger.info(
                                "Setting up integrated Web Crawler UI elements...",
                            )
                            (
                                crawler_input_ui_elements,  # List of inputs like URL, depth, model, patterns
                                web_crawl_button,  # Specific button to trigger crawl
                                web_crawl_progress_bar,
                                web_crawl_status_textbox,
                                web_crawl_custom_system_prompt,
                                web_crawl_custom_user_prompt_template,
                                web_crawl_use_sitemap_checkbox,
                                web_crawl_sitemap_url_textbox,
                            ) = create_crawler_main_mode_elements()

                            # Unpack crawler_input_ui_elements for clarity and use
                            web_crawl_url_input = crawler_input_ui_elements[0]
                            web_crawl_max_depth_slider = crawler_input_ui_elements[1]
                            web_crawl_req_per_sec_slider = crawler_input_ui_elements[2]
                            web_crawl_model_dropdown = crawler_input_ui_elements[3]
                            web_crawl_include_patterns_textbox = (
                                crawler_input_ui_elements[4]
                            )
                            web_crawl_exclude_patterns_textbox = (
                                crawler_input_ui_elements[5]
                            )
                            # --- END INTEGRATED CRAWLER UI ---

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

                            # Agent System Controls (simplified since we're agent-only)
                            if AGENTS_AVAILABLE_APP:
                                # Hidden dropdown for compatibility - always set to agent_only
                                agent_mode_dropdown = gr.Dropdown(
                                    choices=[("Agent Only", "agent_only")],
                                    value="agent_only",
                                    label="Agent Mode",
                                    visible=False,
                                )

                                with gr.Accordion("Agent Configuration", open=False):
                                    gr.Markdown("**Core Generation Pipeline**")
                                    enable_subject_expert = gr.Checkbox(
                                        label="Subject Expert Agent",
                                        value=True,
                                        info="Domain-specific expertise",
                                    )
                                    enable_generation_coordinator = gr.Checkbox(
                                        label="Generation Coordinator",
                                        value=True,
                                        info="Orchestrates multi-agent generation",
                                    )

                                    gr.Markdown("**Quality Assurance**")
                                    enable_content_judge = gr.Checkbox(
                                        label="Content Accuracy Judge",
                                        value=True,
                                        info="Factual correctness validation",
                                    )
                                    enable_clarity_judge = gr.Checkbox(
                                        label="Clarity Judge",
                                        value=True,
                                        info="Language clarity and comprehension",
                                    )

                                    gr.Markdown("**Optional Enhancements**")
                                    enable_pedagogical_agent = gr.Checkbox(
                                        label="Pedagogical Agent",
                                        value=False,
                                        info="Educational effectiveness review",
                                    )
                                    enable_pedagogical_judge = gr.Checkbox(
                                        label="Pedagogical Judge",
                                        value=False,
                                        info="Learning theory compliance",
                                    )
                                    enable_enhancement_agent = gr.Checkbox(
                                        label="Enhancement Agent",
                                        value=False,
                                        info="Content enrichment and metadata",
                                    )

                                    with gr.Accordion(
                                        "🛠️ Agent Model Selection", open=False
                                    ):
                                        gr.Markdown("**Individual Agent Models**")

                                        # Generator models
                                        subject_expert_model = gr.Dropdown(
                                            choices=model_choices_ui,
                                            value="gpt-4.1",
                                            label="Subject Expert Model",
                                            info="Model for domain expertise",
                                            allow_custom_value=True,
                                        )
                                        generation_coordinator_model = gr.Dropdown(
                                            choices=model_choices_ui,
                                            value="gpt-4.1-nano",
                                            label="Generation Coordinator Model",
                                            info="Model for orchestration",
                                            allow_custom_value=True,
                                        )

                                        # Judge models
                                        content_judge_model = gr.Dropdown(
                                            choices=model_choices_ui,
                                            value="gpt-4.1",
                                            label="Content Accuracy Judge Model",
                                            info="Model for fact-checking",
                                            allow_custom_value=True,
                                        )
                                        clarity_judge_model = gr.Dropdown(
                                            choices=model_choices_ui,
                                            value="gpt-4.1-nano",
                                            label="Clarity Judge Model",
                                            info="Model for language clarity",
                                            allow_custom_value=True,
                                        )

                                        # Enhancement models
                                        pedagogical_agent_model = gr.Dropdown(
                                            choices=model_choices_ui,
                                            value="gpt-4.1",
                                            label="Pedagogical Agent Model",
                                            info="Model for educational theory",
                                            allow_custom_value=True,
                                        )
                                        enhancement_agent_model = gr.Dropdown(
                                            choices=model_choices_ui,
                                            value="gpt-4.1",
                                            label="Enhancement Agent Model",
                                            info="Model for content enrichment",
                                            allow_custom_value=True,
                                        )
                            else:
                                # Placeholder when agents not available
                                agent_mode_dropdown = gr.Dropdown(
                                    choices=[("Legacy Only", "legacy")],
                                    value="legacy",
                                    label="Agent Mode",
                                    info="Agent system not available",
                                    interactive=False,
                                )
                                enable_subject_expert = gr.Checkbox(
                                    value=False, visible=False
                                )
                                enable_generation_coordinator = gr.Checkbox(
                                    value=False, visible=False
                                )
                                enable_pedagogical_agent = gr.Checkbox(
                                    value=False, visible=False
                                )
                                enable_content_judge = gr.Checkbox(
                                    value=False, visible=False
                                )
                                enable_clarity_judge = gr.Checkbox(
                                    value=False, visible=False
                                )
                                enable_pedagogical_judge = gr.Checkbox(
                                    value=False, visible=False
                                )
                                enable_enhancement_agent = gr.Checkbox(
                                    value=False, visible=False
                                )

                                # Hidden model dropdowns for non-agent mode
                                subject_expert_model = gr.Dropdown(
                                    value="gpt-4.1", visible=False
                                )
                                generation_coordinator_model = gr.Dropdown(
                                    value="gpt-4.1-nano", visible=False
                                )
                                content_judge_model = gr.Dropdown(
                                    value="gpt-4.1", visible=False
                                )
                                clarity_judge_model = gr.Dropdown(
                                    value="gpt-4.1-nano", visible=False
                                )
                                pedagogical_agent_model = gr.Dropdown(
                                    value="gpt-4.1", visible=False
                                )
                                enhancement_agent_model = gr.Dropdown(
                                    value="gpt-4.1", visible=False
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
                    "*Click to copy subjects to main input*",
                    elem_classes="hint-text",
                )

            with gr.Group() as cards_output:
                gr.Markdown("### Generated Cards")
                with gr.Accordion("Output Format", open=False):
                    gr.Markdown(
                        "Cards: Index, Topic, Type, Q, A, Explanation, Example, Prerequisites, Outcomes, Misconceptions, Difficulty. Export: CSV, .apkg",
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
                        "Common_Misconceptions",
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
                    description,
                    source_text,
                    web_crawl_url_input,
                ],
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
                    web_crawl_url_input,
                    output,
                    subjects_list,
                    learning_order,
                    projects,
                    total_cards_html,
                ],
            )

            # Define an async wrapper for the analyze_learning_path partial
            async def handle_analyze_click(
                api_key_val,
                description_val,
                model_choice_val,
                progress=gr.Progress(track_tqdm=True),  # Added progress tracker
            ):
                try:
                    # Call analyze_learning_path directly, as client_manager and response_cache are in scope
                    return await analyze_learning_path(
                        client_manager,  # from global scope
                        response_cache,  # from global scope
                        api_key_val,
                        description_val,
                        model_choice_val,
                    )
                except gr.Error as e:  # Catch the specific Gradio error
                    logger.error(f"Learning path analysis failed: {e}", exc_info=True)
                    # Re-raise the error so Gradio displays it to the user
                    # And return appropriate empty updates for the outputs
                    # to prevent a subsequent Gradio error about mismatched return values.
                    gr.Error(str(e))  # This will be shown in the UI.
                    empty_subjects_df = pd.DataFrame(
                        columns=["Subject", "Prerequisites", "Time Estimate"],
                    )
                    return (
                        gr.update(
                            value=empty_subjects_df,
                        ),  # For subjects_list (DataFrame)
                        gr.update(value=""),  # For learning_order (Markdown)
                        gr.update(value=""),  # For projects (Markdown)
                    )

            analyze_button.click(
                fn=handle_analyze_click,  # MODIFIED: Use the new async handler
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
                    web_crawl_url_input,
                    topic_number,
                    preference_prompt,
                    output,
                    subjects_list,
                    learning_order,
                    projects,
                    total_cards_html,
                ],
            )

            # Define an async wrapper for the orchestrate_card_generation partial
            async def handle_generate_click(
                api_key_input_val,
                subject_val,
                generation_mode_val,
                source_text_val,
                url_input_val,
                model_choice_val,
                topic_number_val,
                cards_per_topic_val,
                preference_prompt_val,
                generate_cloze_checkbox_val,
                agent_mode_val,
                enable_subject_expert_val,
                enable_generation_coordinator_val,
                enable_pedagogical_agent_val,
                enable_content_judge_val,
                enable_clarity_judge_val,
                enable_pedagogical_judge_val,
                enable_enhancement_agent_val,
                subject_expert_model_val,
                generation_coordinator_model_val,
                content_judge_model_val,
                clarity_judge_model_val,
                pedagogical_agent_model_val,
                enhancement_agent_model_val,
                progress=gr.Progress(track_tqdm=True),  # Added progress tracker
            ):
                # Apply agent settings if agents are available
                if AGENTS_AVAILABLE_APP:
                    import os

                    # Set agent mode
                    os.environ["ANKIGEN_AGENT_MODE"] = agent_mode_val

                    # Set individual agent flags (using correct environment variable names)
                    os.environ["ANKIGEN_ENABLE_SUBJECT_EXPERT"] = str(
                        enable_subject_expert_val
                    ).lower()
                    os.environ["ANKIGEN_ENABLE_GENERATION_COORDINATOR"] = str(
                        enable_generation_coordinator_val
                    ).lower()
                    os.environ["ANKIGEN_ENABLE_PEDAGOGICAL_AGENT"] = str(
                        enable_pedagogical_agent_val
                    ).lower()
                    os.environ["ANKIGEN_ENABLE_CONTENT_JUDGE"] = str(
                        enable_content_judge_val
                    ).lower()
                    os.environ["ANKIGEN_ENABLE_CLARITY_JUDGE"] = str(
                        enable_clarity_judge_val
                    ).lower()
                    os.environ["ANKIGEN_ENABLE_PEDAGOGICAL_JUDGE"] = str(
                        enable_pedagogical_judge_val
                    ).lower()
                    os.environ["ANKIGEN_ENABLE_ENHANCEMENT_AGENT"] = str(
                        enable_enhancement_agent_val
                    ).lower()

                    # Enable additional required flags for proper agent coordination
                    os.environ["ANKIGEN_ENABLE_JUDGE_COORDINATOR"] = (
                        "true"  # Required for judge coordination
                    )
                    os.environ["ANKIGEN_ENABLE_PARALLEL_JUDGING"] = (
                        "true"  # Enable parallel judging for performance
                    )

                    # Configure agent models from UI selections
                    model_overrides = {
                        "subject_expert": subject_expert_model_val,
                        "generation_coordinator": generation_coordinator_model_val,
                        "content_accuracy_judge": content_judge_model_val,
                        "clarity_judge": clarity_judge_model_val,
                        "pedagogical_agent": pedagogical_agent_model_val,
                        "enhancement_agent": enhancement_agent_model_val,
                    }

                    # Template variables for Jinja rendering
                    template_vars = {
                        "subject": subject_val or "general studies",
                        "difficulty": "intermediate",  # Could be made configurable
                        "topic": subject_val or "general concepts",
                    }

                    # Initialize config manager with model overrides and template variables
                    from ankigen_core.agents.config import get_config_manager

                    get_config_manager(model_overrides, template_vars)

                    # Log the agent configuration
                    logger.info(f"Agent mode set to: {agent_mode_val}")
                    logger.info(f"Model overrides: {model_overrides}")
                    logger.info(
                        f"Active agents: Subject Expert={enable_subject_expert_val}, Generation Coordinator={enable_generation_coordinator_val}, Content Judge={enable_content_judge_val}, Clarity Judge={enable_clarity_judge_val}"
                    )

                    # Reload feature flags to pick up the new environment variables
                    try:
                        # Agent system is available
                        logger.info("Agent system enabled")
                    except Exception as e:
                        logger.warning(f"Failed to reload feature flags: {e}")

                # Recreate the partial function call, but now it can be awaited
                # The actual orchestrate_card_generation is already partially applied with client_manager and response_cache
                # So, we need to get that specific partial object if it's stored, or redefine the partial logic here.
                # For simplicity and clarity, let's assume direct call to orchestrate_card_generation directly here
                return await orchestrate_card_generation(
                    client_manager,  # from global scope
                    response_cache,  # from global scope
                    api_key_input_val,
                    subject_val,
                    generation_mode_val,
                    source_text_val,
                    url_input_val,
                    model_choice_val,
                    topic_number_val,
                    cards_per_topic_val,
                    preference_prompt_val,
                    generate_cloze_checkbox_val,
                )
                # Expect 3-tuple return (dataframe, total_cards_html, token_usage_html)

            generate_button.click(
                fn=handle_generate_click,  # MODIFIED: Use the new async handler
                inputs=[
                    api_key_input,
                    subject,
                    generation_mode,
                    source_text,
                    web_crawl_url_input,
                    model_choice,
                    topic_number,
                    cards_per_topic,
                    preference_prompt,
                    generate_cloze_checkbox,
                    agent_mode_dropdown,
                    enable_subject_expert,
                    enable_generation_coordinator,
                    enable_pedagogical_agent,
                    enable_content_judge,
                    enable_clarity_judge,
                    enable_pedagogical_judge,
                    enable_enhancement_agent,
                    subject_expert_model,
                    generation_coordinator_model,
                    content_judge_model,
                    clarity_judge_model,
                    pedagogical_agent_model,
                    enhancement_agent_model,
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

            async def handle_web_crawl_click(
                api_key_val: str,
                url: str,
                max_depth: int,
                req_per_sec: float,
                model: str,  # This is the model for LLM processing of crawled content
                include_patterns: str,
                exclude_patterns: str,
                custom_system_prompt: str,
                custom_user_prompt_template: str,
                use_sitemap: bool,
                sitemap_url: str,
                progress=gr.Progress(track_tqdm=True),
            ):
                progress(0, desc="Initializing web crawl...")
                yield {
                    web_crawl_status_textbox: gr.update(
                        value="Initializing web crawl...",
                    ),
                    output: gr.update(value=None),  # Clear main output table
                    total_cards_html: gr.update(
                        visible=False,
                        value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
                    ),
                }

                if not api_key_val:
                    logger.error("API Key is missing for web crawler operation.")
                    yield {
                        web_crawl_status_textbox: gr.update(
                            value="Error: OpenAI API Key is required.",
                        ),
                    }
                    return
                try:
                    await client_manager.initialize_client(api_key_val)
                except Exception as e:
                    logger.error(
                        f"Failed to initialize OpenAI client for crawler: {e}",
                        exc_info=True,
                    )
                    yield {
                        web_crawl_status_textbox: gr.update(
                            value=f"Error: Client init failed: {e!s}",
                        ),
                    }
                    return

                message, cards_list_of_dicts, _ = await crawl_and_generate(
                    url=url,
                    max_depth=max_depth,
                    crawler_requests_per_second=req_per_sec,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                    model=model,
                    export_format_ui="",  # No longer used for direct export from crawl_and_generate
                    custom_system_prompt=custom_system_prompt,
                    custom_user_prompt_template=custom_user_prompt_template,
                    use_sitemap=use_sitemap,
                    sitemap_url_str=sitemap_url,
                    client_manager=client_manager,  # Passed from global scope
                    progress=progress,  # Gradio progress object
                    status_textbox=web_crawl_status_textbox,  # Specific status textbox for crawl
                )

                if cards_list_of_dicts:
                    try:
                        # Convert List[Dict] to Pandas DataFrame for the main output component
                        preview_df_value = pd.DataFrame(cards_list_of_dicts)
                        # Ensure columns match the main output dataframe
                        # The `generate_cards_from_crawled_content` which produces `cards_list_of_dicts`
                        # should already format it correctly. If not, mapping is needed here.
                        # For now, assume it matches the main table structure expected by `gr.Dataframe(value=example_data)`

                        # Check if columns match example_data, if not, reorder/rename or log warning
                        if not preview_df_value.empty:
                            expected_cols = example_data.columns.tolist()
                            # Basic check, might need more robust mapping if structures differ significantly
                            if not all(
                                col in preview_df_value.columns for col in expected_cols
                            ):
                                logger.warning(
                                    "Crawled card data columns mismatch main output, attempting to use available data.",
                                )
                                # Potentially select only common columns or reindex if necessary
                                # For now, we'll pass it as is, Gradio might handle extra/missing cols gracefully or error.

                        num_cards = len(preview_df_value)
                        total_cards_update = f"<div><b>Total Cards Prepared from Crawl:</b> <span id='total-cards-count'>{num_cards}</span></div>"

                        yield {
                            web_crawl_status_textbox: gr.update(value=message),
                            output: gr.update(value=preview_df_value),
                            total_cards_html: gr.update(
                                visible=True,
                                value=total_cards_update,
                            ),
                        }
                    except Exception as e:
                        logger.error(
                            f"Error converting crawled cards to DataFrame: {e}",
                            exc_info=True,
                        )
                        yield {
                            web_crawl_status_textbox: gr.update(
                                value=f"{message} (Error displaying cards: {e!s})",
                            ),
                            output: gr.update(value=None),
                            total_cards_html: gr.update(visible=False),
                        }
                else:
                    yield {
                        web_crawl_status_textbox: gr.update(
                            value=message,
                        ),  # Message from crawl_and_generate (e.g. no cards)
                        output: gr.update(value=None),
                        total_cards_html: gr.update(visible=False),
                    }

            web_crawl_button.click(
                fn=handle_web_crawl_click,
                inputs=[
                    api_key_input,
                    web_crawl_url_input,
                    web_crawl_max_depth_slider,
                    web_crawl_req_per_sec_slider,
                    web_crawl_model_dropdown,  # Model for LLM processing of content
                    web_crawl_include_patterns_textbox,
                    web_crawl_exclude_patterns_textbox,
                    web_crawl_custom_system_prompt,
                    web_crawl_custom_user_prompt_template,
                    web_crawl_use_sitemap_checkbox,
                    web_crawl_sitemap_url_textbox,
                ],
                outputs=[
                    web_crawl_status_textbox,  # Specific status for crawl
                    output,  # Main output DataFrame
                    total_cards_html,  # Main total cards display
                ],
                # Removed progress_bar from outputs as it's handled by gr.Progress(track_tqdm=True)
            )

    logger.info("AnkiGen Gradio interface creation complete.")
    return ankigen


# --- Main Execution --- (Runs if script is executed directly)
if __name__ == "__main__":
    try:
        ankigen_interface = create_ankigen_interface()
        logger.info("Launching AnkiGen Gradio interface...")
        ankigen_interface.launch()
    except Exception as e:
        logger.critical(f"Failed to launch Gradio interface: {e}", exc_info=True)
