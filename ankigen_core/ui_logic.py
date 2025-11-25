# Module for functions that build or manage UI sections/logic

import gradio as gr
import pandas as pd  # Needed for use_selected_subjects type hinting
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
)
from urllib.parse import urlparse

# --- Imports moved from later in the file (Task 7, etc.) ---
import re  # For URL validation and filename sanitization
import asyncio

from ankigen_core.crawler import CrawledPage, WebCrawler
from ankigen_core.llm_interface import (
    OpenAIClientManager,
)
from ankigen_core.card_generator import (
    generate_cards_from_crawled_content,
    AVAILABLE_MODELS,
)
from ankigen_core.utils import get_logger

# Only import models that are actually used in this file
from ankigen_core.models import (
    Card,
    # ModelSettings, # Removed
    # LearningPathInput, # Removed
    # LearningPath, # Removed
    # GeneratedPath, # Removed
    # SubjectAnalysis, # Removed
    # SubjectCardRequest, # Removed
    # TextCardRequest, # Removed
    # LearningPathRequest, # Removed
)

# Import agent system for web crawling
# Agent system is required for web crawling
from ankigen_core.agents.integration import AgentOrchestrator

AGENTS_AVAILABLE_UI = True
# --- End moved imports ---

# Get an instance of the logger for this module
crawler_ui_logger = get_logger()  # Keep this definition


def update_mode_visibility(
    mode: str,
    current_subject: str,
    current_description: str,
    current_text: str,
    current_url: str,
):
    """Updates visibility and values of UI elements based on generation mode."""
    is_subject = mode == "subject"
    is_path = mode == "path"
    is_text = mode == "text"
    is_web = mode == "web"

    # Determine value persistence or clearing
    subject_val = current_subject if is_subject else ""
    description_val = current_description if is_path else ""
    text_val = current_text if is_text else ""
    url_val = current_url if is_web else ""

    cards_output_visible = is_subject or is_text or is_web

    # Define standard columns for empty DataFrames
    main_output_df_columns = [
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
    ]
    subjects_list_df_columns = ["Subject", "Prerequisites", "Time Estimate"]

    return (
        gr.update(visible=is_subject),  # 1 subject_mode (Group)
        gr.update(visible=is_path),  # 2 path_mode (Group)
        gr.update(visible=is_text),  # 3 text_mode (Group)
        gr.update(visible=is_web),  # 4 web_mode (Group for crawler UI)
        gr.update(visible=is_path),  # 5 path_results (Group)
        gr.update(
            visible=cards_output_visible
        ),  # 6 cards_output (Group for main table)
        gr.update(value=subject_val),  # Now 7th item (was 8th)
        gr.update(value=description_val),  # Now 8th item (was 9th)
        gr.update(value=text_val),  # Now 9th item (was 10th)
        gr.update(value=url_val),  # Now 10th item (was 11th)
        gr.update(
            value=pd.DataFrame(columns=main_output_df_columns)
        ),  # Now 11th item (was 12th)
        gr.update(
            value=pd.DataFrame(columns=subjects_list_df_columns)
        ),  # Now 12th item (was 13th)
        gr.update(value=""),  # Now 13th item (was 14th)
        gr.update(value=""),  # Now 14th item (was 15th)
        gr.update(
            value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
            visible=False,
        ),  # Now 15th item (was 16th)
    )


def use_selected_subjects(subjects_df: pd.DataFrame | None):
    """Updates UI to use subjects from learning path analysis."""
    if subjects_df is None or subjects_df.empty:
        gr.Warning("No subjects available to copy from Learning Path analysis.")
        # Return updates that change nothing for all 18 outputs
        return (
            gr.update(),  # 1 generation_mode
            gr.update(),  # 2 subject_mode
            gr.update(),  # 3 path_mode
            gr.update(),  # 4 text_mode
            gr.update(),  # 5 web_mode
            gr.update(),  # 6 path_results
            gr.update(),  # 7 cards_output
            gr.update(),  # 8 subject
            gr.update(),  # 9 description
            gr.update(),  # 10 source_text
            gr.update(),  # 11 web_crawl_url_input
            gr.update(),  # 12 topic_number
            gr.update(),  # 13 preference_prompt
            gr.update(
                value=pd.DataFrame(
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
                    ]
                )
            ),  # 14 output (DataFrame)
            gr.update(
                value=pd.DataFrame(
                    columns=["Subject", "Prerequisites", "Time Estimate"]
                )
            ),  # 15 subjects_list (DataFrame)
            gr.update(),  # 16 learning_order
            gr.update(),  # 17 projects
            gr.update(visible=False),  # 18 total_cards_html
        )

    try:
        subjects = subjects_df["Subject"].tolist()
        combined_subject = ", ".join(subjects)
        # Ensure suggested_topics is an int, Gradio sliders expect int/float for value
        suggested_topics = int(min(len(subjects) + 1, 20))
    except KeyError:
        gr.Error("Learning path analysis result is missing the 'Subject' column.")
        # Return no-change updates for all 18 outputs
        return (
            gr.update(),  # 1 generation_mode
            gr.update(),  # 2 subject_mode
            gr.update(),  # 3 path_mode
            gr.update(),  # 4 text_mode
            gr.update(),  # 5 web_mode
            gr.update(),  # 6 path_results
            gr.update(),  # 7 cards_output
            gr.update(),  # 8 subject
            gr.update(),  # 9 description
            gr.update(),  # 10 source_text
            gr.update(),  # 11 web_crawl_url_input
            gr.update(),  # 12 topic_number
            gr.update(),  # 13 preference_prompt
            gr.update(
                value=pd.DataFrame(
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
                    ]
                )
            ),  # 14 output (DataFrame)
            gr.update(
                value=pd.DataFrame(
                    columns=["Subject", "Prerequisites", "Time Estimate"]
                )
            ),  # 15 subjects_list (DataFrame)
            gr.update(),  # 16 learning_order
            gr.update(),  # 17 projects
            gr.update(visible=False),  # 18 total_cards_html
        )

    # Corresponds to outputs in app.py for use_subjects.click:
    # [generation_mode, subject_mode, path_mode, text_mode, web_mode, path_results, cards_output,
    #  subject, description, source_text, web_crawl_url_input, topic_number, preference_prompt,
    #  output, subjects_list, learning_order, projects, total_cards_html]
    return (
        gr.update(value="subject"),  # 1 generation_mode (Radio)
        gr.update(visible=True),  # 2 subject_mode (Group)
        gr.update(visible=False),  # 3 path_mode (Group)
        gr.update(visible=False),  # 4 text_mode (Group)
        gr.update(visible=False),  # 5 web_mode (Group)
        gr.update(visible=False),  # 6 path_results (Group)
        gr.update(visible=True),  # 7 cards_output (Group)
        gr.update(value=combined_subject),  # 8 subject (Textbox)
        gr.update(value=""),  # 9 description (Textbox)
        gr.update(value=""),  # 10 source_text (Textbox)
        gr.update(value=""),  # 11 web_crawl_url_input (Textbox)
        gr.update(value=suggested_topics),  # 12 topic_number (Slider)
        gr.update(
            value="Focus on connections between these subjects and their practical applications."
        ),  # 13 preference_prompt (Textbox)
        gr.update(
            value=pd.DataFrame(
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
                ]
            )
        ),  # 14 output (DataFrame) - Clear it
        gr.update(
            value=subjects_df
        ),  # 15 subjects_list (DataFrame) - Keep the value that triggered this
        gr.update(
            value=""
        ),  # 16 learning_order (Markdown) - Clear it or decide to keep
        gr.update(value=""),  # 17 projects (Markdown) - Clear it or decide to keep
        gr.update(visible=False),  # 18 total_cards_html (HTML)
    )


def create_crawler_main_mode_elements() -> Tuple[
    List[gr.components.Component],  # ui_components (url_input, max_depth, etc.)
    gr.Button,  # crawl_button
    gr.Progress,  # progress_bar
    gr.Textbox,  # progress_status_textbox
    gr.Textbox,  # custom_system_prompt
    gr.Textbox,  # custom_user_prompt_template
    gr.Checkbox,  # use_sitemap_checkbox
    gr.Textbox,  # sitemap_url_textbox
]:
    """Creates the UI components for the Web Crawler mode integrated into the main tab."""
    ui_components: List[gr.components.Component] = []

    # URL Input
    url_input = gr.Textbox(
        label="Start URL",
        placeholder="Enter the full URL to start crawling (e.g., https://example.com/docs)",
        elem_id="crawler_url_input",
    )
    ui_components.append(url_input)

    with gr.Row():
        max_depth_slider = gr.Slider(
            minimum=0,
            maximum=5,
            value=1,
            step=1,
            label="Max Crawl Depth",
            elem_id="crawler_max_depth_slider",
        )
        ui_components.append(max_depth_slider)

        crawler_req_per_sec_slider = gr.Slider(
            minimum=0.1,
            maximum=10,
            value=2,
            step=0.1,
            label="Requests per Second (Crawler)",
            elem_id="crawler_req_per_sec_slider",
        )
        ui_components.append(crawler_req_per_sec_slider)

    model_choices_ui_crawler = [(m["label"], m["value"]) for m in AVAILABLE_MODELS]
    default_model_value_crawler = next(
        (m["value"] for m in AVAILABLE_MODELS if "nano" in m["value"].lower()),
        AVAILABLE_MODELS[0]["value"] if AVAILABLE_MODELS else "",
    )
    model_dropdown = gr.Dropdown(
        choices=model_choices_ui_crawler,
        label="AI Model for Content Processing",  # Clarified label
        value=default_model_value_crawler,
        elem_id="crawler_model_dropdown",
        allow_custom_value=True,
    )
    ui_components.append(model_dropdown)

    with gr.Row():
        include_patterns_textbox = gr.Textbox(
            label="Include URL Patterns (one per line, regex compatible)",
            placeholder="""e.g., /blog/.*
example.com/articles/.*""",
            lines=3,
            elem_id="crawler_include_patterns",
            scale=1,
        )
        ui_components.append(include_patterns_textbox)

        exclude_patterns_textbox = gr.Textbox(
            label="Exclude URL Patterns (one per line, regex compatible)",
            placeholder="""e.g., /category/.*
.*/login""",
            lines=3,
            elem_id="crawler_exclude_patterns",
            scale=1,
        )
        ui_components.append(exclude_patterns_textbox)

    with gr.Accordion(
        "Sitemap Options", open=False, elem_id="crawler_sitemap_options_accordion"
    ):
        use_sitemap_checkbox = gr.Checkbox(
            label="Use Sitemap?",
            value=False,
            elem_id="crawler_use_sitemap_checkbox",
        )
        # ui_components.append(use_sitemap_checkbox) # Appended later with its group

        sitemap_url_textbox = gr.Textbox(
            label="Sitemap URL (e.g., /sitemap.xml or full URL)",
            placeholder="Enter sitemap URL relative to start URL or full path",
            visible=False,
            elem_id="crawler_sitemap_url_textbox",
        )
        # ui_components.append(sitemap_url_textbox) # Appended later with its group

        use_sitemap_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_sitemap_checkbox],
            outputs=[sitemap_url_textbox],
        )
    # Add sitemap components to the main list for return
    # sitemap_elements_for_return = [use_sitemap_checkbox, sitemap_url_textbox] # Unused variable

    with gr.Accordion(
        "Advanced Prompt Options",
        open=False,
        elem_id="crawler_advanced_options_accordion",
    ):  # Removed assignment to advanced_options_accordion_component
        custom_system_prompt = gr.Textbox(
            label="Custom System Prompt (Optional)",
            placeholder="Leave empty to use the default system prompt for card generation.",
            lines=5,
            info="Define the overall role and instructions for the AI.",
            elem_id="crawler_custom_system_prompt",
        )
        # ui_components.append(custom_system_prompt) # Appended later

        custom_user_prompt_template = gr.Textbox(
            label="Custom User Prompt Template (Optional)",
            placeholder="Leave empty to use default. Available placeholders: {url}, {content}",
            lines=5,
            info="Define how the page URL and content are presented to the AI.",
            elem_id="crawler_custom_user_prompt_template",
        )
        # ui_components.append(custom_user_prompt_template) # Appended later
    # Add prompt components to the main list for return
    # prompt_elements_for_return = [custom_system_prompt, custom_user_prompt_template] # Unused variable

    # Crawl button (will trigger crawl_and_generate, results populate main DataFrame)
    crawl_button = gr.Button(
        "Crawl Content & Prepare Cards",  # Changed button text
        variant="secondary",  # Differentiate from main generate button
        elem_id="crawler_crawl_content_button",
    )
    # ui_components.append(crawl_button) # Returned separately

    # Progress bar and status for the crawling process
    progress_bar = (
        gr.Progress()
    )  # Removed elem_id as gr.Progress might not support it directly
    progress_status_textbox = gr.Textbox(
        label="Crawl Status",
        interactive=False,
        lines=3,  # Reduced lines
        placeholder="Crawling process status will appear here...",
        elem_id="crawler_status_textbox",
    )
    # ui_components.append(progress_status_textbox) # Returned separately

    # REMOVED UI elements:
    # - export_format_radio (no longer needed here)
    # - All preview related: preview_row_component, preview_dataframe_component, update_cards_button_component
    # - All preview export related: export_format_preview_component, deck_name_preview_component, export_button_preview_component
    # - All direct file download related: download_row_group, generated_file_output, download_button

    # The main ui_components list should contain all elements whose values are needed as inputs to the crawl/generation
    # or whose visibility might be managed together.
    # For clarity, specific components like buttons or progress bars are returned separately if they have specific event handlers
    # or are managed distinctly.

    # Add all input fields to ui_components for easier management if needed, or return them individually.
    # For now, returning them grouped for clarity.

    return (
        ui_components,
        crawl_button,
        progress_bar,
        progress_status_textbox,
        custom_system_prompt,
        custom_user_prompt_template,
        use_sitemap_checkbox,
        sitemap_url_textbox,
    )


# --- Crawl and Generate Logic (Task 7) ---

# MODIFIED: Get model values from AVAILABLE_MODELS for validation
CRAWLER_AVAILABLE_MODELS_VALUES = [m["value"] for m in AVAILABLE_MODELS]


def _basic_sanitize_filename(name: str) -> str:
    """Basic filename sanitization by replacing non-alphanumeric characters with underscores."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def _validate_crawl_url(url: str) -> bool:
    """Validate URL for crawling."""
    if not url or not url.startswith(("http://", "https://")):
        gr.Warning("Invalid URL provided. Please enter a valid http/https URL.")
        return False
    try:
        urlparse(url)
        return True
    except Exception:
        return False


def _create_web_crawler(
    url: str,
    max_depth: int,
    include_patterns: str,
    exclude_patterns: str,
    use_sitemap: bool,
    sitemap_url_str: str,
) -> WebCrawler:
    """Create configured WebCrawler instance."""
    include_list = [p.strip() for p in include_patterns.split(",") if p.strip()]
    exclude_list = [p.strip() for p in exclude_patterns.split(",") if p.strip()]

    return WebCrawler(
        start_url=url,
        max_depth=max_depth,
        include_patterns=include_list,
        exclude_patterns=exclude_list,
        use_sitemap=use_sitemap,
        sitemap_url=sitemap_url_str
        if use_sitemap and sitemap_url_str.strip()
        else None,
    )


def _create_crawl_progress_callback(
    progress: gr.Progress,
) -> Tuple[Callable[[int, int, str], None], List[int]]:
    """Create progress callback for crawler with mutable state container."""
    total_urls_container = [0]  # Mutable container for nonlocal-like behavior

    def callback(processed_count: int, total_urls: int, current_url: str):
        total_urls_container[0] = total_urls
        if total_urls_container[0] > 0:
            progress(
                0.1 + (processed_count / total_urls_container[0]) * 0.4,
                desc=f"Crawling: {processed_count}/{total_urls_container[0]} URLs. Current: {current_url}",
            )
        else:
            progress(
                0.1 + processed_count * 0.01,
                desc=f"Crawling: {processed_count} URLs discovered. Current: {current_url}",
            )

    return callback, total_urls_container


async def _perform_web_crawl(
    crawler: WebCrawler,
    progress: gr.Progress,
    url: str,
) -> Optional[List[CrawledPage]]:
    """Execute web crawl and return pages or None if empty."""
    callback, _ = _create_crawl_progress_callback(progress)

    crawler_ui_logger.info(f"Starting crawl for {url}...")
    progress(0.15, desc=f"Starting crawl for {url}...")

    crawled_pages = await asyncio.to_thread(crawler.crawl, progress_callback=callback)

    crawler_ui_logger.info(f"Crawling finished. Found {len(crawled_pages)} pages.")
    progress(0.5, desc=f"Crawling finished. Found {len(crawled_pages)} pages.")

    return crawled_pages if crawled_pages else None


async def _process_crawled_with_agents(
    crawled_pages: List[CrawledPage],
    client_manager: OpenAIClientManager,
    url: str,
    progress: gr.Progress,
) -> Tuple[List[Card], str]:
    """Process crawled content with agent system."""
    crawler_ui_logger.info("Using agent system for web crawling card generation")

    orchestrator = AgentOrchestrator(client_manager)
    # API key is already configured in client_manager, pass empty string as placeholder
    await orchestrator.initialize("")

    combined_content = "\n\n--- PAGE BREAK ---\n\n".join(
        [
            f"URL: {page.url}\nTitle: {page.title}\nContent: {page.text_content[:2000]}..."
            for page in crawled_pages[:10]
        ]
    )

    context = {
        "source_text": combined_content,
        "crawl_source": url,
        "pages_crawled": len(crawled_pages),
    }

    progress(0.6, desc="Processing with agent system...")

    agent_cards, _ = await orchestrator.generate_cards_with_agents(
        topic=f"Content from {url}",
        subject="web_content",
        num_cards=min(len(crawled_pages) * 3, 50),
        difficulty="intermediate",
        enable_quality_pipeline=True,
        context=context,
    )

    if agent_cards:
        progress(0.9, desc=f"Agent system generated {len(agent_cards)} cards")
        final_message = (
            f"Agent system processed content from {len(crawled_pages)} pages. "
            f"Generated {len(agent_cards)} high-quality cards."
        )
    else:
        final_message = "Agent system returned no cards"

    return agent_cards or [], final_message


async def crawl_and_generate(
    url: str,
    max_depth: int,
    crawler_requests_per_second: float,
    include_patterns: str,
    exclude_patterns: str,
    model: str,
    export_format_ui: str,
    custom_system_prompt: str,
    custom_user_prompt_template: str,
    use_sitemap: bool,
    sitemap_url_str: str,
    client_manager: OpenAIClientManager,
    progress: gr.Progress,
    status_textbox: gr.Textbox,
) -> Tuple[str, List[dict], List[Card]]:
    """Crawls a website, generates Anki cards, and prepares them for export/display."""
    crawler_ui_logger.info(f"Crawl and generate called for URL: {url}")

    if not _validate_crawl_url(url):
        return "Invalid URL", [], []

    try:
        crawler = _create_web_crawler(
            url,
            max_depth,
            include_patterns,
            exclude_patterns,
            use_sitemap,
            sitemap_url_str,
        )

        crawled_pages = await _perform_web_crawl(crawler, progress, url)
        if not crawled_pages:
            progress(1.0, desc="No pages were crawled. Check URL and patterns.")
            return (
                "No pages were crawled. Check URL and patterns.",
                pd.DataFrame().to_dict(orient="records"),
                [],
            )

        agent_cards, final_message = await _process_crawled_with_agents(
            crawled_pages,
            client_manager,
            url,
            progress,
        )

        if agent_cards:
            cards_for_dataframe_export = generate_cards_from_crawled_content(
                agent_cards
            )
            progress(1.0, desc=final_message)
            return final_message, cards_for_dataframe_export, agent_cards
        else:
            progress(1.0, desc=final_message)
            return final_message, pd.DataFrame().to_dict(orient="records"), []

    except ConnectionError as e:
        crawler_ui_logger.error(f"Connection error during crawl: {e}", exc_info=True)
        progress(1.0, desc=f"Connection error: {e}")
        return f"Connection error: {e}", pd.DataFrame().to_dict(orient="records"), []
    except ValueError as e:
        crawler_ui_logger.error(f"Value error: {e}", exc_info=True)
        progress(1.0, desc=f"Input error: {e}")
        return f"Input error: {e}", pd.DataFrame().to_dict(orient="records"), []
    except RuntimeError as e:  # Catch RuntimeError from client_manager.get_client()
        crawler_ui_logger.error(
            f"Runtime error (e.g., OpenAI client not init): {e}", exc_info=True
        )
        progress(1.0, desc=f"Runtime error: {e}")
        return f"Runtime error: {e}", pd.DataFrame().to_dict(orient="records"), []
    except Exception as e:
        crawler_ui_logger.error(
            f"Unexpected error in crawl_and_generate: {e}", exc_info=True
        )
        progress(1.0, desc=f"Unexpected error: {e}")
        return (
            f"An unexpected error occurred: {e}",
            pd.DataFrame().to_dict(orient="records"),
            [],
        )


# --- Card Preview and Editing Utilities (Task 13.3) ---


def cards_to_dataframe(cards: List[Card]) -> pd.DataFrame:
    """Converts a list of Card objects to a Pandas DataFrame for UI display."""
    data_for_df = []
    for i, card in enumerate(cards):
        # Extract tags from metadata if they exist
        tags_list = card.metadata.get("tags", []) if card.metadata else []
        tags_str = ", ".join(tags_list) if tags_list else ""

        # Topic from metadata or a default
        topic_str = card.metadata.get("topic", "N/A") if card.metadata else "N/A"

        data_for_df.append(
            {
                "ID": i + 1,  # 1-indexed ID for display
                "Topic": topic_str,  # Added Topic
                "Front": card.front.question,
                "Back": card.back.answer,
                "Tags": tags_str,
                "Card Type": card.card_type or "Basic",  # Mapped from note_type
                "Explanation": card.back.explanation or "",  # Added Explanation
                "Example": card.back.example or "",  # Added Example
                "Source_URL": card.metadata.get("source_url", "")
                if card.metadata
                else "",  # Added Source URL
            }
        )
    # Define all columns explicitly for consistent DataFrame structure
    df_columns = [
        "ID",
        "Topic",
        "Front",
        "Back",
        "Tags",
        "Card Type",
        "Explanation",
        "Example",
        "Source_URL",
    ]
    df = pd.DataFrame(data_for_df, columns=df_columns)
    return df


def dataframe_to_cards(df: pd.DataFrame, original_cards: List[Card]) -> List[Card]:
    """
    Updates a list of Card objects based on edits from a Pandas DataFrame.
    Assumes the DataFrame 'ID' column corresponds to the 1-based index of original_cards.
    """
    updated_cards: List[Card] = []
    if df.empty and not original_cards:
        return []
    if df.empty and original_cards:
        return []  # Or original_cards if no change is intended on empty df

    for index, row in df.iterrows():
        try:
            card_id = int(row["ID"])  # DataFrame ID is 1-indexed
            original_card_index = card_id - 1

            if 0 <= original_card_index < len(original_cards):
                card_to_update = original_cards[original_card_index]

                # Create new CardFront and CardBack objects for immutability if preferred,
                # or update existing ones since Pydantic models are mutable.
                new_front = card_to_update.front.copy(
                    update={
                        "question": str(row.get("Front", card_to_update.front.question))
                    }
                )
                new_back = card_to_update.back.copy(
                    update={
                        "answer": str(row.get("Back", card_to_update.back.answer)),
                        "explanation": str(
                            row.get("Explanation", card_to_update.back.explanation)
                        ),
                        "example": str(row.get("Example", card_to_update.back.example)),
                    }
                )

                tags_str = str(
                    row.get(
                        "Tags",
                        ",".join(
                            card_to_update.metadata.get("tags", [])
                            if card_to_update.metadata
                            else []
                        ),
                    )
                )
                new_tags = [t.strip() for t in tags_str.split(",") if t.strip()]

                new_metadata = (
                    card_to_update.metadata.copy() if card_to_update.metadata else {}
                )
                new_metadata["tags"] = new_tags
                new_metadata["topic"] = str(
                    row.get("Topic", new_metadata.get("topic", "N/A"))
                )
                # Source URL is generally not editable from this simple table

                updated_card = card_to_update.copy(
                    update={
                        "front": new_front,
                        "back": new_back,
                        "card_type": str(
                            row.get("Card Type", card_to_update.card_type or "Basic")
                        ),
                        "metadata": new_metadata,
                    }
                )
                updated_cards.append(updated_card)
            else:
                crawler_ui_logger.warning(
                    f"Card ID {card_id} from DataFrame is out of bounds for original_cards list."
                )
        except (ValueError, KeyError, AttributeError) as e:
            crawler_ui_logger.error(
                f"Error processing row {index} from DataFrame: {row}. Error: {e}"
            )
            if 0 <= original_card_index < len(original_cards):
                updated_cards.append(
                    original_cards[original_card_index]
                )  # Re-add original on error
            continue
    return updated_cards
