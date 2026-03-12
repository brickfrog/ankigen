# Module for CSV and APKG export functions

import gradio as gr
import pandas as pd
import genanki
import random
import html
from typing import List, Dict, Any, Optional
import csv
from datetime import datetime
import os

from ankigen.utils import get_logger, strip_html_tags

logger = get_logger()


# --- Helper function for formatting fields ---
def _format_field_as_string(value: Any) -> str:
    if isinstance(value, list) or isinstance(value, tuple):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip()


def _generate_timestamped_filename(
    base_name: str, extension: str, include_timestamp: bool = True
) -> str:
    """Generate a filename with optional timestamp.

    Args:
        base_name: The base name for the file (without extension)
        extension: File extension (e.g., 'csv', 'apkg')
        include_timestamp: Whether to include timestamp in filename

    Returns:
        Generated filename with extension
    """
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.{extension}"
    return f"{base_name}.{extension}"


def _ensure_output_directory(filepath: str) -> None:
    """Ensure the output directory exists for the given filepath.

    Args:
        filepath: Full path to the file

    Creates the directory if it doesn't exist.
    """
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")


def _validate_non_empty_data(data: Any, data_type: str) -> None:
    """Validate that data is not empty.

    Args:
        data: The data to validate (list, DataFrame, etc.)
        data_type: Description of data type for error messages

    Raises:
        ValueError: If data is empty or None
    """
    if data is None:
        raise ValueError(f"No {data_type} provided to export.")
    if isinstance(data, list) and not data:
        raise ValueError(f"No {data_type} provided to export.")
    if isinstance(data, pd.DataFrame) and data.empty:
        raise ValueError(f"No {data_type} available to export.")


# --- Constants for APKG Generation (Subtask 10) ---
ANKI_BASIC_MODEL_NAME = "AnkiGen Basic"
ANKI_CLOZE_MODEL_NAME = "AnkiGen Cloze"

# It's good practice to generate unique IDs. These are examples.
# Real applications might use a persistent way to store/retrieve these if models are updated.
DEFAULT_BASIC_MODEL_ID = random.randrange(1 << 30, 1 << 31)
DEFAULT_CLOZE_MODEL_ID = random.randrange(1 << 30, 1 << 31)

# --- Shared CSS with dark mode support ---
CARD_CSS = """
    /* CSS Variables - Light Mode (default) */
    .card {
        --bg-card: #ffffff;
        --bg-answer: #f0f9ff;
        --bg-explanation: #f0fdf4;
        --bg-example: #fefce8;
        --bg-back-extra: #eef2ff;
        --bg-prereq: #f8fafc;
        --bg-code: #2d2d2d;

        --text-primary: #1a1a1a;
        --text-secondary: #4b5563;
        --text-muted: #666666;
        --text-heading: #1f2937;
        --text-code: #f8f8f2;

        --accent-blue: #2563eb;
        --accent-blue-light: #60a5fa;
        --accent-green: #4ade80;
        --accent-yellow: #facc15;
        --accent-indigo: #818cf8;
        --accent-red: #ef4444;

        --border-light: #e5e7eb;
        --border-dashed: #cbd5e1;

        --shadow: rgba(0, 0, 0, 0.05);
    }

    /* Dark Mode Overrides */
    .nightMode .card,
    .night_mode .card {
        --bg-card: #1e1e1e;
        --bg-answer: #1e293b;
        --bg-explanation: #14291a;
        --bg-example: #292518;
        --bg-back-extra: #1e1b2e;
        --bg-prereq: #262626;
        --bg-code: #0d0d0d;

        --text-primary: #e4e4e7;
        --text-secondary: #a1a1aa;
        --text-muted: #9ca3af;
        --text-heading: #f4f4f5;
        --text-code: #f8f8f2;

        --accent-blue: #60a5fa;
        --accent-blue-light: #93c5fd;
        --accent-green: #4ade80;
        --accent-yellow: #fde047;
        --accent-indigo: #a5b4fc;
        --accent-red: #f87171;

        --border-light: #3f3f46;
        --border-dashed: #52525b;

        --shadow: rgba(0, 0, 0, 0.3);
    }

    /* Base styles */
    .card {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        font-size: 16px;
        line-height: 1.6;
        color: var(--text-primary);
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background: var(--bg-card);
    }

    @media (max-width: 768px) {
        .card {
            font-size: 14px;
            padding: 15px;
        }
    }

    /* Question side */
    .question-side {
        position: relative;
        min-height: 200px;
    }

    .difficulty-indicator {
        position: absolute;
        top: 10px;
        right: 10px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }

    .difficulty-indicator.beginner { background: var(--accent-green); }
    .difficulty-indicator.intermediate { background: var(--accent-yellow); }
    .difficulty-indicator.advanced { background: var(--accent-red); }

    .question {
        font-size: 1.3em;
        font-weight: 600;
        color: var(--accent-blue);
        margin-bottom: 1.5em;
    }

    .prerequisites {
        margin-top: 1em;
        font-size: 0.9em;
        color: var(--text-muted);
    }

    .prerequisites-toggle {
        color: var(--accent-blue);
        cursor: pointer;
        text-decoration: underline;
    }

    .prerequisites-content {
        display: none;
        margin-top: 0.5em;
        padding: 0.5em;
        background: var(--bg-prereq);
        border-radius: 4px;
    }

    .prerequisites.show .prerequisites-content {
        display: block;
    }

    /* Answer side sections */
    .answer-section,
    .explanation-section,
    .example-section,
    .back-extra-section {
        margin: 1.5em 0;
        padding: 1.2em;
        border-radius: 8px;
        box-shadow: 0 2px 4px var(--shadow);
    }

    .answer-section {
        background: var(--bg-answer);
        border-left: 4px solid var(--accent-blue);
    }

    .back-extra-section {
        background: var(--bg-back-extra);
        border-left: 4px solid var(--accent-indigo);
    }

    .explanation-section {
        background: var(--bg-explanation);
        border-left: 4px solid var(--accent-green);
    }

    .example-section {
        background: var(--bg-example);
        border-left: 4px solid var(--accent-yellow);
    }

    .example-section pre {
        background-color: var(--bg-code);
        color: var(--text-code);
        padding: 1em;
        border-radius: 0.3em;
        overflow-x: auto;
        font-family: 'Consolas', 'Monaco', 'Menlo', monospace;
        font-size: 0.9em;
        line-height: 1.4;
    }

    .example-section code {
        font-family: 'Consolas', 'Monaco', 'Menlo', monospace;
    }

    .metadata-section {
        margin-top: 2em;
        padding-top: 1em;
        border-top: 1px solid var(--border-light);
        font-size: 0.9em;
        color: var(--text-secondary);
    }

    .metadata-section h3 {
        font-size: 1em;
        color: var(--text-heading);
        margin-bottom: 0.5em;
    }

    .metadata-section > div {
        margin-bottom: 0.8em;
    }

    .source-url a {
        color: var(--accent-blue);
        text-decoration: none;
    }
    .source-url a:hover {
        text-decoration: underline;
    }

    /* Cloze deletion styles */
    .cloze {
        font-weight: bold;
        color: var(--accent-blue);
    }

    /* General utility */
    hr {
        border: none;
        border-top: 1px dashed var(--border-dashed);
        margin: 1.5em 0;
    }

    /* Rich text field styling */
    .field ul, .field ol {
        margin-left: 1.5em;
        padding-left: 0.5em;
    }
    .field li {
        margin-bottom: 0.3em;
    }

    /* Responsive design */
    @media (max-width: 640px) {
        .answer-section,
        .explanation-section,
        .example-section,
        .back-extra-section {
            padding: 1em;
            margin: 1em 0;
        }
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .card {
        animation: fadeIn 0.3s ease-in-out;
    }
"""

# --- Full Model Definitions ---

BASIC_MODEL = genanki.Model(
    DEFAULT_BASIC_MODEL_ID,  # Use the generated ID
    ANKI_BASIC_MODEL_NAME,  # Use the constant name
    fields=[
        {"name": "Question"},
        {"name": "Answer"},
        {"name": "Explanation"},
        {"name": "Example"},
        {"name": "Prerequisites"},
        {"name": "Learning_Outcomes"},
        {"name": "Difficulty"},
        {"name": "SourceURL"},  # Added for consistency if used by template
        {"name": "TagsStr"},  # Added for consistency if used by template
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": """
            <div class=\"card question-side\">
                <div class=\"difficulty-indicator {{Difficulty}}\"></div>
                <div class=\"content\">
                    <div class=\"question\">{{Question}}</div>
                    <div class=\"prerequisites\" onclick=\"event.stopPropagation();\">
                        <div class=\"prerequisites-toggle\">Show Prerequisites</div>
                        <div class=\"prerequisites-content\">{{Prerequisites}}</div>
                    </div>
                </div>
            </div>
            <script>
                document.querySelector('.prerequisites-toggle').addEventListener('click', function(e) {
                    e.stopPropagation();
                    this.parentElement.classList.toggle('show');
                });
            </script>
            """,
            "afmt": """
            <div class=\"card answer-side\">
                <div class=\"content\">
                    <div class=\"question-section\">
                        <div class=\"question\">{{Question}}</div>
                        <div class=\"prerequisites\">
                            <strong>Prerequisites:</strong> {{Prerequisites}}
                        </div>
                    </div>
                    <hr>

                    <div class=\"answer-section\">
                        <h3>Answer</h3>
                        <div class=\"answer\">{{Answer}}</div>
                    </div>

                    <div class=\"explanation-section\">
                        <h3>Explanation</h3>
                        <div class=\"explanation-text\">{{Explanation}}</div>
                    </div>

                    <div class=\"example-section\">
                        <h3>Example</h3>
                        <div class=\"example-text\">{{Example}}</div>
                        <!-- Example field might contain pre/code or plain text -->
                        <!-- Handled by how HTML is put into the Example field -->
                    </div>

                    <div class=\"metadata-section\">
                        <div class=\"learning-outcomes\">
                            <h3>Learning Outcomes</h3>
                            <div>{{Learning_Outcomes}}</div>
                        </div>


                        <div class=\"difficulty\">
                            <h3>Difficulty Level</h3>
                            <div>{{Difficulty}}</div>
                        </div>
                        {{#SourceURL}}<div class=\"source-url\"><small>Source: <a href=\"{{SourceURL}}\">{{SourceURL}}</a></small></div>{{/SourceURL}}
                    </div>
                </div>
            </div>
            """,
        }
    ],
    css=CARD_CSS,
)

CLOZE_MODEL = genanki.Model(
    DEFAULT_CLOZE_MODEL_ID,  # Use the generated ID
    ANKI_CLOZE_MODEL_NAME,  # Use the constant name
    fields=[
        {"name": "Text"},
        {"name": "Back Extra"},
        {"name": "Explanation"},
        {"name": "Example"},
        {"name": "Prerequisites"},
        {"name": "Learning_Outcomes"},
        {"name": "Difficulty"},
        {"name": "SourceURL"},
        {"name": "TagsStr"},
    ],
    templates=[
        {
            "name": "Cloze Card",
            "qfmt": """
            <div class=\"card question-side\">
                <div class=\"difficulty-indicator {{Difficulty}}\"></div>
                <div class=\"content\">
                    <div class=\"question\">{{cloze:Text}}</div>
                    <div class=\"prerequisites\" onclick=\"event.stopPropagation();\">
                        <div class=\"prerequisites-toggle\">Show Prerequisites</div>
                        <div class=\"prerequisites-content\">{{Prerequisites}}</div>
                    </div>
                </div>
            </div>
            <script>
                document.querySelector('.prerequisites-toggle').addEventListener('click', function(e) {
                    e.stopPropagation();
                    this.parentElement.classList.toggle('show');
                });
            </script>
            """,
            "afmt": """
            <div class=\"card answer-side\">
                <div class=\"content\">
                    <div class=\"question-section\">
                        <div class=\"question\">{{cloze:Text}}</div>
                        <div class=\"prerequisites\">
                            <strong>Prerequisites:</strong> {{Prerequisites}}
                        </div>
                    </div>
                    <hr>

                    {{#Back Extra}}
                    <div class=\"back-extra-section\">
                        <h3>Additional Information</h3>
                        <div class=\"back-extra-text\">{{Back Extra}}</div>
                    </div>
                    {{/Back Extra}}

                    <div class=\"explanation-section\">
                        <h3>Explanation</h3>
                        <div class=\"explanation-text\">{{Explanation}}</div>
                    </div>

                    <div class=\"example-section\">
                        <h3>Example</h3>
                        <div class=\"example-text\">{{Example}}</div>
                    </div>

                    <div class=\"metadata-section\">
                        <div class=\"learning-outcomes\">
                            <h3>Learning Outcomes</h3>
                            <div>{{Learning_Outcomes}}</div>
                        </div>


                        <div class=\"difficulty\">
                            <h3>Difficulty Level</h3>
                            <div>{{Difficulty}}</div>
                        </div>
                        {{#SourceURL}}<div class=\"source-url\"><small>Source: <a href=\"{{SourceURL}}\">{{SourceURL}}</a></small></div>{{/SourceURL}}
                    </div>
                </div>
            </div>
            """,
        }
    ],
    css=CARD_CSS,
    model_type=1,  # Cloze model type
)


# --- Helper functions for APKG (Subtask 10) ---
def _get_or_create_model(
    model_id: int,
    name: str,
    fields: List[Dict[str, str]],
    templates: List[Dict[str, str]],
) -> genanki.Model:
    return genanki.Model(model_id, name, fields=fields, templates=templates)


# --- New CSV Exporter for List of Dictionaries ---


def export_cards_to_csv(
    cards: List[Dict[str, Any]], filename: Optional[str] = None
) -> str:
    """Export a list of card dictionaries to a CSV file.

    Args:
        cards: A list of dictionaries, where each dictionary represents a card
               and should contain 'front' and 'back' keys. Other keys like
               'tags' and 'note_type' are optional.
        filename: Optional. The desired filename/path for the CSV.
                  If None, a timestamped filename will be generated.

    Returns:
        The path to the generated CSV file.

    Raises:
        IOError: If there is an issue writing to the file.
        KeyError: If a card dictionary is missing essential keys like 'front' or 'back'.
        ValueError: If the cards list is empty or not provided.
    """
    # Validation using helper
    _validate_non_empty_data(cards, "cards")

    # Filename generation using helper
    if not filename:
        filename = _generate_timestamped_filename("ankigen_cards", "csv")
        logger.info(f"No filename provided, generated: {filename}")

    # Ensure output directory exists using helper
    _ensure_output_directory(filename)

    # Define the fieldnames expected in the CSV.
    fieldnames = ["front", "back", "tags", "note_type"]

    try:
        logger.info(f"Attempting to export {len(cards)} cards to {filename}")
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            for i, card in enumerate(cards):
                try:
                    # Ensure mandatory fields exist
                    if "front" not in card or "back" not in card:
                        raise KeyError(
                            f"Card at index {i} is missing 'front' or 'back' key."
                        )

                    row_to_write = {
                        "front": card["front"],
                        "back": card["back"],
                        "tags": card.get("tags", ""),
                        "note_type": card.get("note_type", "Basic"),
                    }
                    writer.writerow(row_to_write)
                except KeyError as e_inner:
                    logger.error(
                        f"Skipping card due to KeyError: {e_inner}. Card data: {card}"
                    )
                    continue
        logger.info(f"Successfully exported cards to {filename}")
        return filename
    except IOError as e_io:
        logger.error(f"IOError during CSV export to {filename}: {e_io}", exc_info=True)
        raise
    except Exception as e_general:
        logger.error(
            f"Unexpected error during CSV export to {filename}: {e_general}",
            exc_info=True,
        )
        raise


def export_cards_to_apkg(
    cards: List[Dict[str, Any]],
    filename: Optional[str] = None,
    deck_name: str = "Ankigen Generated Cards",
) -> str:
    """Exports a list of card dictionaries to an Anki .apkg file.

    Args:
        cards: List of dictionaries, where each dictionary represents a card.
               It's expected that these dicts are prepared by export_dataframe_to_apkg
               and contain keys like 'Question', 'Answer', 'Explanation', etc.
        filename: The full path (including filename) for the exported file.
                  If None, a default filename will be generated in the current directory.
        deck_name: The name of the deck if exporting to .apkg format.

    Returns:
        The path to the exported file.
    """
    logger.info(f"Starting APKG export for {len(cards)} cards to deck '{deck_name}'.")

    # Validation using helper - note this now raises ValueError instead of gr.Error
    _validate_non_empty_data(cards, "cards")

    # Filename generation using helper
    if not filename:
        filename = _generate_timestamped_filename("ankigen_deck", "apkg")
    elif not filename.lower().endswith(".apkg"):
        filename += ".apkg"

    # Ensure output directory exists using helper
    _ensure_output_directory(filename)

    anki_basic_model = BASIC_MODEL
    anki_cloze_model = CLOZE_MODEL

    deck_id = random.randrange(1 << 30, 1 << 31)
    anki_deck = genanki.Deck(deck_id, deck_name)

    notes_added_count = 0
    for card_dict in cards:
        note_type = card_dict.get("note_type", "Basic")
        tags_for_note_object = card_dict.get("tags_for_note_object", [])

        # Extract all potential fields, defaulting to empty strings
        # Security: Sanitize HTML to prevent XSS when viewing cards in Anki
        question = html.escape(card_dict.get("Question", ""))
        answer = html.escape(card_dict.get("Answer", ""))
        explanation = html.escape(card_dict.get("Explanation", ""))
        example = html.escape(card_dict.get("Example", ""))
        prerequisites = html.escape(card_dict.get("Prerequisites", ""))
        learning_outcomes = html.escape(card_dict.get("Learning_Outcomes", ""))
        difficulty = html.escape(card_dict.get("Difficulty", ""))
        source_url = html.escape(card_dict.get("SourceURL", ""))
        tags_str_field = html.escape(card_dict.get("TagsStr", ""))

        if not question:
            logger.error(
                f"SKIPPING CARD DUE TO EMPTY 'Question' (front/text) field. Card data: {card_dict}"
            )
            continue

        try:
            if note_type.lower() == "cloze":
                # CLOZE_MODEL fields
                note_fields = [
                    question,  # Text
                    answer,  # Back Extra
                    explanation,
                    example,
                    prerequisites,
                    learning_outcomes,
                    difficulty,
                    source_url,
                    tags_str_field,
                ]
                note = genanki.Note(
                    model=anki_cloze_model,
                    fields=note_fields,
                    tags=tags_for_note_object,
                )
            else:  # Basic
                # BASIC_MODEL fields
                note_fields = [
                    question,
                    answer,
                    explanation,
                    example,
                    prerequisites,
                    learning_outcomes,
                    difficulty,
                    source_url,
                    tags_str_field,
                ]
                note = genanki.Note(
                    model=anki_basic_model,
                    fields=note_fields,
                    tags=tags_for_note_object,
                )
            anki_deck.add_note(note)
            notes_added_count += 1
        except Exception as e:
            logger.error(
                f"Failed to create genanki.Note for card: {card_dict}. Error: {e}",
                exc_info=True,
            )
            logger.warning(f"Skipping card due to error: Question='{question[:50]}...'")

    if notes_added_count == 0:
        logger.error(
            "No valid notes could be created from the provided cards. APKG generation aborted."
        )
        raise gr.Error("Failed to create any valid Anki notes from the input.")

    logger.info(
        f"Added {notes_added_count} notes to deck '{deck_name}'. Proceeding to package."
    )

    # Package and write
    package = genanki.Package(anki_deck)
    try:
        package.write_to_file(filename)
        logger.info(f"Successfully exported Anki deck to {filename}")
    except Exception as e:
        logger.error(f"Failed to write .apkg file to {filename}: {e}", exc_info=True)
        raise IOError(f"Could not write .apkg file: {e}")

    return filename


def export_cards_from_crawled_content(
    cards: List[Dict[str, Any]],
    output_path: Optional[
        str
    ] = None,  # Changed from filename to output_path for clarity
    export_format: str = "csv",  # Added export_format parameter
    deck_name: str = "Ankigen Generated Cards",
) -> str:
    """Exports cards (list of dicts) to the specified format (CSV or APKG).

    Args:
        cards: List of dictionaries, where each dictionary represents a card.
               Expected keys: 'front', 'back'. Optional: 'tags' (space-separated string), 'source_url', 'note_type' ('Basic' or 'Cloze').
        output_path: The full path (including filename) for the exported file.
                     If None, a default filename will be generated in the current directory.
        export_format: The desired format, either 'csv' or 'apkg'.
        deck_name: The name of the deck if exporting to .apkg format.

    Returns:
        The path to the exported file.
    """
    if not cards:
        logger.warning("No cards provided to export_cards_from_crawled_content.")
        # MODIFIED: Raise error immediately if no cards, as per test expectation
        raise ValueError("No cards provided to export.")

    logger.info(
        f"Exporting {len(cards)} cards to format '{export_format}' with deck name '{deck_name}'."
    )

    if export_format.lower() == "csv":
        return export_cards_to_csv(cards, filename=output_path)
    elif export_format.lower() == "apkg":
        return export_cards_to_apkg(cards, filename=output_path, deck_name=deck_name)
    else:
        supported_formats = ["csv", "apkg"]
        logger.error(
            f"Unsupported export format: {export_format}. Supported formats: {supported_formats}"
        )
        # MODIFIED: Updated error message to include supported formats
        raise ValueError(
            f"Unsupported export format: {export_format}. Supported formats: {supported_formats}"
        )


# --- New DataFrame CSV Exporter (Subtask 11) ---
def export_dataframe_to_csv(
    data: Optional[pd.DataFrame],
    filename_suggestion: Optional[str] = "ankigen_cards.csv",
) -> Optional[str]:
    """Exports a Pandas DataFrame to a CSV file, designed for Gradio download.

    Args:
        data: The Pandas DataFrame to export.
        filename_suggestion: A suggestion for the base filename (e.g., from subject).

    Returns:
        The path to the temporary CSV file, or None if an error occurs or data is empty.
    """
    logger.info(
        f"Attempting to export DataFrame to CSV. Suggested filename: {filename_suggestion}"
    )

    # Validation using helper
    try:
        _validate_non_empty_data(data, "card data")
    except ValueError:
        logger.warning(
            "No data provided to export_dataframe_to_csv. Skipping CSV export."
        )
        raise gr.Error("No card data available")

    try:
        # Generate filename from suggestion
        base_name_from_suggestion = "ankigen_cards"  # Default base part

        # Sanitize and use the suggestion (e.g., subject name) if provided
        if filename_suggestion and isinstance(filename_suggestion, str):
            # Remove .csv if present, then sanitize
            processed_suggestion = filename_suggestion.removesuffix(".csv")
            safe_suggestion = (
                processed_suggestion.replace(" ", "_")
                .replace("/", "-")
                .replace("\\", "-")
            )
            if safe_suggestion:
                base_name_from_suggestion = f"ankigen_{safe_suggestion[:50]}"

        # Generate timestamped filename using helper
        final_filename = _generate_timestamped_filename(
            base_name_from_suggestion, "csv"
        )

        # Ensure output directory exists using helper
        _ensure_output_directory(final_filename)

        data.to_csv(final_filename, index=False)
        logger.info(f"Successfully exported DataFrame to CSV: {final_filename}")
        gr.Info(f"CSV ready for download: {os.path.basename(final_filename)}")
        return final_filename
    except Exception as e:
        logger.error(f"Error exporting DataFrame to CSV: {e}", exc_info=True)
        gr.Error(f"Error exporting DataFrame to CSV: {e}")
        return None


# --- New DataFrame to APKG Exporter (for Main Generator Tab) ---
def export_dataframe_to_apkg(
    df: pd.DataFrame,
    output_path: Optional[str],
    deck_name: str,
) -> str:
    """Exports a DataFrame of cards to an Anki .apkg file."""
    # Validation using helper
    _validate_non_empty_data(df, "cards in DataFrame")

    logger.info(
        f"Starting APKG export for DataFrame with {len(df)} rows to deck '{deck_name}'. Output: {output_path}"
    )

    cards_for_apkg: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            note_type_val = (
                _format_field_as_string(row.get("Card_Type", "Basic")) or "Basic"
            )
            topic = _format_field_as_string(row.get("Topic", ""))
            difficulty_raw = _format_field_as_string(row.get("Difficulty", ""))
            difficulty_plain_for_tag = strip_html_tags(difficulty_raw)

            tags_list_for_note_obj = []
            if topic:
                tags_list_for_note_obj.append(topic.replace(" ", "_").replace(",", "_"))
            if difficulty_plain_for_tag:
                safe_difficulty_tag = difficulty_plain_for_tag.replace(" ", "_")
                tags_list_for_note_obj.append(safe_difficulty_tag)

            tags_str_for_field = " ".join(tags_list_for_note_obj)

            card_data_for_note = {
                "note_type": note_type_val,
                "tags_for_note_object": tags_list_for_note_obj,
                "TagsStr": tags_str_for_field,
                "Question": _format_field_as_string(row.get("Question", "")),
                "Answer": _format_field_as_string(row.get("Answer", "")),
                "Explanation": _format_field_as_string(row.get("Explanation", "")),
                "Example": _format_field_as_string(row.get("Example", "")),
                "Prerequisites": _format_field_as_string(row.get("Prerequisites", "")),
                "Learning_Outcomes": _format_field_as_string(
                    row.get("Learning_Outcomes", "")
                ),
                "Difficulty": difficulty_raw,
                "SourceURL": _format_field_as_string(row.get("Source_URL", "")),
            }
            cards_for_apkg.append(card_data_for_note)
        except Exception as e:
            logger.error(
                f"Error processing DataFrame row for APKG: {row}. Error: {e}",
                exc_info=True,
            )
            continue

    if not cards_for_apkg:
        logger.error("No cards could be processed from DataFrame for APKG export.")
        raise ValueError("No processable cards found in DataFrame for APKG export.")

    return export_cards_to_apkg(
        cards_for_apkg, filename=output_path, deck_name=deck_name
    )


# --- Compatibility Exports for Tests and Legacy Code ---
# These aliases ensure that tests expecting these names will find them.

# Export functions under expected names
export_csv = (
    export_dataframe_to_csv  # Update this to export_dataframe_to_csv for compatibility
)


# MODIFIED: export_deck is now a wrapper to provide a default deck_name
def export_deck(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    deck_name: str = "Ankigen Generated Cards",
) -> str:
    """Alias for exporting a DataFrame to APKG, providing a default deck name."""
    if df is None or df.empty:
        logger.warning("export_deck called with None or empty DataFrame.")
        # Match the error type and message expected by tests
        raise gr.Error("No card data available")

    # Original logic to call export_dataframe_to_apkg
    # Ensure all necessary parameters for export_dataframe_to_apkg are correctly passed.
    # The export_dataframe_to_apkg function itself will handle its specific error conditions.
    # The 'output_path' for export_dataframe_to_apkg needs to be handled.
    # If 'output_path' is None here, export_cards_to_apkg (called by export_dataframe_to_apkg)
    # will generate a default filename.

    # If output_path is not provided to export_deck, it's None.
    # export_dataframe_to_apkg expects output_path: Optional[str].
    # And export_cards_to_apkg (which it calls) also handles Optional[str] filename.
    # So, passing output_path directly should be fine.

    return export_dataframe_to_apkg(df, output_path=output_path, deck_name=deck_name)


export_dataframe_csv = export_dataframe_to_csv
export_dataframe_apkg = export_dataframe_to_apkg

__all__ = [
    "BASIC_MODEL",
    "CLOZE_MODEL",
    "export_csv",
    "export_deck",
    "export_dataframe_csv",
    "export_dataframe_apkg",
    "export_cards_to_csv",
    "export_cards_to_apkg",
    "export_cards_from_crawled_content",
    "export_dataframe_to_csv",
    "export_dataframe_to_apkg",
]
