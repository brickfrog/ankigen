# Module for functions that build or manage UI sections/logic

import gradio as gr
import pandas as pd
from typing import List

from ankigen.utils import get_logger
from ankigen.models import Card

logger = get_logger()


def update_mode_visibility(mode: str, current_subject: str):
    """Updates visibility and values of UI elements based on generation mode.

    Currently only 'subject' mode is supported. This function is kept for
    future extensibility.
    """
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

    return (
        gr.update(visible=True),  # subject_mode (Group) - always visible
        gr.update(visible=True),  # cards_output - always visible
        gr.update(value=current_subject),  # subject textbox value
        gr.update(
            value=pd.DataFrame(columns=main_output_df_columns)
        ),  # output DataFrame
        gr.update(
            value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
            visible=False,
        ),  # total_cards_html
    )


# --- Card Preview and Editing Utilities ---


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
                "Topic": topic_str,
                "Front": card.front.question,
                "Back": card.back.answer,
                "Tags": tags_str,
                "Card Type": card.card_type or "Basic",
                "Explanation": card.back.explanation or "",
                "Example": card.back.example or "",
                "Source_URL": card.metadata.get("source_url", "")
                if card.metadata
                else "",
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
        return []

    for index, row in df.iterrows():
        original_card_index = None
        try:
            card_id = int(row["ID"])  # DataFrame ID is 1-indexed
            original_card_index = card_id - 1

            if 0 <= original_card_index < len(original_cards):
                card_to_update = original_cards[original_card_index]

                new_front = card_to_update.front.model_copy(
                    update={
                        "question": str(row.get("Front", card_to_update.front.question))
                    }
                )
                new_back = card_to_update.back.model_copy(
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

                updated_card = card_to_update.model_copy(
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
                logger.warning(
                    f"Card ID {card_id} from DataFrame is out of bounds for original_cards list."
                )
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(
                f"Error processing row {index} from DataFrame: {row}. Error: {e}"
            )
            if original_card_index is not None and 0 <= original_card_index < len(
                original_cards
            ):
                updated_cards.append(original_cards[original_card_index])
            continue
    return updated_cards
