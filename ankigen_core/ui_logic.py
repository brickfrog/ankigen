# Module for functions that build or manage UI sections/logic

import gradio as gr
import pandas as pd  # Needed for use_selected_subjects type hinting


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

    # Return a dictionary mapping component instances (which will be in app.py scope)
    # to their updated configurations using gr.update()
    # Keys here are placeholders; they need to match the actual Gradio components passed in the outputs list
    # when this function is used as an event handler in app.py.
    return {
        # Visibility updates for mode-specific groups
        "subject_mode_group": gr.update(visible=is_subject),
        "path_mode_group": gr.update(visible=is_path),
        "text_mode_group": gr.update(visible=is_text),
        "web_mode_group": gr.update(visible=is_web),
        # Visibility updates for output areas
        "path_results_group": gr.update(visible=is_path),
        "cards_output_group": gr.update(visible=is_subject or is_text or is_web),
        # Value updates for inputs (clear if mode changes)
        "subject_textbox": gr.update(value=subject_val),
        "description_textbox": gr.update(value=description_val),
        "source_text_textbox": gr.update(value=text_val),
        "url_textbox": gr.update(value=url_val),
        # Clear previous results/outputs
        "output_dataframe": gr.update(value=None),
        "subjects_dataframe": gr.update(value=None),
        "learning_order_markdown": gr.update(value=""),
        "projects_markdown": gr.update(value=""),
        "progress_html": gr.update(value="", visible=False),
        "total_cards_number": gr.update(value=0, visible=False),
    }


def use_selected_subjects(subjects_df: pd.DataFrame | None):
    """Updates UI to use subjects from learning path analysis."""
    if subjects_df is None or subjects_df.empty:
        gr.Warning("No subjects available to copy from Learning Path analysis.")
        # Return updates that change nothing or clear relevant fields if necessary
        # Returning updates for all potential outputs to match the original signature
        return {
            "generation_mode_radio": gr.update(),
            "subject_mode_group": gr.update(),
            "path_mode_group": gr.update(),
            "text_mode_group": gr.update(),
            "web_mode_group": gr.update(),
            "path_results_group": gr.update(),
            "cards_output_group": gr.update(),
            "subject_textbox": gr.update(),
            "description_textbox": gr.update(),
            "source_text_textbox": gr.update(),
            "url_textbox": gr.update(),
            "topic_number_slider": gr.update(),
            "preference_prompt_textbox": gr.update(),
            "output_dataframe": gr.update(),
            "subjects_dataframe": gr.update(),
            "learning_order_markdown": gr.update(),
            "projects_markdown": gr.update(),
            "progress_html": gr.update(),
            "total_cards_number": gr.update(),
        }

    try:
        subjects = subjects_df["Subject"].tolist()
        combined_subject = ", ".join(subjects)
        suggested_topics = min(len(subjects) + 1, 20)
    except KeyError:
        gr.Error("Learning path analysis result is missing the 'Subject' column.")
        # Return no-change updates
        return {
            "generation_mode_radio": gr.update(),
            "subject_mode_group": gr.update(),
            "path_mode_group": gr.update(),
            "text_mode_group": gr.update(),
            "web_mode_group": gr.update(),
            "path_results_group": gr.update(),
            "cards_output_group": gr.update(),
            "subject_textbox": gr.update(),
            "description_textbox": gr.update(),
            "source_text_textbox": gr.update(),
            "url_textbox": gr.update(),
            "topic_number_slider": gr.update(),
            "preference_prompt_textbox": gr.update(),
            "output_dataframe": gr.update(),
            "subjects_dataframe": gr.update(),
            "learning_order_markdown": gr.update(),
            "projects_markdown": gr.update(),
            "progress_html": gr.update(),
            "total_cards_number": gr.update(),
        }

    # Keys here are placeholders, matching the outputs list in app.py's .click handler
    return {
        "generation_mode_radio": "subject",  # Switch mode to subject
        "subject_mode_group": gr.update(visible=True),
        "path_mode_group": gr.update(visible=False),
        "text_mode_group": gr.update(visible=False),
        "web_mode_group": gr.update(visible=False),
        "path_results_group": gr.update(visible=False),
        "cards_output_group": gr.update(visible=True),
        "subject_textbox": combined_subject,
        "description_textbox": "",  # Clear path description
        "source_text_textbox": "",  # Clear text input
        "url_textbox": "",  # Clear URL input
        "topic_number_slider": suggested_topics,
        "preference_prompt_textbox": "Focus on connections between these subjects and their practical applications.",  # Suggest preference
        "output_dataframe": gr.update(value=None),  # Clear previous card output if any
        "subjects_dataframe": subjects_df,  # Keep the dataframe in its output component
        "learning_order_markdown": gr.update(),  # Keep learning order visible for reference if desired
        "projects_markdown": gr.update(),  # Keep projects visible for reference if desired
        "progress_html": gr.update(visible=False),
        "total_cards_number": gr.update(visible=False),
    }
