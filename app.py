from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
import gradio as gr
import os
from datetime import datetime
from gradio.components import State, JSON
import logging
from logging.handlers import RotatingFileHandler
import sys
import json


class Step(BaseModel):
    explanation: str
    output: str


class Subtopics(BaseModel):
    steps: List[Step]
    result: List[str]


class Topics(BaseModel):
    result: List[Subtopics]


class CardFront(BaseModel):
    question: Optional[str] = None


class CardBack(BaseModel):
    answer: Optional[str] = None
    explanation: str
    example: str


class Card(BaseModel):
    front: CardFront
    back: CardBack


class CardList(BaseModel):
    topic: str
    cards: List[Card]


def setup_logging():
    """Configure logging to both file and console"""
    logger = logging.getLogger('ankigen')
    logger.setLevel(logging.DEBUG)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # File handler (detailed logging)
    file_handler = RotatingFileHandler(
        'ankigen.log',
        maxBytes=1024*1024,  # 1MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler (info and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize logger
logger = setup_logging()


def structured_output_completion(
    client, model, response_format, system_prompt, user_prompt
):
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            response_format=response_format,
        )

    except Exception as e:
        print(f"An error occurred during the API call: {e}")
        return None

    try:
        if not hasattr(completion, "choices") or not completion.choices:
            print("No choices returned in the completion.")
            return None

        first_choice = completion.choices[0]
        if not hasattr(first_choice, "message"):
            print("No message found in the first choice.")
            return None

        if not hasattr(first_choice.message, "parsed"):
            print("Parsed message not available in the first choice.")
            return None

        return first_choice.message.parsed

    except Exception as e:
        print(f"An error occurred while processing the completion: {e}")
        raise gr.Error(f"Processing error: {e}")


def generate_cards(
    api_key_input,
    subject,
    topic_number=1,
    cards_per_topic=2,
    preference_prompt="assume I'm a beginner",
):
    logger.info(f"Starting card generation for subject: {subject}")
    logger.debug(f"Parameters: topics={topic_number}, cards_per_topic={cards_per_topic}")

    # Input validation
    if not api_key_input:
        logger.warning("No API key provided")
        raise gr.Error("OpenAI API key is required")
    if not api_key_input.startswith("sk-"):
        logger.warning("Invalid API key format")
        raise gr.Error("Invalid API key format. OpenAI keys should start with 'sk-'")
    if not subject.strip():
        logger.warning("No subject provided")
        raise gr.Error("Subject is required")
    
    gr.Info("🚀 Starting card generation...")
    
    try:
        logger.debug("Initializing OpenAI client")
        client = OpenAI(api_key=api_key_input)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}", exc_info=True)
        raise gr.Error(f"Failed to initialize OpenAI client: {str(e)}")

    # Update model name - looks like a typo in original
    model = "gpt-4o-mini"

    all_card_lists = []
    
    gr.Info(f"📚 Generating {topic_number} topics for {subject}...")

    system_prompt = f"""
    You are an expert in {subject}, assisting the user to master the topic while 
    keeping in mind the user's preferences: {preference_prompt}.
    """

    topic_prompt = f"""
    Generate the top {topic_number} important subjects to know on {subject} in 
    order of ascending difficulty.
    """

    try:
        topics_response = structured_output_completion(
            client, model, Topics, system_prompt, topic_prompt
        )
        if topics_response is None:
            raise gr.Error("Failed to generate topics. Please try again.")
        if not hasattr(topics_response, "result") or not topics_response.result:
            raise gr.Error("Invalid response format from API. Please try again.")
            
        topic_list = [
            item for subtopic in topics_response.result for item in subtopic.result
        ][:topic_number]
        
        gr.Info(f"✨ Generated {len(topic_list)} topics successfully!")
        
    except Exception as e:
        raise gr.Error(f"Topic generation failed: {str(e)}")

    # Card generation with progress updates
    for i, topic in enumerate(topic_list, 1):
        gr.Info(f"📝 Generating cards for topic {i}/{len(topic_list)}: {topic}")
        
        card_prompt = f"""
        You are to generate {cards_per_topic} cards on {subject}: "{topic}" 
        keeping in mind the user's preferences: {preference_prompt}.
        
        Questions should cover both sample problems and concepts.

        Use the explanation field to help the user understand the reason behind things 
        and maximize learning. Additionally, offer tips (performance, gotchas, etc.).
        """
        
        try:
            cards = structured_output_completion(
                client, model, CardList, system_prompt, card_prompt
            )
            if cards is None:
                gr.Warning(f"Skipping topic '{topic}' - failed to generate cards")
                continue
                
            if not hasattr(cards, "topic") or not hasattr(cards, "cards"):
                gr.Warning(f"Skipping topic '{topic}' - invalid card format")
                continue
                
            all_card_lists.append(cards)
            gr.Info(f"✅ Generated {len(cards.cards)} cards for {topic}")
            
        except Exception as e:
            gr.Warning(f"Failed to generate cards for '{topic}': {str(e)}")
            continue

    if not all_card_lists:
        raise gr.Error("Failed to generate any valid cards. Please try again.")

    flattened_data = []

    for card_list_index, card_list in enumerate(all_card_lists, start=1):
        try:
            topic = card_list.topic
            # Get the total number of cards in this list to determine padding
            total_cards = len(card_list.cards)
            # Calculate the number of digits needed for padding
            padding = len(str(total_cards))

            for card_index, card in enumerate(card_list.cards, start=1):
                # Format the index with zero-padding
                index = f"{card_list_index}.{card_index:0{padding}}"
                question = card.front.question
                answer = card.back.answer
                explanation = card.back.explanation
                example = card.back.example
                row = [index, topic, question, answer, explanation, example]
                flattened_data.append(row)
        except Exception as e:
            print(f"An error occurred while processing card {index}: {e}")
            continue

    # At the end, just return the flattened data
    logger.debug(f"Generated flattened data structure: {type(flattened_data)}")
    logger.debug(f"First row sample: {flattened_data[0] if flattened_data else 'No data'}")
    return flattened_data


def export_csv(d):
    MIN_ROWS = 2

    if d is None:
        raise gr.Error("No data to export. Please generate cards first.")
        
    if len(d) < MIN_ROWS:
        raise gr.Error(f"Need at least {MIN_ROWS} cards to export.")

    try:
        gr.Info("💾 Exporting to CSV...")
        d.to_csv("anki_deck.csv", index=False)
        gr.Info("✅ Export complete!")
        return gr.File(value="anki_deck.csv", visible=True)
    except Exception as e:
        raise gr.Error(f"Failed to export CSV: {str(e)}")


# Add this near the top where we define our CSS
js_storage = """
async () => {
    // Load decks from localStorage
    const loadDecks = () => {
        const decks = localStorage.getItem('ankigen_decks');
        return decks ? JSON.parse(decks) : [];
    };

    // Save decks to localStorage
    const saveDecks = (decks) => {
        localStorage.setItem('ankigen_decks', JSON.stringify(decks));
    };

    // Add methods to window for Gradio to access
    window.loadStoredDecks = loadDecks;
    window.saveStoredDecks = saveDecks;
    
    // Initial load
    return loadDecks();
}
"""

with gr.Blocks(
    gr.themes.Soft(), 
    title="AnkiGen", 
    css="#footer{display:none !important} .tall-dataframe{height: 800px !important}",
    js=js_storage,  # Add the JavaScript
) as ankigen:
    gr.Markdown("# 📚 AnkiGen - Anki Card Generator")
    gr.Markdown("#### Generate an LLM generated Anki comptible csv based on your subject and preferences.") #noqa

    with gr.Row():
        # Left Column - Controls
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")

            # Basic Settings
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="Enter your OpenAI API key",
                value=os.getenv("OPENAI_API_KEY", ""),
                info="Your OpenAI API key starting with 'sk-'",
            )
            subject = gr.Textbox(
                label="Subject",
                placeholder="Enter the subject, e.g., 'Basic SQL Concepts'",
                info="The topic you want to generate flashcards for",
            )
            
            # Generation Button
            generate_button = gr.Button("Generate Cards", variant="primary")

            # Advanced Settings in Accordion
            with gr.Accordion("Advanced Settings", open=False):
                topic_number = gr.Slider(
                    label="Number of Topics",
                    minimum=2,
                    maximum=20,
                    step=1,
                    value=2,
                    info="How many distinct topics to cover within the subject",
                )
                cards_per_topic = gr.Slider(
                    label="Cards per Topic",
                    minimum=2,
                    maximum=30,
                    step=1,
                    value=3,
                    info="How many flashcards to generate for each topic",
                )
                preference_prompt = gr.Textbox(
                    label="Learning Preferences",
                    placeholder="e.g., 'Assume I'm a beginner' or 'Focus on practical examples'",
                    info="Customize how the content is presented",
                    lines=3,
                )

        # Right Column - Output
        with gr.Column(scale=2):
            gr.Markdown("### Generated Cards")
            
            # Output Format Documentation
            with gr.Accordion("Output Format", open=True):
                gr.Markdown(
                    """
                    The generated CSV will contain the following fields:
                    * **Index**: Unique identifier for each card
                    * **Topic**: The subject subtopic this card belongs to
                    * **Question**: The front of the flashcard
                    * **Answer**: The core answer
                    * **Explanation**: Detailed explanation of the concept
                    * **Example**: A practical example to reinforce learning
                    """
                )
            
            # Dataframe Output
            output = gr.Dataframe(
                headers=[
                    "Index",
                    "Topic",
                    "Question",
                    "Answer",
                    "Explanation",
                    "Example",
                ],
                interactive=True,
                elem_classes="tall-dataframe",
                wrap=True,
                column_widths=[50, 100, 200, 200, 250, 200],
            )

            # Export Controls
            with gr.Row():
                export_button = gr.Button("Export to CSV", variant="secondary")
                download_link = gr.File(interactive=False, visible=False)
                clear_button = gr.ClearButton(
                    components=[subject, preference_prompt],
                    value="Clear Form",
                )

    # Simplified event handlers
    generate_button.click(
        fn=generate_cards,
        inputs=[
            api_key_input,
            subject,
            topic_number,
            cards_per_topic,
            preference_prompt,
        ],
        outputs=output,
        show_progress="full",
    )

    export_button.click(
        fn=export_csv,
        inputs=output,
        outputs=download_link,
        show_progress="full",
    )

if __name__ == "__main__":
    logger.info("Starting AnkiGen application")
    ankigen.launch(share=False, favicon_path="./favicon.ico")
