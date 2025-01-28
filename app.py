from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
import gradio as gr
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import hashlib
import genanki
import random
import json
import tempfile
from pathlib import Path


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


# Replace the caching implementation with a proper cache dictionary
_response_cache = {}  # Global cache dictionary

@lru_cache(maxsize=100)
def get_cached_response(cache_key: str):
    """Get response from cache"""
    return _response_cache.get(cache_key)

def set_cached_response(cache_key: str, response):
    """Set response in cache"""
    _response_cache[cache_key] = response

def create_cache_key(prompt: str, model: str) -> str:
    """Create a unique cache key for the API request"""
    return hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()


# Add retry decorator for API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying API call (attempt {retry_state.attempt_number})"
    )
)
def structured_output_completion(
    client, model, response_format, system_prompt, user_prompt
):
    """Make API call with retry logic and caching"""
    cache_key = create_cache_key(f"{system_prompt}:{user_prompt}", model)
    cached_response = get_cached_response(cache_key)
    
    if cached_response is not None:
        logger.info("Using cached response")
        return cached_response

    try:
        logger.debug(f"Making API call with model {model}")
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            response_format=response_format,
        )

        if not hasattr(completion, "choices") or not completion.choices:
            logger.warning("No choices returned in the completion.")
            return None

        first_choice = completion.choices[0]
        if not hasattr(first_choice, "message"):
            logger.warning("No message found in the first choice.")
            return None

        if not hasattr(first_choice.message, "parsed"):
            logger.warning("Parsed message not available in the first choice.")
            return None

        result = first_choice.message.parsed
        # Cache the successful response
        set_cached_response(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"API call failed: {str(e)}", exc_info=True)
        raise


def generate_cards_batch(
    client,
    model: str,
    topic: str,
    num_cards: int,
    system_prompt: str,
    batch_size: int = 3
) -> List[Card]:
    """Generate cards in batches to avoid overloading the API"""
    cards = []
    remaining = num_cards
    
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        
        card_prompt = f"""
        You are to generate {current_batch} cards on: "{topic}".
        Questions should cover both sample problems and concepts.
        Use the explanation field to help the user understand the reason behind things 
        and maximize learning. Additionally, offer tips (performance, gotchas, etc.).
        """
        
        try:
            batch_response = structured_output_completion(
                client, model, CardList, system_prompt, card_prompt
            )
            if batch_response and hasattr(batch_response, "cards"):
                cards.extend(batch_response.cards)
                remaining -= len(batch_response.cards)
                logger.info(f"Generated batch of {len(batch_response.cards)} cards")
            else:
                logger.warning(f"Failed to generate batch for topic {topic}")
                break  # Break if we get an invalid response
                
        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}", exc_info=True)
            gr.Warning(f"Failed to generate some cards for '{topic}'")
            break  # Break on error to avoid infinite loops
            
    return cards


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

    # Instead of returning a generator, use gr.Progress()
    progress_tracker = gr.Progress(track_tqdm=True)
    flattened_data = []
    total = 0
    
    # Use progress_tracker to show progress
    for i, topic in enumerate(progress_tracker.tqdm(topic_list, desc="Generating cards")):
        progress_html = f"""
        <div style="text-align: center">
            <p>Generating cards for topic {i+1}/{len(topic_list)}: {topic}</p>
            <p>Cards generated so far: {total}</p>
        </div>
        """
        
        try:
            cards = generate_cards_batch(
                client,
                model,
                topic,
                cards_per_topic,
                system_prompt,
                batch_size=3
            )
            
            if cards:
                card_list = CardList(topic=topic, cards=cards)
                for card_index, card in enumerate(card_list.cards, start=1):
                    index = f"{i+1}.{card_index}"
                    row = [
                        index, 
                        topic, 
                        card.front.question,
                        card.back.answer,
                        card.back.explanation,
                        card.back.example
                    ]
                    flattened_data.append(row)
                    total += 1
                
                gr.Info(f"✅ Generated {len(cards)} cards for {topic}")
            
        except Exception as e:
            logger.error(f"Failed to generate cards for topic {topic}: {str(e)}")
            gr.Warning(f"Failed to generate cards for '{topic}'")
            continue

    final_html = f"""
    <div style="text-align: center">
        <p>✅ Generation complete!</p>
        <p>Total cards generated: {total}</p>
    </div>
    """
    
    return flattened_data, final_html, total


# Add these constants after the imports
BASIC_MODEL = genanki.Model(
    random.randrange(1 << 30, 1 << 31),  # Random model ID
    'AnkiGen Basic',
    fields=[
        {'name': 'Question'},
        {'name': 'Answer'},
        {'name': 'Explanation'},
        {'name': 'Example'},
    ],
    templates=[{
        'name': 'Card 1',
        'qfmt': '''
            <div class="card question">
                <div class="content">{{Question}}</div>
            </div>
        ''',
        'afmt': '''
            <div class="card answer">
                <div class="question">{{Question}}</div>
                <hr>
                <div class="content">
                    <div class="answer-section">
                        <h3>Answer:</h3>
                        <div>{{Answer}}</div>
                    </div>
                    
                    <div class="explanation-section">
                        <h3>Explanation:</h3>
                        <div>{{Explanation}}</div>
                    </div>
                    
                    <div class="example-section">
                        <h3>Example:</h3>
                        <pre><code>{{Example}}</code></pre>
                    </div>
                </div>
            </div>
        ''',
    }],
    css='''
        .card {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            font-size: 16px;
            text-align: left;
            color: #333;
            line-height: 1.5;
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            background: #fff;
        }
        
        .question {
            font-size: 1.2em;
            font-weight: 500;
            color: #2563eb;
            margin-bottom: 1em;
        }
        
        hr {
            border: none;
            border-top: 2px solid #e5e7eb;
            margin: 1.5em 0;
        }
        
        h3 {
            color: #1f2937;
            font-size: 1.1em;
            margin: 1em 0 0.5em 0;
        }
        
        .answer-section {
            background: #f0f9ff;
            padding: 1em;
            border-radius: 6px;
            margin: 1em 0;
        }
        
        .explanation-section {
            background: #f0fdf4;
            padding: 1em;
            border-radius: 6px;
            margin: 1em 0;
        }
        
        .example-section {
            background: #fef2f2;
            padding: 1em;
            border-radius: 6px;
            margin: 1em 0;
        }
        
        pre code {
            display: block;
            background: #1f2937;
            color: #e5e7eb;
            padding: 1em;
            border-radius: 4px;
            overflow-x: auto;
            font-family: 'Fira Code', monospace;
        }
    '''
)

# Split the export functions
def export_csv(data):
    """Export the generated cards as a CSV file"""
    if data is None:
        raise gr.Error("No data to export. Please generate cards first.")
        
    if len(data) < 2:  # Minimum 2 cards
        raise gr.Error("Need at least 2 cards to export.")

    try:
        gr.Info("💾 Exporting to CSV...")
        csv_path = "anki_cards.csv"
        data.to_csv(csv_path, index=False)
        gr.Info("✅ CSV export complete!")
        return gr.File(value=csv_path, visible=True)
    
    except Exception as e:
        logger.error(f"Failed to export CSV: {str(e)}", exc_info=True)
        raise gr.Error(f"Failed to export CSV: {str(e)}")

def export_deck(data, subject):
    """Export the generated cards as an Anki deck"""
    if data is None:
        raise gr.Error("No data to export. Please generate cards first.")
        
    if len(data) < 2:  # Minimum 2 cards
        raise gr.Error("Need at least 2 cards to export.")

    try:
        gr.Info("💾 Creating Anki deck...")
        
        # Create a new deck with a random ID
        deck_id = random.randrange(1 << 30, 1 << 31)
        deck = genanki.Deck(deck_id, f"AnkiGen - {subject}")
        
        # Convert DataFrame to records for easier access
        records = data.to_dict('records')
        
        # Add notes to the deck
        for record in records:
            note = genanki.Note(
                model=BASIC_MODEL,
                fields=[
                    str(record['Question']),
                    str(record['Answer']),
                    str(record['Explanation']),
                    str(record['Example'])
                ]
            )
            deck.add_note(note)
        
        # Create a temporary directory for the package
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "anki_deck.apkg"
            package = genanki.Package(deck)
            package.write_to_file(output_path)
            
            # Copy to a more permanent location
            final_path = "anki_deck.apkg"
            with open(output_path, 'rb') as src, open(final_path, 'wb') as dst:
                dst.write(src.read())
        
        gr.Info("✅ Anki deck export complete!")
        return gr.File(value=final_path, visible=True)
    
    except Exception as e:
        logger.error(f"Failed to export Anki deck: {str(e)}", exc_info=True)
        raise gr.Error(f"Failed to export Anki deck: {str(e)}")


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

# Create a custom theme
custom_theme = gr.themes.Soft().set(
    body_background_fill="*background_fill_secondary",
    block_background_fill="*background_fill_primary",
    block_border_width="0",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
)

with gr.Blocks(
    theme=custom_theme,
    title="AnkiGen",
    css="""
        #footer {display:none !important}
        .tall-dataframe {height: 800px !important}
        .contain {max-width: 1200px; margin: auto;}
        .output-cards {border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);}
    """,
    js=js_storage,  # Add the JavaScript
) as ankigen:
    with gr.Column(elem_classes="contain"):
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

                    # Add near the output format documentation
                    with gr.Accordion("Example Card Format", open=False):
                        gr.Code(
                            label="Example Card",
                            value='''
{
    "front": {
        "question": "What is a PRIMARY KEY constraint in SQL?"
    },
    "back": {
        "answer": "A PRIMARY KEY constraint uniquely identifies each record in a table",
        "explanation": "It ensures that a column or set of columns has unique values and cannot contain NULL values. This is essential for maintaining data integrity and establishing relationships between tables.",
        "example": "CREATE TABLE Users (\n  user_id INT PRIMARY KEY,\n  username VARCHAR(50)\n);"
    }
}
                            ''',
                            language="json"
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
                    with gr.Column():
                        gr.Markdown("### Export Options")
                        with gr.Row():
                            export_csv_button = gr.Button("Export to CSV", variant="secondary")
                            export_anki_button = gr.Button("Export to Anki Deck", variant="secondary")
                        download_csv = gr.File(label="Download CSV", interactive=False, visible=False)
                        download_anki = gr.File(label="Download Anki Deck", interactive=False, visible=False)

        # Add near the top of the Blocks
        with gr.Row():
            progress = gr.HTML(visible=False)
            total_cards = gr.Number(label="Total Cards Generated", value=0, visible=False)

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
            outputs=[output, progress, total_cards],
            show_progress=True,
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

if __name__ == "__main__":
    logger.info("Starting AnkiGen application")
    ankigen.launch(share=False, favicon_path="./favicon.ico")
