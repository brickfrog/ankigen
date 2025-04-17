from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
import gradio as gr
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from functools import lru_cache
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import hashlib
import genanki
import random
import json
import tempfile
from pathlib import Path
import pandas as pd


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
    metadata: Optional[dict] = None
    card_type: str = "basic"  # Add card_type, default to basic


class CardList(BaseModel):
    topic: str
    cards: List[Card]


class ConceptBreakdown(BaseModel):
    main_concept: str
    prerequisites: List[str]
    learning_outcomes: List[str]
    common_misconceptions: List[str]
    difficulty_level: str  # "beginner", "intermediate", "advanced"


class CardGeneration(BaseModel):
    concept: str
    thought_process: str
    verification_steps: List[str]
    card: Card


class LearningSequence(BaseModel):
    topic: str
    concepts: List[ConceptBreakdown]
    cards: List[CardGeneration]
    suggested_study_order: List[str]
    review_recommendations: List[str]


def setup_logging():
    """Configure logging to both file and console"""
    logger = logging.getLogger("ankigen")
    logger.setLevel(logging.DEBUG)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # File handler (detailed logging)
    file_handler = RotatingFileHandler(
        "ankigen.log",
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5,
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
    ),
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

        # Add JSON instruction to system prompt
        system_prompt = f"{system_prompt}\nProvide your response as a JSON object matching the specified schema."

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        if not hasattr(completion, "choices") or not completion.choices:
            logger.warning("No choices returned in the completion.")
            return None

        first_choice = completion.choices[0]
        if not hasattr(first_choice, "message"):
            logger.warning("No message found in the first choice.")
            return None

        # Parse the JSON response
        result = json.loads(first_choice.message.content)

        # Cache the successful response
        set_cached_response(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"API call failed: {str(e)}", exc_info=True)
        raise


def generate_cards_batch(
    client, model, topic, num_cards, system_prompt, generate_cloze=False, batch_size=3
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
                    "question": "question text (potentially with {{c1::cloze syntax}})"
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
        response = structured_output_completion(
            client, model, {"type": "json_object"}, system_prompt, cards_prompt
        )

        if not response or "cards" not in response:
            logger.error("Invalid cards response format")
            raise ValueError("Failed to generate cards. Please try again.")

        # Convert the JSON response into Card objects
        cards = []
        for card_data in response["cards"]:
            # Ensure required fields are present before creating Card object
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

            card = Card(
                card_type=card_data.get("card_type", "basic"),
                front=CardFront(**card_data["front"]),
                back=CardBack(**card_data["back"]),
                metadata=card_data.get("metadata", {}),
            )
            cards.append(card)

        return cards

    except Exception as e:
        logger.error(
            f"Failed to generate cards batch for {topic}: {str(e)}", exc_info=True
        )
        raise


# Add near the top with other constants
AVAILABLE_MODELS = [
    {
        "value": "gpt-4.1-mini",  # Default model
        "label": "gpt-4.1 Mini (Fastest)",
        "description": "Balanced speed and quality",
    },
    {
        "value": "gpt-4.1",
        "label": "gpt-4.1 (Better Quality)",
        "description": "Higher quality, slower generation",
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
]


def generate_cards(
    api_key_input,
    subject,
    model_name="gpt-4.1-mini",
    topic_number=1,
    cards_per_topic=2,
    preference_prompt="assume I'm a beginner",
    generate_cloze=False,
):
    logger.info(f"Starting card generation for subject: {subject}")
    logger.debug(
        f"Parameters: topics={topic_number}, cards_per_topic={cards_per_topic}, cloze={generate_cloze}"
    )

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

    model = model_name
    flattened_data = []
    total = 0

    progress_tracker = gr.Progress(track_tqdm=True)

    system_prompt = f"""
    You are an expert educator in {subject}, creating an optimized learning sequence.
    Your goal is to:
    1. Break down the subject into logical concepts
    2. Identify prerequisites and learning outcomes
    3. Generate cards that build upon each other
    4. Address and correct common misconceptions
    5. Include verification steps to minimize hallucinations
    6. Provide a recommended study order

    For explanations and examples:
    - Keep explanations in plain text
    - Format code examples with triple backticks (```)
    - Separate conceptual examples from code examples
    - Use clear, concise language

    Keep in mind the user's preferences: {preference_prompt}
    """

    topic_prompt = f"""
    Generate the top {topic_number} important subjects to know about {subject} in 
    order of ascending difficulty. Return your response as a JSON object with the following structure:
    {{
        "topics": [
            {{
                "name": "topic name",
                "difficulty": "beginner/intermediate/advanced",
                "description": "brief description"
            }}
        ]
    }}
    """

    try:
        logger.info("Generating topics...")
        topics_response = structured_output_completion(
            client, model, {"type": "json_object"}, system_prompt, topic_prompt
        )

        if not topics_response or "topics" not in topics_response:
            logger.error("Invalid topics response format")
            raise gr.Error("Failed to generate topics. Please try again.")

        topics = topics_response["topics"]

        gr.Info(f"✨ Generated {len(topics)} topics successfully!")

        # Generate cards for each topic
        for i, topic in enumerate(
            progress_tracker.tqdm(topics, desc="Generating cards")
        ):
            try:
                cards = generate_cards_batch(
                    client,
                    model,
                    topic["name"],
                    cards_per_topic,
                    system_prompt,
                    generate_cloze=generate_cloze,
                    batch_size=3,
                )

                if cards:
                    for card_index, card in enumerate(cards, start=1):
                        index = f"{i + 1}.{card_index}"
                        metadata = card.metadata or {}

                        row = [
                            index,
                            topic["name"],
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
                        flattened_data.append(row)
                        total += 1

                    gr.Info(f"✅ Generated {len(cards)} cards for {topic['name']}")

            except Exception as e:
                logger.error(
                    f"Failed to generate cards for topic {topic['name']}: {str(e)}"
                )
                gr.Warning(f"Failed to generate cards for '{topic['name']}'")
                continue

        final_html = f"""
        <div style="text-align: center">
            <p>✅ Generation complete!</p>
            <p>Total cards generated: {total}</p>
        </div>
        """

        # Convert to DataFrame with all columns
        df = pd.DataFrame(
            flattened_data,
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

        return df, final_html, total

    except Exception as e:
        logger.error(f"Card generation failed: {str(e)}", exc_info=True)
        raise gr.Error(f"Card generation failed: {str(e)}")


# Update the BASIC_MODEL definition with enhanced CSS/HTML
BASIC_MODEL = genanki.Model(
    random.randrange(1 << 30, 1 << 31),
    "AnkiGen Enhanced",
    fields=[
        {"name": "Question"},
        {"name": "Answer"},
        {"name": "Explanation"},
        {"name": "Example"},
        {"name": "Prerequisites"},
        {"name": "Learning_Outcomes"},
        {"name": "Common_Misconceptions"},
        {"name": "Difficulty"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": """
            <div class="card question-side">
                <div class="difficulty-indicator {{Difficulty}}"></div>
                <div class="content">
                    <div class="question">{{Question}}</div>
                    <div class="prerequisites" onclick="event.stopPropagation();">
                        <div class="prerequisites-toggle">Show Prerequisites</div>
                        <div class="prerequisites-content">{{Prerequisites}}</div>
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
            <div class="card answer-side">
                <div class="content">
                    <div class="question-section">
                        <div class="question">{{Question}}</div>
                        <div class="prerequisites">
                            <strong>Prerequisites:</strong> {{Prerequisites}}
                        </div>
                    </div>
                    <hr>
                    
                    <div class="answer-section">
                        <h3>Answer</h3>
                        <div class="answer">{{Answer}}</div>
                    </div>
                    
                    <div class="explanation-section">
                        <h3>Explanation</h3>
                        <div class="explanation-text">{{Explanation}}</div>
                    </div>
                    
                    <div class="example-section">
                        <h3>Example</h3>
                        <div class="example-text"></div>
                        <pre><code>{{Example}}</code></pre>
                    </div>
                    
                    <div class="metadata-section">
                        <div class="learning-outcomes">
                            <h3>Learning Outcomes</h3>
                            <div>{{Learning_Outcomes}}</div>
                        </div>
                        
                        <div class="misconceptions">
                            <h3>Common Misconceptions - Debunked</h3>
                            <div>{{Common_Misconceptions}}</div>
                        </div>
                        
                        <div class="difficulty">
                            <h3>Difficulty Level</h3>
                            <div>{{Difficulty}}</div>
                        </div>
                    </div>
                </div>
            </div>
        """,
        }
    ],
    css="""
        /* Base styles */
        .card {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            font-size: 16px;
            line-height: 1.6;
            color: #1a1a1a;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #ffffff;
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
        
        .difficulty-indicator.beginner { background: #4ade80; }
        .difficulty-indicator.intermediate { background: #fbbf24; }
        .difficulty-indicator.advanced { background: #ef4444; }
        
        .question {
            font-size: 1.3em;
            font-weight: 600;
            color: #2563eb;
            margin-bottom: 1.5em;
        }
        
        .prerequisites {
            margin-top: 1em;
            font-size: 0.9em;
            color: #666;
        }
        
        .prerequisites-toggle {
            color: #2563eb;
            cursor: pointer;
            text-decoration: underline;
        }
        
        .prerequisites-content {
            display: none;
            margin-top: 0.5em;
            padding: 0.5em;
            background: #f8fafc;
            border-radius: 4px;
        }
        
        .prerequisites.show .prerequisites-content {
            display: block;
        }
        
        /* Answer side */
        .answer-section,
        .explanation-section,
        .example-section {
            margin: 1.5em 0;
            padding: 1.2em;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .answer-section {
            background: #f0f9ff;
            border-left: 4px solid #2563eb;
        }
        
        .explanation-section {
            background: #f0fdf4;
            border-left: 4px solid #4ade80;
        }
        
        .example-section {
            background: #fff7ed;
            border-left: 4px solid #f97316;
        }
        
        /* Code blocks */
        pre code {
            display: block;
            padding: 1em;
            background: #1e293b;
            color: #e2e8f0;
            border-radius: 6px;
            overflow-x: auto;
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 0.9em;
        }
        
        /* Metadata tabs */
        .metadata-tabs {
            margin-top: 2em;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .tab-buttons {
            display: flex;
            background: #f8fafc;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .tab-btn {
            flex: 1;
            padding: 0.8em;
            border: none;
            background: none;
            cursor: pointer;
            font-weight: 500;
            color: #64748b;
            transition: all 0.2s;
        }
        
        .tab-btn:hover {
            background: #f1f5f9;
        }
        
        .tab-btn.active {
            color: #2563eb;
            background: #fff;
            border-bottom: 2px solid #2563eb;
        }
        
        .tab-content {
            display: none;
            padding: 1.2em;
        }
        
        .tab-content.active {
            display: block;
        }
        
        /* Responsive design */
        @media (max-width: 640px) {
            .tab-buttons {
                flex-direction: column;
            }
            
            .tab-btn {
                width: 100%;
                text-align: left;
                padding: 0.6em;
            }
            
            .answer-section,
            .explanation-section,
            .example-section {
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
        
        .tab-content.active {
            animation: fadeIn 0.2s ease-in-out;
        }
    """,
)


# Define the Cloze Model (based on Anki's default Cloze type)
CLOZE_MODEL = genanki.Model(
    random.randrange(1 << 30, 1 << 31),  # Needs a unique ID
    "AnkiGen Cloze Enhanced",
    model_type=genanki.Model.CLOZE,  # Specify model type as CLOZE
    fields=[
        {"name": "Text"},  # Field for the text containing the cloze deletion
        {"name": "Extra"},  # Field for additional info shown on the back
        {"name": "Difficulty"},  # Keep metadata
        {"name": "SourceTopic"},  # Add topic info
    ],
    templates=[
        {
            "name": "Cloze Card",
            "qfmt": "{{cloze:Text}}",
            "afmt": """
                {{cloze:Text}}
                <hr>
                <div class="extra-info">{{Extra}}</div>
                <div class="metadata-footer">Difficulty: {{Difficulty}} | Topic: {{SourceTopic}}</div>
            """,
        }
    ],
    css="""
        .card {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            font-size: 16px; line-height: 1.6; color: #1a1a1a;
            max-width: 800px; margin: 0 auto; padding: 20px;
            background: #ffffff;
        }
        .cloze {
            font-weight: bold; color: #2563eb;
        }
        .extra-info {
            margin-top: 1em; padding-top: 1em;
            border-top: 1px solid #e5e7eb;
            font-size: 0.95em; color: #333;
            background: #f8fafc; padding: 1em; border-radius: 6px;
        }
        .extra-info h3 { margin-top: 0.5em; font-size: 1.1em; color: #1e293b; }
        .extra-info pre code {
            display: block; padding: 1em; background: #1e293b;
            color: #e2e8f0; border-radius: 6px; overflow-x: auto;
            font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.9em;
            margin-top: 0.5em;
        }
        .metadata-footer {
            margin-top: 1.5em; font-size: 0.85em; color: #64748b; text-align: right;
        }
    """,
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
    """Export the generated cards as an Anki deck with pedagogical metadata"""
    if data is None:
        raise gr.Error("No data to export. Please generate cards first.")

    if len(data) < 2:  # Minimum 2 cards
        raise gr.Error("Need at least 2 cards to export.")

    try:
        gr.Info("💾 Creating Anki deck...")

        deck_id = random.randrange(1 << 30, 1 << 31)
        deck = genanki.Deck(deck_id, f"AnkiGen - {subject}")

        records = data.to_dict("records")

        # Ensure both models are added to the deck package
        deck.add_model(BASIC_MODEL)
        deck.add_model(CLOZE_MODEL)

        # Add notes to the deck
        for record in records:
            card_type = record.get("Card_Type", "basic").lower()

            if card_type == "cloze":
                # Create Cloze note
                extra_content = f"""
                    <h3>Explanation:</h3>
                    <div>{record["Explanation"]}</div>
                    <h3>Example:</h3>
                    <pre><code>{record["Example"]}</code></pre>
                    <h3>Prerequisites:</h3>
                    <div>{record["Prerequisites"]}</div>
                    <h3>Learning Outcomes:</h3>
                    <div>{record["Learning_Outcomes"]}</div>
                    <h3>Watch out for:</h3>
                    <div>{record["Common_Misconceptions"]}</div>
                """
                note = genanki.Note(
                    model=CLOZE_MODEL,
                    fields=[
                        str(record["Question"]),  # Contains {{c1::...}}
                        extra_content,  # All other info goes here
                        str(record["Difficulty"]),
                        str(record["Topic"]),
                    ],
                )
            else:  # Default to basic card
                # Create Basic note (existing logic)
                note = genanki.Note(
                    model=BASIC_MODEL,
                    fields=[
                        str(record["Question"]),
                        str(record["Answer"]),
                        str(record["Explanation"]),
                        str(record["Example"]),
                        str(record["Prerequisites"]),
                        str(record["Learning_Outcomes"]),
                        str(record["Common_Misconceptions"]),
                        str(record["Difficulty"]),
                    ],
                )

            deck.add_note(note)

        # Create a temporary directory for the package
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "anki_deck.apkg"
            package = genanki.Package(deck)
            package.write_to_file(output_path)

            # Copy to a more permanent location
            final_path = "anki_deck.apkg"
            with open(output_path, "rb") as src, open(final_path, "wb") as dst:
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


def analyze_learning_path(api_key, description, model):
    """Analyze a job description or learning goal to create a structured learning path"""

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        raise gr.Error(f"Failed to initialize OpenAI client: {str(e)}")

    system_prompt = """You are an expert curriculum designer and educational consultant.
    Your task is to analyze learning goals and create structured, achievable learning paths.
    Break down complex topics into manageable subjects, identify prerequisites,
    and suggest practical projects that reinforce learning.
    Focus on creating a logical progression that builds upon previous knowledge."""

    path_prompt = f"""
    Analyze this description and create a structured learning path.
    Return your analysis as a JSON object with the following structure:
    {{
        "subjects": [
            {{
                "Subject": "name of the subject",
                "Prerequisites": "required prior knowledge",
                "Time Estimate": "estimated time to learn"
            }}
        ],
        "learning_order": "recommended sequence of study",
        "projects": "suggested practical projects"
    }}

    Description to analyze:
    {description}
    """

    try:
        response = structured_output_completion(
            client, model, {"type": "json_object"}, system_prompt, path_prompt
        )

        if (
            not response
            or "subjects" not in response
            or "learning_order" not in response
            or "projects" not in response
        ):
            logger.error("Invalid response format from API")
            raise gr.Error("Failed to analyze learning path. Please try again.")

        subjects_df = pd.DataFrame(response["subjects"])
        learning_order_text = (
            f"### Recommended Learning Order\n{response['learning_order']}"
        )
        projects_text = f"### Suggested Projects\n{response['projects']}"

        return subjects_df, learning_order_text, projects_text

    except Exception as e:
        logger.error(f"Failed to analyze learning path: {str(e)}")
        raise gr.Error(f"Failed to analyze learning path: {str(e)}")


# --- Example Data for Initialization ---
example_data = pd.DataFrame(
    [
        [
            "1.1",
            "SQL Basics",
            "basic",
            "What is a SELECT statement used for?",
            "Retrieving data from one or more database tables.",
            "The SELECT statement is the most common command in SQL. It allows you to specify which columns and rows you want to retrieve from a table based on certain conditions.",
            "```sql\\nSELECT column1, column2 FROM my_table WHERE condition;\\n```",
            ["Understanding of database tables"],
            ["Retrieve specific data", "Filter results"],
            ["❌ SELECT * is always efficient (Reality: Can be slow for large tables)"],
            "beginner",
        ],
        [
            "2.1",
            "Python Fundamentals",
            "cloze",
            "The primary keyword to define a function in Python is {{c1::def}}.",
            "def",
            "Functions are defined using the `def` keyword, followed by the function name, parentheses for arguments, and a colon. The indented block below defines the function body.",
            # Use a raw triple-quoted string for the code block to avoid escaping issues
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

with gr.Blocks(
    theme=custom_theme,
    title="AnkiGen",
    css="""
        #footer {display:none !important}
        .tall-dataframe {min-height: 500px !important}
        .contain {max-width: 95% !important; margin: auto;}
        .output-cards {border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);}
        .hint-text {font-size: 0.9em; color: #666; margin-top: 4px;}
        .export-group > .gradio-group { margin-bottom: 0 !important; padding-bottom: 5px !important; }
    """,
    js=js_storage,
) as ankigen:
    with gr.Column(elem_classes="contain"):
        gr.Markdown("# 📚 AnkiGen - Advanced Anki Card Generator")
        gr.Markdown("""
        #### Generate comprehensive Anki flashcards using AI. 
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")

                # Add mode selection
                generation_mode = gr.Radio(
                    choices=["subject", "path"],
                    value="subject",
                    label="Generation Mode",
                    info="Choose how you want to generate content",
                )

                # Create containers for different modes
                with gr.Group() as subject_mode:
                    subject = gr.Textbox(
                        label="Subject",
                        placeholder="Enter the subject, e.g., 'Basic SQL Concepts'",
                        info="The topic you want to generate flashcards for",
                    )

                with gr.Group(visible=False) as path_mode:
                    description = gr.Textbox(
                        label="Learning Goal",
                        placeholder="Paste a job description or describe what you want to learn...",
                        info="We'll break this down into learnable subjects",
                        lines=5,
                    )
                    analyze_button = gr.Button(
                        "Analyze & Break Down", variant="secondary"
                    )

                # Common settings
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="Enter your OpenAI API key",
                    value=os.getenv("OPENAI_API_KEY", ""),
                    info="Your OpenAI API key starting with 'sk-'",
                )

                # Generation Button
                generate_button = gr.Button("Generate Cards", variant="primary")

                # Advanced Settings in Accordion
                with gr.Accordion("Advanced Settings", open=False):
                    model_choice = gr.Dropdown(
                        choices=["gpt-4.1-mini", "gpt-4.1"],
                        value="gpt-4.1-mini",
                        label="Model Selection",
                        info="Select the AI model to use for generation",
                    )

                    # Add tooltip/description for models
                    model_info = gr.Markdown("""
                    **Model Information:**
                    - **gpt-4.1-mini**: Fastest option, good for most use cases
                    - **gpt-4.1**: Better quality, takes longer to generate
                    """)

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
                    generate_cloze_checkbox = gr.Checkbox(
                        label="Generate Cloze Cards (Experimental)",
                        value=False,
                        info="Allow the AI to generate fill-in-the-blank style cards where appropriate.",
                    )

            # Right column - add a new container for learning path results
            with gr.Column(scale=2):
                with gr.Group(visible=False) as path_results:
                    gr.Markdown("### Learning Path Analysis")
                    subjects_list = gr.Dataframe(
                        headers=["Subject", "Prerequisites", "Time Estimate"],
                        label="Recommended Subjects",
                        interactive=False,
                    )
                    learning_order = gr.Markdown("### Recommended Learning Order")
                    projects = gr.Markdown("### Suggested Projects")

                    # Replace generate_selected with use_subjects
                    use_subjects = gr.Button(
                        "Use These Subjects ℹ️",  # Added info emoji to button text
                        variant="primary",
                    )
                    gr.Markdown(
                        "*Click to copy subjects to main input for card generation*",
                        elem_classes="hint-text",
                    )

                # Existing output components
                with gr.Group() as cards_output:
                    gr.Markdown("### Generated Cards")

                    # Output Format Documentation
                    with gr.Accordion("Output Format", open=False):
                        gr.Markdown("""
                        The generated cards include:
                        
                        * **Index**: Unique identifier for each card
                        * **Topic**: The specific subtopic within your subject
                        * **Card_Type**: Type of card (basic or cloze)
                        * **Question**: Clear, focused question for the flashcard front
                        * **Answer**: Concise core answer
                        * **Explanation**: Detailed conceptual explanation
                        * **Example**: Practical implementation or code example
                        * **Prerequisites**: Required knowledge for this concept
                        * **Learning Outcomes**: What you should understand after mastering this card
                        * **Common Misconceptions**: Incorrect assumptions debunked with explanations
                        * **Difficulty**: Concept complexity level for optimal study sequencing
                        
                        Export options:
                        - **CSV**: Raw data for custom processing
                        - **Anki Deck**: Ready-to-use deck with formatted cards and metadata
                        """)

                        # Add near the output format documentation
                        with gr.Accordion("Example Card Format", open=False):
                            gr.Code(
                                label="Example Card",
                                value="""
{
    "front": {
        "question": "What is a PRIMARY KEY constraint in SQL?"
    },
    "back": {
        "answer": "A PRIMARY KEY constraint uniquely identifies each record in a table",
        "explanation": "A primary key serves as a unique identifier for each row in a database table. It enforces data integrity by ensuring that:\n1. Each value is unique\n2. No null values are allowed\n3. The value remains stable over time\n\nThis is fundamental for:\n- Establishing relationships between tables\n- Maintaining data consistency\n- Efficient data retrieval",
        "example": "-- Creating a table with a primary key\nCREATE TABLE Users (\n  user_id INT PRIMARY KEY,\n  username VARCHAR(50) NOT NULL,\n  email VARCHAR(100) UNIQUE\n);"
    },
    "metadata": {
        "prerequisites": ["Basic SQL table concepts", "Understanding of data types"],
        "learning_outcomes": ["Understand the purpose and importance of primary keys", "Know how to create and use primary keys"],
        "common_misconceptions": [
            "❌ Misconception: Primary keys must always be single columns\n✓ Reality: Primary keys can be composite (multiple columns)",
            "❌ Misconception: Primary keys must be integers\n✓ Reality: Any data type that ensures uniqueness can be used"
        ],
        "difficulty": "beginner"
    }
}
                                """,
                                language="json",
                            )

                    # Dataframe Output
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

                    # Export Controls
                    with gr.Group(elem_classes="export-group"):
                        gr.Markdown("#### Export Generated Cards")
                        with gr.Row():
                            export_csv_button = gr.Button(
                                "Export to CSV", variant="secondary"
                            )
                            export_anki_button = gr.Button(
                                "Export to Anki Deck (.apkg)", variant="secondary"
                            )
                        # Re-wrap File components in an invisible Row
                        with gr.Row(visible=False):
                            download_csv = gr.File(
                                label="Download CSV", interactive=False, visible=False
                            )
                            download_anki = gr.File(
                                label="Download Anki Deck",
                                interactive=False,
                                visible=False,
                            )

        # Add near the top of the Blocks
        with gr.Row():
            progress = gr.HTML(visible=False)
            total_cards = gr.Number(
                label="Total Cards Generated", value=0, visible=False
            )

        # Add JavaScript to handle mode switching
        def update_mode_visibility(mode):
            """Update component visibility based on selected mode and clear values"""
            is_subject = mode == "subject"
            is_path = mode == "path"

            # Clear values when switching modes
            if is_path:
                subject.value = ""  # Clear subject when switching to path mode
            else:
                description.value = (
                    ""  # Clear description when switching to subject mode
                )

            return {
                subject_mode: gr.update(visible=is_subject),
                path_mode: gr.update(visible=is_path),
                path_results: gr.update(visible=is_path),
                cards_output: gr.update(visible=not is_path),
                subject: gr.update(value="") if is_path else gr.update(),
                description: gr.update(value="") if not is_path else gr.update(),
                output: gr.update(value=None),  # Clear previous output
                progress: gr.update(value="", visible=False),
                total_cards: gr.update(value=0, visible=False),
            }

        # Update the mode switching handler to include all components that need clearing
        generation_mode.change(
            fn=update_mode_visibility,
            inputs=[generation_mode],
            outputs=[
                subject_mode,
                path_mode,
                path_results,
                cards_output,
                subject,
                description,
                output,
                progress,
                total_cards,
            ],
        )

        # Add handler for path analysis
        analyze_button.click(
            fn=analyze_learning_path,
            inputs=[api_key_input, description, model_choice],
            outputs=[subjects_list, learning_order, projects],
        )

        # Add this function to handle copying subjects to main input
        def use_selected_subjects(subjects_df):
            """Copy selected subjects to main input and switch to subject mode"""
            if subjects_df is None or subjects_df.empty:
                gr.Warning("No subjects available to copy from Learning Path analysis.")
                # Return updates for all relevant output components to avoid errors
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

            subjects = subjects_df["Subject"].tolist()
            combined_subject = ", ".join(subjects)
            suggested_topics = min(
                len(subjects) + 1, 20
            )  # Suggest topics = num subjects + 1

            # Return updates for relevant components
            return (
                "subject",  # Set mode to subject
                gr.update(visible=True),  # Show subject_mode group
                gr.update(visible=False),  # Hide path_mode group
                gr.update(visible=False),  # Hide path_results group
                gr.update(visible=True),  # Show cards_output group
                combined_subject,  # Update subject textbox value
                suggested_topics,  # Update topic_number slider value
                # Update preference prompt
                "Focus on connections between these subjects and their practical applications.",
                example_data,  # Reset output to example data - THIS NOW WORKS
            )

        # Correct the outputs for the use_subjects click handler
        use_subjects.click(
            fn=use_selected_subjects,
            inputs=[subjects_list],  # Only needs the dataframe
            outputs=[  # Match the return tuple of the function
                generation_mode,
                subject_mode,  # Group visibility
                path_mode,  # Group visibility
                path_results,  # Group visibility
                cards_output,  # Group visibility
                subject,  # Component value
                topic_number,  # Component value
                preference_prompt,  # Component value
                output,  # Component value
            ],
        )

        # Simplified event handlers
        generate_button.click(
            fn=generate_cards,
            inputs=[
                api_key_input,
                subject,
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

if __name__ == "__main__":
    logger.info("Starting AnkiGen application")
    ankigen.launch(share=False, favicon_path="./favicon.ico")
