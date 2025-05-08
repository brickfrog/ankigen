# Module for CSV and APKG export functions

import gradio as gr
import pandas as pd
import genanki
import random
import tempfile

from ankigen_core.utils import get_logger

logger = get_logger()

# --- Anki Model Definitions --- (Moved from app.py)

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


# --- Export Functions --- (Moved from app.py)


def export_csv(data: pd.DataFrame | None):
    """Export the generated cards DataFrame as a CSV file string."""
    if data is None or data.empty:
        logger.warning("Attempted to export empty or None DataFrame to CSV.")
        raise gr.Error("No card data available to export. Please generate cards first.")

    # No minimum card check here, allow exporting even 1 card if generated.

    try:
        logger.info(f"Exporting DataFrame with {len(data)} rows to CSV format.")
        csv_string = data.to_csv(index=False)

        # Save to a temporary file to return its path to Gradio
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".csv", encoding="utf-8"
        ) as temp_file:
            temp_file.write(csv_string)
            csv_path = temp_file.name

        logger.info(f"CSV data prepared and saved to temporary file: {csv_path}")
        # Return the path for Gradio File component
        return csv_path

    except Exception as e:
        logger.error(f"Failed to export data to CSV: {str(e)}", exc_info=True)
        raise gr.Error(f"Failed to export to CSV: {str(e)}")


def export_deck(data: pd.DataFrame | None, subject: str | None):
    """Export the generated cards DataFrame as an Anki deck (.apkg file)."""
    if data is None or data.empty:
        logger.warning("Attempted to export empty or None DataFrame to Anki deck.")
        raise gr.Error("No card data available to export. Please generate cards first.")

    if not subject or not subject.strip():
        logger.warning("Subject name is empty, using default deck name.")
        deck_name = "AnkiGen Deck"
    else:
        deck_name = f"AnkiGen - {subject.strip()}"

    # No minimum card check here.

    try:
        logger.info(f"Creating Anki deck '{deck_name}' with {len(data)} cards.")

        deck_id = random.randrange(1 << 30, 1 << 31)
        deck = genanki.Deck(deck_id, deck_name)

        # Add models to the deck package
        deck.add_model(BASIC_MODEL)
        deck.add_model(CLOZE_MODEL)

        records = data.to_dict("records")

        for record in records:
            # Ensure necessary keys exist, provide defaults if possible
            card_type = str(record.get("Card_Type", "basic")).lower()
            question = str(record.get("Question", ""))
            answer = str(record.get("Answer", ""))
            explanation = str(record.get("Explanation", ""))
            example = str(record.get("Example", ""))
            prerequisites = str(
                record.get("Prerequisites", "[]")
            )  # Convert list/None to str
            learning_outcomes = str(record.get("Learning_Outcomes", "[]"))
            common_misconceptions = str(record.get("Common_Misconceptions", "[]"))
            difficulty = str(record.get("Difficulty", "N/A"))
            topic = str(record.get("Topic", "Unknown Topic"))

            if not question:
                logger.warning(f"Skipping record due to empty Question field: {record}")
                continue

            note = None
            if card_type == "cloze":
                # For Cloze, the main text goes into the first field ("Text")
                # All other details go into the second field ("Extra")
                extra_content = f"""<h3>Answer/Context:</h3> <div>{answer}</div><hr>
<h3>Explanation:</h3> <div>{explanation}</div><hr>
<h3>Example:</h3> <pre><code>{example}</code></pre><hr>
<h3>Prerequisites:</h3> <div>{prerequisites}</div><hr>
<h3>Learning Outcomes:</h3> <div>{learning_outcomes}</div><hr>
<h3>Common Misconceptions:</h3> <div>{common_misconceptions}</div>"""
                try:
                    note = genanki.Note(
                        model=CLOZE_MODEL,
                        fields=[question, extra_content, difficulty, topic],
                    )
                except Exception as e:
                    logger.error(
                        f"Error creating Cloze note: {e}. Record: {record}",
                        exc_info=True,
                    )
                    continue  # Skip this note

            else:  # Default to basic card
                try:
                    note = genanki.Note(
                        model=BASIC_MODEL,
                        fields=[
                            question,
                            answer,
                            explanation,
                            example,
                            prerequisites,
                            learning_outcomes,
                            common_misconceptions,
                            difficulty,
                        ],
                    )
                except Exception as e:
                    logger.error(
                        f"Error creating Basic note: {e}. Record: {record}",
                        exc_info=True,
                    )
                    continue  # Skip this note

            if note:
                deck.add_note(note)

        if not deck.notes:
            logger.warning("No valid notes were added to the deck. Export aborted.")
            raise gr.Error("Failed to create any valid Anki notes from the data.")

        # Create package in a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".apkg") as temp_file:
            apkg_path = temp_file.name
            package = genanki.Package(deck)
            package.write_to_file(apkg_path)

        logger.info(
            f"Anki deck '{deck_name}' created successfully at temporary path: {apkg_path}"
        )
        # Return the path for Gradio File component
        return apkg_path

    except Exception as e:
        logger.error(f"Failed to export Anki deck: {str(e)}", exc_info=True)
        raise gr.Error(f"Failed to export Anki deck: {str(e)}")
