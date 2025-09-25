---
title: AnkiGen
emoji: 📚
app_file: app.py
requirements: requirements.txt
python_version: 3.12
sdk: gradio
sdk_version: 5.38.1
---

# AnkiGen - Anki Card Generator

AnkiGen is a Gradio-based web application that generates high-quality Anki flashcards using OpenAI's GPT models. It creates CSV and `.apkg` deck files with intelligent subject-specific card generation and quality review.

## Features

- Generate Anki cards for various subjects or from provided text/URLs
- Create structured learning paths for complex topics
- Export to CSV or `.apkg` format with default styling
- Customizable number of topics and cards per topic
- Built-in quality review system
- User-friendly Gradio interface

## Installation

Preferred usage: [uv](https://github.com/astral-sh/uv)

1. Clone this repository:
   ```bash
   git clone https://github.com/brickfrog/ankigen.git
   cd ankigen
   uv venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   uv pip install -e .
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add: `OPENAI_API_KEY="your_api_key_here"`

## Usage

1. Run the application:
   ```bash
   uv run python app.py
   ```

2. Open your browser to `http://127.0.0.1:7860`

3. Select a generation mode:
   - Single Subject: Generate cards for a specific topic
   - Learning Path: Create a structured learning curriculum
   - From Text: Generate cards from pasted text
   - From Web: Generate cards from a URL

4. Configure parameters and click "Generate Cards"

5. Export results as CSV or `.apkg` file

## Project Structure

- `app.py`: Main Gradio application
- `ankigen_core/`: Core logic modules
  - `agents/`: Agent system implementation
  - `card_generator.py`: Card generation orchestration
  - `learning_path.py`: Learning path analysis
  - `exporters.py`: CSV and `.apkg` export functionality
  - `models.py`: Data structures
- `tests/`: Unit and integration tests

## Development

1. Install development dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```

2. Run tests:
   ```bash
   uv run pytest tests/
   ```

3. Run with coverage:
   ```bash
   uv run pytest --cov=ankigen_core tests/
   ```

## License

BSD 2-Clause License

## Acknowledgments

- Gradio library for the web interface
- OpenAI for GPT models
- Card design principles from ["An Opinionated Guide to Using Anki Correctly"](https://www.lesswrong.com/posts/7Q7DPSk4iGFJd8DRk/an-opinionated-guide-to-using-anki-correctly)