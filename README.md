---
title: AnkiGen
emoji: 📚
app_file: app.py
requirements: requirements.txt
python: 3.12
sdk: gradio
sdk_version: 5.13.1
---

# AnkiGen - Anki Card Generator

AnkiGen is a Gradio-based web application that generates Anki-compatible CSV and `.apkg` deck files using Large Language Models (LLMs) based on user-specified subjects and preferences.

## Features

- Generate Anki cards for various subjects or from provided text/URLs.
- Generate a structured learning path for a complex topic.
- Customizable number of topics and cards per topic.
- User-friendly interface powered by Gradio.
- Exports to CSV for manual import or `.apkg` format with default styling.
- Utilizes OpenAI's structured output capabilities.

## Screenshot

![AnkiGen Screenshot](example.png)

## Installation for Local Use

Preferred usage: [uv](https://github.com/astral-sh/uv)

1.  Clone this repository:

    ```bash
    git clone https://github.com/brickfrog/ankigen.git
    cd ankigen
    uv venv
    source .venv/bin/activate # Activate the virtual environment
    ```

2.  Install the required dependencies:

    ```bash
    uv pip install -e . # Install the package in editable mode
    ```

3.  Set up your OpenAI API key:
    - Create a `.env` file in the project root (`ankigen/`).
    - Add your key like this: `OPENAI_API_KEY="your_sk-xxxxxxxx_key_here"`
    - The application will load this key automatically.

## Usage

1.  Ensure your virtual environment is active (`source .venv/bin/activate`).

2.  Run the application:

    ```bash
    uv run python app.py
    ```
    *(Note: The `gradio app.py` command might also work but using `python app.py` within the `uv run` context is recommended.)*

3.  Open your web browser and navigate to the provided local URL (typically `http://127.0.0.1:7860`).

4.  In the application interface:
    - Your API key should be loaded automatically if using a `.env` file, otherwise enter it.
    - Select the desired generation mode ("Single Subject", "Learning Path", "From Text", "From Web").
    - Fill in the relevant inputs for the chosen mode.
    - Adjust generation parameters (model, number of topics/cards, preferences).
    - Click "Generate Cards" or "Analyze Learning Path".

5.  Review the generated output.

6.  For card generation, click "Export to CSV" or "Export to Anki Deck (.apkg)" to download the results.

## Project Structure

The codebase has been refactored from a single script into a more modular structure:

-   `app.py`: Main Gradio application interface and event handling.
-   `ankigen_core/`: Directory containing the core logic modules:
    -   `models.py`: Pydantic models for data structures.
    -   `utils.py`: Logging, caching, web fetching utilities.
    -   `llm_interface.py`: Interaction logic with the OpenAI API.
    -   `card_generator.py`: Core logic for generating topics and cards.
    -   `learning_path.py`: Logic for the learning path analysis feature.
    -   `exporters.py`: Functions for exporting data to CSV and `.apkg`.
    -   `ui_logic.py`: Functions handling UI component updates and visibility.
-   `tests/`: Contains unit and integration tests.
    -   `unit/`: Tests for individual modules in `ankigen_core`.
    -   `integration/`: Tests for interactions between modules and the app.
-   `pyproject.toml`: Defines project metadata, dependencies, and build system configuration.
-   `README.md`: This file.

## Development

This project uses `uv` for environment and package management and `pytest` for testing.

1.  **Setup:** Follow the Installation steps above.

2.  **Install Development Dependencies:**
    ```bash
    uv pip install -e ".[dev]"
    ```

3.  **Running Tests:**
    - To run all tests:
      ```bash
      uv run pytest tests/
      ```
    - To run with coverage:
      ```bash
      uv run pytest --cov=ankigen_core tests/
      ```
    *(Current test coverage target is >= 80%. As of the last run, coverage was ~89%.)*

4.  **Code Style:** Please use `black` and `ruff` for formatting and linting (configured in `pyproject.toml` implicitly via dev dependencies, can be run manually).

5.  **Making Changes:**
    - Core logic changes should primarily be made within the `ankigen_core` modules.
    - UI layout and event wiring are in `app.py`.
    - Add or update tests in the `tests/` directory for any new or modified functionality.

## TODO

- [ ] Edit columns /fields
- [ ] Improve crawler / RAG(?)
- [ ] Chain of thought / reasoning, recurse for better questions

## License

BSD 2-Clause License

## Acknowledgments

- This project uses the Gradio library (https://gradio.app/) for the web interface.
- Card generation is powered by OpenAI's language models.
