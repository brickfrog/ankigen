---
title: AnkiGen
emoji: 📚
app_file: app.py
requirements: requirements.txt
python_version: 3.12
sdk: gradio
sdk_version: 6.9.0
---

# AnkiGen - Anki Card Generator

AnkiGen generates Anki flashcards using OpenAI's GPT models. Available as both a web interface (Gradio) and command-line tool, it creates CSV and `.apkg` deck files with intelligent auto-configuration.

## Features

- Generate Anki cards for various subjects with automatic topic decomposition
- AI-powered auto-configuration (intelligently determines topics, card counts, and models)
- Export to CSV or `.apkg` format with default styling
- Customizable number of topics and cards per topic
- Built-in quality review system
- **CLI for quick terminal-based generation**
- **Web interface for interactive use**

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
   # Base installation (web interface)
   uv pip install -e .

   # With CLI support
   uv pip install -e ".[cli]"
   ```

3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

## Usage

### CLI (Quick & Direct)

Generate flashcards directly from your terminal with intelligent auto-configuration:

```bash
# Quick generation (auto-detects best settings)
uv run python -m ankigen_core.cli -p "Basic SQL"

# Custom settings
uv run python -m ankigen_core.cli -p "React Hooks" \
  --topics 5 \
  --cards-per-topic 8 \
  --output hooks.apkg

# Export to CSV
uv run python -m ankigen_core.cli -p "Docker basics" \
  --format csv \
  -o docker.csv

# Skip confirmation prompt
uv run python -m ankigen_core.cli -p "Python Lists" --no-confirm
```

**CLI Options:**
- `-p, --prompt`: Subject/topic (required)
- `--topics`: Number of topics (auto-detected if omitted)
- `--cards-per-topic`: Cards per topic (auto-detected if omitted)
- `--model`: Model choice (`gpt-5.2-auto`, `gpt-5.2-instant`, or `gpt-5.2-thinking`)
- `-o, --output`: Output file path
- `--format`: Export format (`apkg` or `csv`)
- `--no-confirm`: Skip confirmation prompt

### Web Interface (Interactive)

1. Run the application:
   ```bash
   uv run python app.py
   ```

2. Open your browser to `http://127.0.0.1:7860`

3. Enter a subject and optionally click "Auto-fill" to configure settings

4. Configure parameters and click "Generate Cards"

5. Export results as CSV or `.apkg` file

## Project Structure

- `app.py`: Main Gradio web application
- `ankigen_core/`: Core logic modules
  - `cli.py`: Command-line interface
  - `agents/`: Agent system implementation
  - `card_generator.py`: Card generation orchestration
  - `auto_config.py`: AI-powered auto-configuration
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
