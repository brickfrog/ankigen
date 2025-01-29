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

AnkiGen is a Gradio-based web application that generates Anki-compatible CSV files using Large Language Models (LLMs) based on user-specified subjects and preferences.

## Features

- Generate Anki cards for various subjects
- Customizable number of topics and cards per topic
- User-friendly interface powered by Gradio
- Exports to CSV for manual import or .apkg format with out of the box css styling
- Utilizes OpenAI's structured output to mimic chain of thought to minimize hallucinations

## TODO

- [ ] cloze cards? (checkbox?)
- [ ] File upload / parsing longer texts / books as input?

## Screenshot

![AnkiGen Screenshot](example.png)


## Installation for Local Use

Preferred usage: [uv](https://github.com/astral-sh/uv)

1. Clone this repository:

```bash
git clone https://github.com/brickfrog/ankigen.git
cd ankigen
uv venv
```


2. Install the required dependencies:

```bash
uv pip install -r requirements.txt
```

3. Set up your OpenAI API key (required for LLM functionality).

## Usage

1. Run the application:

```bash
uv run gradio app.py --demo-name ankigen
```

2. Open your web browser and navigate to the provided local URL (typically `http://127.0.0.1:7860`).

3. In the application interface:
- Enter your OpenAI API key
- Specify the subject you want to create cards for
- Adjust the number of topics and cards per topic
- (Optional) Add any preference prompts
- Click "Generate Cards"

4. Review the generated cards in the interface.

5. Click "Export to CSV" to download the Anki-compatible file or Export to  Anki Deck to export as a .apkg that can be imported into Anki.

## Development

This project is built with:
- Python 3.12
- Gradio 5.13.1

To contribute or modify:
1. Make your changes in `app.py`
2. Update `requirements.txt` if you add new dependencies
3. Test thoroughly before submitting pull requests

## License

BSD 2.0

## Acknowledgments

- This project uses the Gradio library (https://gradio.app/) for the web interface
- Card generation is powered by OpenAI's language models