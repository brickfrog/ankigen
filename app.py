from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
import gradio as gr


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
    """
    Generates flashcards for a given subject.

    Parameters:
    - subject (str): The subject to generate cards for.
    - topic_number (int): Number of topics to generate.
    - cards_per_topic (int): Number of cards per topic.
    - preference_prompt (str): User preferences to consider.

    Returns:
    - List[List[str]]: A list of rows containing
    [topic, question, answer, explanation, example].
    """

    gr.Info("Starting process")

    if not api_key_input:
        return gr.Error("Error: OpenAI API key is required.")

    client = OpenAI(api_key=api_key_input)
    model = "gpt-4o-mini"

    all_card_lists = []

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
            print("Failed to generate topics.")
            return []
        if not hasattr(topics_response, "result") or not topics_response.result:
            print("Invalid topics response format.")
            return []
        topic_list = [
            item for subtopic in topics_response.result for item in subtopic.result
        ][:topic_number]
    except Exception as e:
        raise gr.Error(f"Topic generation failed due to {e}")

    for topic in topic_list:
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
                print(f"Failed to generate cards for topic '{topic}'.")
                continue
            if not hasattr(cards, "topic") or not hasattr(cards, "cards"):
                print(f"Invalid card response format for topic '{topic}'.")
                continue
            all_card_lists.append(cards)
        except Exception as e:
            print(f"An error occurred while generating cards for topic '{topic}': {e}")
            continue

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

    return flattened_data


def export_csv(d):
    MIN_ROWS = 2

    if len(d) < MIN_ROWS:
        gr.Warning(f"The dataframe has fewer than {MIN_ROWS} rows. Nothing to export.")
        return None

    gr.Info("Exporting...")
    d.to_csv("anki_deck.csv", index=False)
    return gr.File(value="anki_deck.csv", visible=True)


with gr.Blocks(
    gr.themes.Soft(), title="AnkiGen", css="footer{display:none !important}"
) as ankigen:
    gr.Markdown("# 📚 AnkiGen - Anki Card Generator")
    gr.Markdown(
        """
        #### Generate an Anki comptible .csv using LLMs based on your subject and preferences.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")

            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="Enter your OpenAI API key",
            )
            subject = gr.Textbox(
                label="Subject",
                placeholder="Enter the subject, e.g., 'Basic SQL Concepts'",
            )
            topic_number = gr.Slider(
                label="Number of Topics", minimum=2, maximum=20, step=1, value=2
            )
            cards_per_topic = gr.Slider(
                label="Cards per Topic", minimum=2, maximum=30, step=1, value=3
            )
            preference_prompt = gr.Textbox(
                label="Preference Prompt",
                placeholder="Any preferences? e.g., 'Assume I'm a beginner'",
            )
            generate_button = gr.Button("Generate Cards")
        with gr.Column(scale=2):
            gr.Markdown("### Generated Cards")
            gr.Markdown(
                """
                Subject to change: currently exports a .csv with the following fields, you can
                create a new note type with these fields to handle importing.: 
                <b>Index, Topic, Question, Answer, Explanation, Example</b>
                """)
            output = gr.Dataframe(
                headers=[
                    "Index",
                    "Topic",
                    "Question",
                    "Answer",
                    "Explanation",
                    "Example",
                ],
                interactive=False,
                height=800,
            )
            export_button = gr.Button("Export to CSV")
            download_link = gr.File(interactive=False, visible=False)

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
    )

    export_button.click(fn=export_csv, inputs=output, outputs=download_link)

if __name__ == "__main__":
    ankigen.launch(share=False, favicon_path="./favicon.ico")
