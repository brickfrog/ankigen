"""CLI interface for AnkiGen - Generate Anki flashcards from the command line"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

from ankigen_core.agents.token_tracker import get_token_tracker
from ankigen_core.auto_config import AutoConfigService
from ankigen_core.card_generator import orchestrate_card_generation
from ankigen_core.exporters import export_dataframe_to_apkg, export_dataframe_to_csv
from ankigen_core.llm_interface import OpenAIClientManager
from ankigen_core.utils import ResponseCache, get_logger

console = Console()
logger = get_logger()


def get_api_key() -> str:
    """Get OpenAI API key from env or prompt user"""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        console.print("[yellow]OpenAI API key not found in environment[/yellow]")
        api_key = click.prompt("Enter your OpenAI API key", hide_input=True)

    return api_key


async def auto_configure_from_prompt(
    prompt: str,
    api_key: str,
    override_topics: Optional[int] = None,
    override_cards: Optional[int] = None,
    override_model: Optional[str] = None,
) -> dict:
    """Auto-configure settings from a prompt using AI analysis"""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Analyzing subject...", total=None)

        # Initialize client
        client_manager = OpenAIClientManager()
        await client_manager.initialize_client(api_key)
        openai_client = client_manager.get_client()

        # Get auto-config (pass topic count override so LLM decomposes correctly)
        auto_config_service = AutoConfigService()
        config = await auto_config_service.auto_configure(
            prompt, openai_client, target_topic_count=override_topics
        )

    # Apply remaining overrides (topics already handled in auto_configure)
    if override_cards is not None:
        config["cards_per_topic"] = override_cards
    if override_model is not None:
        config["model_choice"] = override_model

    # Display configuration
    table = Table(
        title="Auto-Configuration", show_header=True, header_style="bold cyan"
    )
    table.add_column("Setting", style="dim")
    table.add_column("Value", style="green")

    table.add_row("Topics", str(config.get("topic_number", "N/A")))
    table.add_row("Cards per Topic", str(config.get("cards_per_topic", "N/A")))
    table.add_row(
        "Total Cards",
        str(config.get("topic_number", 0) * config.get("cards_per_topic", 0)),
    )
    table.add_row("Model", config.get("model_choice", "N/A"))

    if config.get("library_name"):
        table.add_row("Library", config.get("library_name"))
    if config.get("library_topic"):
        table.add_row("Library Topic", config.get("library_topic"))

    # Display discovered topics
    if config.get("topics_list"):
        topics = config["topics_list"]
        # Show first few topics, indicate if there are more
        if len(topics) <= 4:
            topics_str = ", ".join(topics)
        else:
            topics_str = ", ".join(topics[:3]) + f", ... (+{len(topics) - 3} more)"
        table.add_row("Subtopics", topics_str)

    if config.get("preference_prompt"):
        table.add_row(
            "Learning Focus", config.get("preference_prompt", "")[:50] + "..."
        )

    console.print(table)

    return config


async def generate_cards_from_config(
    prompt: str,
    config: dict,
    api_key: str,
) -> tuple:
    """Generate cards using the configuration"""

    client_manager = OpenAIClientManager()
    response_cache = ResponseCache()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Generating {config['topic_number'] * config['cards_per_topic']} cards...",
            total=100,
        )

        # Generate cards
        (
            output_df,
            total_cards_html,
            token_usage_html,
        ) = await orchestrate_card_generation(
            client_manager=client_manager,
            cache=response_cache,
            api_key_input=api_key,
            subject=prompt,
            generation_mode="subject",
            source_text="",
            url_input="",
            model_name=config.get("model_choice", "gpt-5.1"),
            topic_number=config.get("topic_number", 3),
            cards_per_topic=config.get("cards_per_topic", 5),
            preference_prompt=config.get("preference_prompt", ""),
            generate_cloze=config.get("generate_cloze_checkbox", False),
            library_name=config.get("library_name")
            if config.get("library_name")
            else None,
            library_topic=config.get("library_topic")
            if config.get("library_topic")
            else None,
            topics_list=config.get("topics_list"),
        )

        progress.update(task, completed=100)

    return output_df, total_cards_html, token_usage_html


def export_cards(
    df: pd.DataFrame,
    output_path: str,
    deck_name: str,
    export_format: str = "apkg",
) -> str:
    """Export cards to file"""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Exporting to {export_format.upper()}...", total=None)

        if export_format == "apkg":
            # Ensure .apkg extension
            if not output_path.endswith(".apkg"):
                output_path = (
                    output_path.replace(".csv", ".apkg")
                    if ".csv" in output_path
                    else f"{output_path}.apkg"
                )

            exported_path = export_dataframe_to_apkg(df, output_path, deck_name)
        else:  # csv
            # Ensure .csv extension
            if not output_path.endswith(".csv"):
                output_path = (
                    output_path.replace(".apkg", ".csv")
                    if ".apkg" in output_path
                    else f"{output_path}.csv"
                )

            exported_path = export_dataframe_to_csv(df, output_path)

    return exported_path


@click.command()
@click.option(
    "-p",
    "--prompt",
    required=True,
    help="Subject or topic for flashcard generation (e.g., 'Basic SQL', 'React Hooks')",
)
@click.option(
    "--topics",
    type=int,
    help="Number of topics (auto-detected if not specified)",
)
@click.option(
    "--cards-per-topic",
    type=int,
    help="Number of cards per topic (auto-detected if not specified)",
)
@click.option(
    "--model",
    type=click.Choice(
        ["gpt-5.1", "gpt-4.1", "gpt-4.1-nano"],
        case_sensitive=False,
    ),
    help="Model to use for generation (auto-selected if not specified)",
)
@click.option(
    "-o",
    "--output",
    default="deck.apkg",
    help="Output file path (default: deck.apkg)",
)
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["apkg", "csv"], case_sensitive=False),
    default="apkg",
    help="Export format (default: apkg)",
)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (or set OPENAI_API_KEY env var)",
)
@click.option(
    "--no-confirm",
    is_flag=True,
    help="Skip confirmation prompt",
)
def main(
    prompt: str,
    topics: Optional[int],
    cards_per_topic: Optional[int],
    model: Optional[str],
    output: str,
    export_format: str,
    api_key: Optional[str],
    no_confirm: bool,
):
    """
    AnkiGen CLI - Generate Anki flashcards from the command line

    Examples:

      # Quick generation with auto-config
      ankigen -p "Basic SQL"

      # With custom settings
      ankigen -p "React Hooks" --topics 5 --cards-per-topic 8 --output hooks.apkg

      # Export to CSV
      ankigen -p "Docker basics" --format csv -o docker.csv
    """

    # Print header
    console.print(
        Panel.fit(
            "[bold cyan]AnkiGen CLI[/bold cyan]\n[dim]Generate Anki flashcards with AI[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # Get API key
    if not api_key:
        api_key = get_api_key()

    # Run async workflow
    async def workflow():
        try:
            # Step 1: Auto-configure
            console.print(f"[bold]Subject:[/bold] {prompt}\n")
            config = await auto_configure_from_prompt(
                prompt=prompt,
                api_key=api_key,
                override_topics=topics,
                override_cards=cards_per_topic,
                override_model=model,
            )

            # Step 2: Confirm (unless --no-confirm)
            if not no_confirm:
                console.print()
                if not click.confirm("Proceed with card generation?", default=True):
                    console.print("[yellow]Cancelled[/yellow]")
                    return

            console.print()

            # Step 3: Generate cards
            df, total_html, token_html = await generate_cards_from_config(
                prompt=prompt,
                config=config,
                api_key=api_key,
            )

            if df.empty:
                console.print("[red]✗[/red] No cards generated")
                sys.exit(1)

            # Step 4: Export
            console.print()
            deck_name = f"AnkiGen - {prompt}"
            exported_path = export_cards(
                df=df,
                output_path=output,
                deck_name=deck_name,
                export_format=export_format,
            )

            # Step 5: Success summary
            console.print()
            file_size = Path(exported_path).stat().st_size / 1024  # KB

            summary = Table.grid(padding=(0, 2))
            summary.add_row("[green]✓[/green] Success!", "")
            summary.add_row("Cards Generated:", f"[bold]{len(df)}[/bold]")
            summary.add_row("Output File:", f"[bold]{exported_path}[/bold]")
            summary.add_row("File Size:", f"{file_size:.1f} KB")

            # Get token usage from tracker
            tracker = get_token_tracker()
            session = tracker.get_session_summary()
            if session["total_tokens"] > 0:
                # Calculate totals across all models
                total_input = sum(u.prompt_tokens for u in tracker.usage_history)
                total_output = sum(u.completion_tokens for u in tracker.usage_history)
                summary.add_row(
                    "Tokens:",
                    f"{total_input:,} in / {total_output:,} out ({session['total_tokens']:,} total)",
                )

            console.print(
                Panel(summary, border_style="green", title="Generation Complete")
            )

        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled by user[/yellow]")
            sys.exit(130)
        except Exception as e:
            logger.error(f"CLI error: {e}", exc_info=True)
            console.print(f"[red]✗ Error:[/red] {str(e)}")
            sys.exit(1)

    # Run the async workflow
    asyncio.run(workflow())


if __name__ == "__main__":
    main()
