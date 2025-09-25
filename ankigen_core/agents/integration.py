# Main integration module for AnkiGen agent system

from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime


from ankigen_core.logging import logger
from ankigen_core.models import Card
from ankigen_core.llm_interface import OpenAIClientManager
from ankigen_core.context7 import Context7Client

from .generators import SubjectExpertAgent


class AgentOrchestrator:
    """Main orchestrator for the AnkiGen agent system"""

    def __init__(self, client_manager: OpenAIClientManager):
        self.client_manager = client_manager
        self.openai_client = None

        self.subject_expert = None

    async def initialize(self, api_key: str, model_overrides: Dict[str, str] = None):
        """Initialize the agent system"""
        try:
            # Initialize OpenAI client
            await self.client_manager.initialize_client(api_key)
            self.openai_client = self.client_manager.get_client()

            # Set up model overrides if provided
            if model_overrides:
                from ankigen_core.agents.config import get_config_manager

                config_manager = get_config_manager()
                config_manager.update_models(model_overrides)
                logger.info(f"Applied model overrides: {model_overrides}")

            logger.info("Agent system initialized successfully (simplified pipeline)")

        except Exception as e:
            logger.error(f"Failed to initialize agent system: {e}")
            raise

    async def generate_cards_with_agents(
        self,
        topic: str,
        subject: str = "general",
        num_cards: int = 5,
        difficulty: str = "intermediate",
        context: Dict[str, Any] = None,
        library_name: Optional[str] = None,
        library_topic: Optional[str] = None,
    ) -> Tuple[List[Card], Dict[str, Any]]:
        """Generate cards using the agent system"""
        start_time = datetime.now()

        try:
            if not self.openai_client:
                raise ValueError("Agent system not initialized")

            logger.info(f"Starting agent-based card generation: {topic} ({subject})")

            # Enhance context with library documentation if requested
            enhanced_context = context or {}
            library_docs = None

            if library_name:
                logger.info(f"Fetching library documentation for: {library_name}")
                try:
                    context7_client = Context7Client()

                    # Dynamic token allocation based on card generation needs
                    # More cards need more comprehensive documentation
                    base_tokens = 8000  # Increased base from 5000
                    if num_cards > 40:
                        token_limit = 12000  # Large card sets need more context
                    elif num_cards > 20:
                        token_limit = 10000  # Medium sets
                    else:
                        token_limit = base_tokens  # Small sets

                    # If topic is specified, we can be more focused and use fewer tokens
                    if library_topic:
                        token_limit = int(
                            token_limit * 0.8
                        )  # Can be more efficient with focused retrieval

                    logger.info(
                        f"Fetching {token_limit} tokens of documentation"
                        + (f" for topic: {library_topic}" if library_topic else "")
                    )

                    library_docs = await context7_client.fetch_library_documentation(
                        library_name, topic=library_topic, tokens=token_limit
                    )

                    if library_docs:
                        enhanced_context["library_documentation"] = library_docs
                        enhanced_context["library_name"] = library_name
                        logger.info(
                            f"Added {len(library_docs)} chars of {library_name} documentation to context"
                        )
                    else:
                        logger.warning(
                            f"Could not fetch documentation for library: {library_name}"
                        )
                except Exception as e:
                    logger.error(f"Error fetching library documentation: {e}")

            cards = await self._generation_phase(
                topic=topic,
                subject=subject,
                num_cards=num_cards,
                difficulty=difficulty,
                context=enhanced_context,
            )

            # Collect metadata
            metadata = {
                "generation_method": "agent_system",
                "generation_time": (datetime.now() - start_time).total_seconds(),
                "cards_generated": len(cards),
                "topic": topic,
                "subject": subject,
                "difficulty": difficulty,
                "library_name": library_name if library_name else None,
                "library_docs_used": bool(library_docs),
            }

            logger.info(
                f"Agent-based generation complete: {len(cards)} cards generated"
            )
            return cards, metadata

        except Exception as e:
            logger.error(f"Agent-based generation failed: {e}")
            raise

    async def _generation_phase(
        self,
        topic: str,
        subject: str,
        num_cards: int,
        difficulty: str,
        context: Dict[str, Any] = None,
    ) -> List[Card]:
        """Execute the card generation phase"""

        if not self.subject_expert or self.subject_expert.subject != subject:
            self.subject_expert = SubjectExpertAgent(self.openai_client, subject)

        # Add difficulty to context if needed
        if context is None:
            context = {}
        context["difficulty"] = difficulty

        cards = await self.subject_expert.generate_cards(
            topic=topic, num_cards=num_cards, context=context
        )

        logger.info(f"Generation phase complete: {len(cards)} cards generated")
        return cards

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the agent system"""

        # Basic performance info only
        return {
            "agents_enabled": True,
        }


async def integrate_with_existing_workflow(
    client_manager: OpenAIClientManager, api_key: str, **generation_params
) -> Tuple[List[Card], Dict[str, Any]]:
    """Integration point for existing AnkiGen workflow"""

    # Agents are always enabled

    # Initialize and use agent system
    orchestrator = AgentOrchestrator(client_manager)
    await orchestrator.initialize(api_key)

    cards, metadata = await orchestrator.generate_cards_with_agents(**generation_params)

    return cards, metadata
