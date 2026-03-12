# Main integration module for AnkiGen agent system

from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime


from ankigen.logging import logger
from ankigen.models import Card
from ankigen.llm_interface import OpenAIClientManager
from ankigen.context7 import Context7Client

from .generators import SubjectExpertAgent


class AgentOrchestrator:
    """Main orchestrator for the AnkiGen agent system"""

    def __init__(self, client_manager: OpenAIClientManager):
        self.client_manager = client_manager
        self.openai_client = None

        self.subject_expert = None

    async def initialize(
        self,
        api_key: str,
        model_overrides: Dict[str, str] = None,
        reasoning_overrides: Dict[str, Optional[str]] = None,
    ):
        """Initialize the agent system"""
        try:
            # Initialize OpenAI client
            await self.client_manager.initialize_client(api_key)
            self.openai_client = self.client_manager.get_client()

            # Set up model overrides if provided
            config_manager = None
            if model_overrides:
                from ankigen.agents.config import get_config_manager

                config_manager = get_config_manager()
                config_manager.update_models(model_overrides)
                logger.info(f"Applied model overrides: {model_overrides}")

            if reasoning_overrides:
                if config_manager is None:
                    from ankigen.agents.config import get_config_manager

                    config_manager = get_config_manager()
                for agent_name, effort in reasoning_overrides.items():
                    config_manager.update_agent_config(
                        agent_name, reasoning_effort=effort
                    )
                logger.info(f"Applied reasoning overrides: {reasoning_overrides}")

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
        generate_cloze: bool = False,
        topics_list: Optional[List[str]] = None,
        cards_per_topic: int = 8,
    ) -> Tuple[List[Card], Dict[str, Any]]:
        """Generate cards using the agent system.

        If topics_list is provided, generates cards for each subtopic separately
        to ensure comprehensive coverage. Otherwise falls back to single-topic mode.
        """
        start_time = datetime.now()

        try:
            if not self.openai_client:
                raise ValueError("Agent system not initialized")

            # Enhance context with library documentation if requested
            enhanced_context = context or {}
            library_docs = None

            if library_name:
                library_docs = await self._fetch_library_docs(
                    library_name, library_topic, num_cards
                )
                if library_docs:
                    enhanced_context["library_documentation"] = library_docs
                    enhanced_context["library_name"] = library_name

            # Generate cards - either per-topic or single-topic mode
            if topics_list and len(topics_list) > 0:
                logger.info(
                    f"Starting multi-topic generation: {len(topics_list)} topics, "
                    f"{cards_per_topic} cards each for '{topic}'"
                )
                cards = await self._generate_cards_per_topic(
                    main_subject=topic,
                    subject=subject,
                    topics_list=topics_list,
                    cards_per_topic=cards_per_topic,
                    difficulty=difficulty,
                    context=enhanced_context,
                    generate_cloze=generate_cloze,
                )
            else:
                # Fallback to single-topic mode
                logger.info(f"Starting single-topic generation: {topic} ({subject})")
                cards = await self._generation_phase(
                    topic=topic,
                    subject=subject,
                    num_cards=num_cards,
                    difficulty=difficulty,
                    context=enhanced_context,
                    generate_cloze=generate_cloze,
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
                "topics_list": topics_list,
                "multi_topic_mode": topics_list is not None and len(topics_list) > 0,
            }

            logger.info(
                f"Agent-based generation complete: {len(cards)} cards generated"
            )
            return cards, metadata

        except Exception as e:
            logger.error(f"Agent-based generation failed: {e}")
            raise

    async def _fetch_library_docs(
        self, library_name: str, library_topic: Optional[str], num_cards: int
    ) -> Optional[str]:
        """Fetch library documentation from Context7."""
        logger.info(f"Fetching library documentation for: {library_name}")
        try:
            context7_client = Context7Client()

            # Dynamic token allocation based on card generation needs
            base_tokens = 8000
            if num_cards > 40:
                token_limit = 12000
            elif num_cards > 20:
                token_limit = 10000
            else:
                token_limit = base_tokens

            if library_topic:
                token_limit = int(token_limit * 0.8)

            logger.info(
                f"Fetching {token_limit} tokens of documentation"
                + (f" for topic: {library_topic}" if library_topic else "")
            )

            library_docs = await context7_client.fetch_library_documentation(
                library_name, topic=library_topic, tokens=token_limit
            )

            if library_docs:
                logger.info(
                    f"Added {len(library_docs)} chars of {library_name} documentation to context"
                )
                return library_docs
            else:
                logger.warning(
                    f"Could not fetch documentation for library: {library_name}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching library documentation: {e}")
            return None

    async def _generate_cards_per_topic(
        self,
        main_subject: str,
        subject: str,
        topics_list: List[str],
        cards_per_topic: int,
        difficulty: str,
        context: Dict[str, Any],
        generate_cloze: bool,
    ) -> List[Card]:
        """Generate cards for each topic in the topics_list."""
        all_cards: List[Card] = []
        total_topics = len(topics_list)

        for i, subtopic in enumerate(topics_list):
            topic_num = i + 1
            logger.info(
                f"Generating topic {topic_num}/{total_topics}: {subtopic} "
                f"({cards_per_topic} cards)"
            )

            # Add topic context
            topic_context = {
                **context,
                "main_subject": main_subject,
                "topic_index": topic_num,
                "total_topics": total_topics,
                "current_subtopic": subtopic,
            }

            cards = await self._generation_phase(
                topic=subtopic,
                subject=subject,
                num_cards=cards_per_topic,
                difficulty=difficulty,
                context=topic_context,
                generate_cloze=generate_cloze,
            )

            all_cards.extend(cards)
            logger.info(
                f"Topic {topic_num}/{total_topics} complete: {len(cards)} cards. "
                f"Total: {len(all_cards)}"
            )

        return all_cards

    async def _generation_phase(
        self,
        topic: str,
        subject: str,
        num_cards: int,
        difficulty: str,
        context: Dict[str, Any] = None,
        generate_cloze: bool = False,
    ) -> List[Card]:
        """Execute the card generation phase"""

        if not self.subject_expert or self.subject_expert.subject != subject:
            self.subject_expert = SubjectExpertAgent(self.openai_client, subject)

        # Add difficulty and cloze preference to context
        if context is None:
            context = {}
        context["difficulty"] = difficulty
        context["generate_cloze"] = generate_cloze

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
