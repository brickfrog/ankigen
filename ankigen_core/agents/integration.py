# Main integration module for AnkiGen agent system

from typing import List, Dict, Any, Tuple
from datetime import datetime


from ankigen_core.logging import logger
from ankigen_core.models import Card
from ankigen_core.llm_interface import OpenAIClientManager

from .generators import GenerationCoordinator, SubjectExpertAgent
from .judges import JudgeCoordinator
from .enhancers import RevisionAgent, EnhancementAgent


class AgentOrchestrator:
    """Main orchestrator for the AnkiGen agent system"""

    def __init__(self, client_manager: OpenAIClientManager):
        self.client_manager = client_manager
        self.openai_client = None

        # Initialize coordinators
        self.generation_coordinator = None
        self.judge_coordinator = None
        self.revision_agent = None
        self.enhancement_agent = None

        # All agents enabled by default
        self.all_agents_enabled = True

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

            # Initialize all agents
            self.generation_coordinator = GenerationCoordinator(self.openai_client)
            self.judge_coordinator = JudgeCoordinator(self.openai_client)
            self.revision_agent = RevisionAgent(self.openai_client)
            self.enhancement_agent = EnhancementAgent(self.openai_client)

            logger.info("Agent system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize agent system: {e}")
            raise

    async def generate_cards_with_agents(
        self,
        topic: str,
        subject: str = "general",
        num_cards: int = 5,
        difficulty: str = "intermediate",
        enable_quality_pipeline: bool = True,
        context: Dict[str, Any] = None,
    ) -> Tuple[List[Card], Dict[str, Any]]:
        """Generate cards using the agent system"""
        start_time = datetime.now()

        try:
            # Agents are always enabled now

            if not self.openai_client:
                raise ValueError("Agent system not initialized")

            logger.info(f"Starting agent-based card generation: {topic} ({subject})")

            # Phase 1: Generation
            cards = await self._generation_phase(
                topic=topic,
                subject=subject,
                num_cards=num_cards,
                difficulty=difficulty,
                context=context,
            )

            # Phase 2: Quality Assessment
            quality_results = {}
            if enable_quality_pipeline and self.judge_coordinator:
                cards, quality_results = await self._quality_phase(cards)

            # Phase 3: Enhancement
            if self.enhancement_agent:
                cards = await self._enhancement_phase(cards)

            # Collect metadata
            metadata = {
                "generation_method": "agent_system",
                "generation_time": (datetime.now() - start_time).total_seconds(),
                "cards_generated": len(cards),
                "quality_results": quality_results,
                "topic": topic,
                "subject": subject,
                "difficulty": difficulty,
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

        if self.generation_coordinator:
            # Use coordinated multi-agent generation
            cards = await self.generation_coordinator.coordinate_generation(
                topic=topic,
                subject=subject,
                num_cards=num_cards,
                difficulty=difficulty,
                enable_review=True,
                enable_structuring=True,
                context=context,
            )
        else:
            # Use subject expert agent directly
            subject_expert = SubjectExpertAgent(self.openai_client, subject)
            cards = await subject_expert.generate_cards(
                topic=topic, num_cards=num_cards, difficulty=difficulty, context=context
            )

        logger.info(f"Generation phase complete: {len(cards)} cards generated")
        return cards

    async def _quality_phase(
        self, cards: List[Card]
    ) -> Tuple[List[Card], Dict[str, Any]]:
        """Execute the quality assessment and improvement phase"""

        if not self.judge_coordinator:
            return cards, {"message": "Judge coordinator not available"}

        logger.info(f"Starting quality assessment for {len(cards)} cards")

        # Judge all cards
        judge_results = await self.judge_coordinator.coordinate_judgment(
            cards=cards,
            enable_parallel=True,
            min_consensus=0.6,
        )

        # Separate approved and rejected cards
        approved_cards = []
        rejected_cards = []

        for card, decisions, approved in judge_results:
            if approved:
                approved_cards.append(card)
            else:
                rejected_cards.append((card, decisions))

        # Attempt to revise rejected cards
        revised_cards = []
        if self.revision_agent and rejected_cards:
            logger.info(f"Attempting to revise {len(rejected_cards)} rejected cards")

            for card, decisions in rejected_cards:
                try:
                    revised_card = await self.revision_agent.revise_card(
                        card=card,
                        judge_decisions=decisions,
                        max_iterations=2,
                    )

                    # Re-judge the revised card
                    revision_results = await self.judge_coordinator.coordinate_judgment(
                        cards=[revised_card],
                        enable_parallel=False,  # Single card, no need for parallel
                        min_consensus=0.6,
                    )

                    if revision_results and revision_results[0][2]:  # If approved
                        revised_cards.append(revised_card)
                    else:
                        logger.warning(
                            f"Revised card still rejected: {card.front.question[:50]}..."
                        )

                except Exception as e:
                    logger.error(f"Failed to revise card: {e}")

        # Combine approved and successfully revised cards
        final_cards = approved_cards + revised_cards

        # Prepare quality results
        quality_results = {
            "total_cards_judged": len(cards),
            "initially_approved": len(approved_cards),
            "initially_rejected": len(rejected_cards),
            "successfully_revised": len(revised_cards),
            "final_approval_rate": len(final_cards) / len(cards) if cards else 0,
            "judge_decisions": len(judge_results),
        }

        logger.info(
            f"Quality phase complete: {len(final_cards)}/{len(cards)} cards approved"
        )
        return final_cards, quality_results

    async def _enhancement_phase(self, cards: List[Card]) -> List[Card]:
        """Execute the enhancement phase"""

        if not self.enhancement_agent:
            return cards

        logger.info(f"Starting enhancement for {len(cards)} cards")

        enhanced_cards = await self.enhancement_agent.enhance_card_batch(
            cards=cards, enhancement_targets=["explanation", "example", "metadata"]
        )

        logger.info(f"Enhancement phase complete: {len(enhanced_cards)} cards enhanced")
        return enhanced_cards

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
