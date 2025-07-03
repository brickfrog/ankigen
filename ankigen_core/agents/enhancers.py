# Enhancement agents for card revision and improvement

import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from openai import AsyncOpenAI

from ankigen_core.logging import logger
from ankigen_core.models import Card, CardFront, CardBack
from .base import BaseAgentWrapper, AgentConfig
from .config import get_config_manager
from .metrics import record_agent_execution
from .judges import JudgeDecision


class RevisionAgent(BaseAgentWrapper):
    """Agent for revising cards based on judge feedback"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        config_manager = get_config_manager()
        base_config = config_manager.get_agent_config("revision_agent")
        
        if not base_config:
            base_config = AgentConfig(
                name="revision_agent",
                instructions="""You are a content revision specialist.
Improve flashcards based on specific feedback from quality judges.
Make targeted improvements while maintaining educational intent.""",
                model="gpt-4o",
                temperature=0.6
            )
        
        super().__init__(base_config, openai_client)
    
    async def revise_card(
        self,
        card: Card,
        judge_decisions: List[JudgeDecision],
        max_iterations: int = 3
    ) -> Card:
        """Revise a card based on judge feedback"""
        start_time = datetime.now()
        
        try:
            # Collect all feedback and improvements
            all_feedback = []
            all_improvements = []
            
            for decision in judge_decisions:
                if not decision.approved:
                    all_feedback.append(f"{decision.judge_name}: {decision.feedback}")
                    all_improvements.extend(decision.improvements)
            
            if not all_feedback:
                # No revisions needed
                return card
            
            # Build revision prompt
            user_input = self._build_revision_prompt(card, all_feedback, all_improvements)
            
            # Execute revision
            response = await self.execute(user_input)
            
            # Parse revised card
            revised_card = self._parse_revised_card(response, card)
            
            # Record successful execution
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                metadata={
                    "cards_revised": 1,
                    "feedback_sources": len(judge_decisions),
                    "improvements_applied": len(all_improvements)
                }
            )
            
            logger.info(f"RevisionAgent successfully revised card: {card.front.question[:50]}...")
            return revised_card
            
        except Exception as e:
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"RevisionAgent failed to revise card: {e}")
            return card  # Return original card on failure
    
    def _build_revision_prompt(
        self,
        card: Card,
        feedback: List[str],
        improvements: List[str]
    ) -> str:
        """Build the revision prompt"""
        feedback_str = "\n".join([f"- {fb}" for fb in feedback])
        improvements_str = "\n".join([f"- {imp}" for imp in improvements])
        
        return f"""Revise this flashcard based on the provided feedback and improvement suggestions:

Original Card:
Question: {card.front.question}
Answer: {card.back.answer}
Explanation: {card.back.explanation}
Example: {card.back.example}
Type: {card.card_type}
Metadata: {json.dumps(card.metadata, indent=2)}

Judge Feedback:
{feedback_str}

Specific Improvements Needed:
{improvements_str}

Instructions:
1. Address each piece of feedback specifically
2. Implement the suggested improvements
3. Maintain the educational intent and core content
4. Preserve correct information while fixing issues
5. Improve clarity, accuracy, and pedagogical value

Return the revised card as JSON:
{{
    "card_type": "{card.card_type}",
    "front": {{
        "question": "Revised, improved question"
    }},
    "back": {{
        "answer": "Revised, improved answer",
        "explanation": "Revised, improved explanation",
        "example": "Revised, improved example"
    }},
    "metadata": {{
        // Enhanced metadata with improvements
    }},
    "revision_notes": "Summary of changes made based on feedback"
}}"""
    
    def _parse_revised_card(self, response: str, original_card: Card) -> Card:
        """Parse the revised card response"""
        try:
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response
            
            # Create revised card
            revised_card = Card(
                card_type=data.get("card_type", original_card.card_type),
                front=CardFront(
                    question=data["front"]["question"]
                ),
                back=CardBack(
                    answer=data["back"]["answer"],
                    explanation=data["back"].get("explanation", ""),
                    example=data["back"].get("example", "")
                ),
                metadata=data.get("metadata", original_card.metadata)
            )
            
            # Add revision tracking to metadata
            if revised_card.metadata is None:
                revised_card.metadata = {}
            
            revised_card.metadata["revision_notes"] = data.get("revision_notes", "Revised based on judge feedback")
            revised_card.metadata["last_revised"] = datetime.now().isoformat()
            
            return revised_card
            
        except Exception as e:
            logger.error(f"Failed to parse revised card: {e}")
            return original_card


class EnhancementAgent(BaseAgentWrapper):
    """Agent for enhancing cards with additional content and metadata"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        config_manager = get_config_manager()
        base_config = config_manager.get_agent_config("enhancement_agent")
        
        if not base_config:
            base_config = AgentConfig(
                name="enhancement_agent",
                instructions="""You are a content enhancement specialist.
Add missing elements and enrich flashcard content without overwhelming learners.
Enhance metadata, examples, and educational value.""",
                model="gpt-4o",
                temperature=0.7
            )
        
        super().__init__(base_config, openai_client)
    
    async def enhance_card(
        self,
        card: Card,
        enhancement_targets: List[str] = None
    ) -> Card:
        """Enhance a card with additional content and metadata"""
        start_time = datetime.now()
        
        try:
            # Default enhancement targets if none specified
            if not enhancement_targets:
                enhancement_targets = [
                    "explanation",
                    "example",
                    "metadata",
                    "learning_outcomes",
                    "prerequisites",
                    "related_concepts"
                ]
            
            user_input = self._build_enhancement_prompt(card, enhancement_targets)
            
            # Execute enhancement
            response = await self.execute(user_input)
            
            # Parse enhanced card
            enhanced_card = self._parse_enhanced_card(response, card)
            
            # Record successful execution
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                metadata={
                    "cards_enhanced": 1,
                    "enhancement_targets": enhancement_targets,
                    "enhancements_applied": len(enhancement_targets)
                }
            )
            
            logger.info(f"EnhancementAgent successfully enhanced card: {card.front.question[:50]}...")
            return enhanced_card
            
        except Exception as e:
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"EnhancementAgent failed to enhance card: {e}")
            return card  # Return original card on failure
    
    def _build_enhancement_prompt(
        self,
        card: Card,
        enhancement_targets: List[str]
    ) -> str:
        """Build the enhancement prompt"""
        targets_str = ", ".join(enhancement_targets)
        
        return f"""Enhance this flashcard by adding missing elements and enriching the content:

Current Card:
Question: {card.front.question}
Answer: {card.back.answer}
Explanation: {card.back.explanation}
Example: {card.back.example}
Type: {card.card_type}
Current Metadata: {json.dumps(card.metadata, indent=2)}

Enhancement Targets: {targets_str}

Enhancement Instructions:
1. Add comprehensive explanations with reasoning
2. Provide relevant, practical examples
3. Enrich metadata with appropriate tags and categorization
4. Add learning outcomes and prerequisites if missing
5. Include connections to related concepts
6. Ensure enhancements add value without overwhelming the learner

Return the enhanced card as JSON:
{{
    "card_type": "{card.card_type}",
    "front": {{
        "question": "Enhanced question (if improvements needed)"
    }},
    "back": {{
        "answer": "Enhanced answer",
        "explanation": "Comprehensive explanation with reasoning and context",
        "example": "Relevant, practical example with details"
    }},
    "metadata": {{
        "topic": "specific topic",
        "subject": "subject area", 
        "difficulty": "beginner|intermediate|advanced",
        "tags": ["comprehensive", "tag", "list"],
        "learning_outcomes": ["specific learning outcome 1", "outcome 2"],
        "prerequisites": ["prerequisite 1", "prerequisite 2"],
        "related_concepts": ["concept 1", "concept 2"],
        "estimated_time": "time in minutes",
        "common_mistakes": ["mistake 1", "mistake 2"],
        "memory_aids": ["mnemonic or memory aid"],
        "real_world_applications": ["application 1", "application 2"]
    }},
    "enhancement_notes": "Summary of enhancements made"
}}"""
    
    def _parse_enhanced_card(self, response: str, original_card: Card) -> Card:
        """Parse the enhanced card response"""
        try:
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response
            
            # Create enhanced card
            enhanced_card = Card(
                card_type=data.get("card_type", original_card.card_type),
                front=CardFront(
                    question=data["front"]["question"]
                ),
                back=CardBack(
                    answer=data["back"]["answer"],
                    explanation=data["back"].get("explanation", original_card.back.explanation),
                    example=data["back"].get("example", original_card.back.example)
                ),
                metadata=data.get("metadata", original_card.metadata)
            )
            
            # Add enhancement tracking to metadata
            if enhanced_card.metadata is None:
                enhanced_card.metadata = {}
            
            enhanced_card.metadata["enhancement_notes"] = data.get("enhancement_notes", "Enhanced with additional content")
            enhanced_card.metadata["last_enhanced"] = datetime.now().isoformat()
            
            return enhanced_card
            
        except Exception as e:
            logger.error(f"Failed to parse enhanced card: {e}")
            return original_card
    
    async def enhance_card_batch(
        self,
        cards: List[Card],
        enhancement_targets: List[str] = None
    ) -> List[Card]:
        """Enhance multiple cards in batch"""
        start_time = datetime.now()
        
        try:
            enhanced_cards = []
            
            # Process cards in parallel for efficiency
            tasks = [
                self.enhance_card(card, enhancement_targets)
                for card in cards
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for card, result in zip(cards, results):
                if isinstance(result, Exception):
                    logger.warning(f"Enhancement failed for card: {result}")
                    enhanced_cards.append(card)  # Keep original
                else:
                    enhanced_cards.append(result)
            
            # Record batch execution
            successful_enhancements = len([r for r in results if not isinstance(r, Exception)])
            
            record_agent_execution(
                agent_name=f"{self.config.name}_batch",
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                metadata={
                    "cards_processed": len(cards),
                    "successful_enhancements": successful_enhancements,
                    "enhancement_rate": successful_enhancements / len(cards) if cards else 0
                }
            )
            
            logger.info(f"EnhancementAgent batch complete: {successful_enhancements}/{len(cards)} cards enhanced")
            return enhanced_cards
            
        except Exception as e:
            record_agent_execution(
                agent_name=f"{self.config.name}_batch",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"EnhancementAgent batch failed: {e}")
            return cards  # Return original cards on failure