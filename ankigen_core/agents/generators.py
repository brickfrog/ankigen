# Specialized generator agents for card generation

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


class SubjectExpertAgent(BaseAgentWrapper):
    """Subject matter expert agent for domain-specific card generation"""
    
    def __init__(self, openai_client: AsyncOpenAI, subject: str = "general"):
        config_manager = get_config_manager()
        base_config = config_manager.get_agent_config("subject_expert")
        
        if not base_config:
            # Fallback config if not found
            base_config = AgentConfig(
                name="subject_expert",
                instructions=f"""You are a world-class expert in {subject} with deep pedagogical knowledge.
Generate high-quality flashcards that demonstrate mastery of {subject} concepts.
Focus on technical accuracy, appropriate depth, and real-world applications.""",
                model="gpt-4o",
                temperature=0.7
            )
        
        # Customize instructions for the specific subject
        if subject != "general" and base_config.custom_prompts:
            subject_prompt = base_config.custom_prompts.get(subject.lower(), "")
            if subject_prompt:
                base_config.instructions += f"\n\nSubject-specific guidance: {subject_prompt}"
        
        super().__init__(base_config, openai_client)
        self.subject = subject
    
    async def generate_cards(
        self,
        topic: str,
        num_cards: int = 5,
        difficulty: str = "intermediate",
        prerequisites: List[str] = None,
        context: Dict[str, Any] = None
    ) -> List[Card]:
        """Generate subject-specific flashcards"""
        start_time = datetime.now()
        
        try:
            user_input = self._build_generation_prompt(
                topic=topic,
                num_cards=num_cards,
                difficulty=difficulty,
                prerequisites=prerequisites or [],
                context=context or {}
            )
            
            # Execute the agent
            response = await self.execute(user_input, context)
            
            # Parse the response into Card objects
            cards = self._parse_cards_response(response, topic)
            
            # Record successful execution
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                metadata={
                    "subject": self.subject,
                    "topic": topic,
                    "cards_generated": len(cards),
                    "difficulty": difficulty
                }
            )
            
            logger.info(f"SubjectExpertAgent generated {len(cards)} cards for {topic}")
            return cards
            
        except Exception as e:
            # Record failed execution
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metadata={"subject": self.subject, "topic": topic}
            )
            
            logger.error(f"SubjectExpertAgent failed to generate cards: {e}")
            raise
    
    def _build_generation_prompt(
        self,
        topic: str,
        num_cards: int,
        difficulty: str,
        prerequisites: List[str],
        context: Dict[str, Any]
    ) -> str:
        """Build the generation prompt"""
        prerequisites_str = ", ".join(prerequisites) if prerequisites else "None"
        
        prompt = f"""Generate {num_cards} high-quality flashcards for the topic: {topic}

Subject: {self.subject}
Difficulty Level: {difficulty}
Prerequisites: {prerequisites_str}

Requirements:
- Focus on {self.subject} concepts and terminology
- Ensure technical accuracy and depth appropriate for {difficulty} level
- Include practical applications and real-world examples
- Test understanding, not just memorization
- Use clear, unambiguous questions

Return your response as a JSON object with this structure:
{{
    "cards": [
        {{
            "card_type": "basic",
            "front": {{
                "question": "Clear, specific question"
            }},
            "back": {{
                "answer": "Concise, accurate answer",
                "explanation": "Detailed explanation with reasoning",
                "example": "Practical example or application"
            }},
            "metadata": {{
                "difficulty": "{difficulty}",
                "prerequisites": {json.dumps(prerequisites)},
                "topic": "{topic}",
                "subject": "{self.subject}",
                "learning_outcomes": ["outcome1", "outcome2"],
                "common_misconceptions": ["misconception1"]
            }}
        }}
    ]
}}"""
        
        if context.get("source_text"):
            prompt += f"\n\nBase the cards on this source material:\n{context['source_text'][:2000]}..."
        
        return prompt
    
    def _parse_cards_response(self, response: str, topic: str) -> List[Card]:
        """Parse the agent response into Card objects"""
        try:
            # Try to parse as JSON
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response
            
            if "cards" not in data:
                raise ValueError("Response missing 'cards' field")
            
            cards = []
            for i, card_data in enumerate(data["cards"]):
                try:
                    # Validate required fields
                    if "front" not in card_data or "back" not in card_data:
                        logger.warning(f"Skipping card {i}: missing front or back")
                        continue
                    
                    front_data = card_data["front"]
                    back_data = card_data["back"]
                    
                    if "question" not in front_data:
                        logger.warning(f"Skipping card {i}: missing question")
                        continue
                    
                    if "answer" not in back_data:
                        logger.warning(f"Skipping card {i}: missing answer")
                        continue
                    
                    # Create Card object
                    card = Card(
                        card_type=card_data.get("card_type", "basic"),
                        front=CardFront(question=front_data["question"]),
                        back=CardBack(
                            answer=back_data["answer"],
                            explanation=back_data.get("explanation", ""),
                            example=back_data.get("example", "")
                        ),
                        metadata=card_data.get("metadata", {})
                    )
                    
                    # Ensure metadata includes subject and topic
                    if card.metadata is not None:
                        if "subject" not in card.metadata:
                            card.metadata["subject"] = self.subject
                        if "topic" not in card.metadata:
                            card.metadata["topic"] = topic
                    
                    cards.append(card)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse card {i}: {e}")
                    continue
            
            return cards
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse cards response as JSON: {e}")
            raise ValueError(f"Invalid JSON response from agent: {e}")
        except Exception as e:
            logger.error(f"Failed to parse cards response: {e}")
            raise


class PedagogicalAgent(BaseAgentWrapper):
    """Pedagogical specialist for educational effectiveness"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        config_manager = get_config_manager()
        base_config = config_manager.get_agent_config("pedagogical")
        
        if not base_config:
            base_config = AgentConfig(
                name="pedagogical",
                instructions="""You are an educational specialist focused on learning theory and instructional design.
Ensure all flashcards follow educational best practices using Bloom's Taxonomy, Spaced Repetition,
and Cognitive Load Theory. Review for clear learning objectives and appropriate difficulty progression.""",
                model="gpt-4o",
                temperature=0.6
            )
        
        super().__init__(base_config, openai_client)
    
    async def review_cards(self, cards: List[Card]) -> List[Dict[str, Any]]:
        """Review cards for pedagogical effectiveness"""
        start_time = datetime.now()
        
        try:
            reviews = []
            
            for i, card in enumerate(cards):
                user_input = self._build_review_prompt(card, i)
                response = await self.execute(user_input)
                
                try:
                    review_data = json.loads(response) if isinstance(response, str) else response
                    reviews.append(review_data)
                except Exception as e:
                    logger.warning(f"Failed to parse review for card {i}: {e}")
                    reviews.append({
                        "approved": True,
                        "feedback": f"Review parsing failed: {e}",
                        "improvements": []
                    })
            
            # Record successful execution
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                metadata={
                    "cards_reviewed": len(cards),
                    "approvals": len([r for r in reviews if r.get("approved", False)])
                }
            )
            
            return reviews
            
        except Exception as e:
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"PedagogicalAgent review failed: {e}")
            raise
    
    def _parse_review_response(self, response) -> Dict[str, Any]:
        """Parse the review response into a dictionary"""
        try:
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response
            
            # Validate required fields
            required_fields = ['pedagogical_quality', 'clarity', 'learning_effectiveness']
            if not all(field in data for field in required_fields):
                raise ValueError("Missing required review fields")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse review response as JSON: {e}")
            raise ValueError(f"Invalid review response: {e}")
        except Exception as e:
            logger.error(f"Failed to parse review response: {e}")
            raise ValueError(f"Invalid review response: {e}")
    
    def _build_review_prompt(self, card: Card, index: int) -> str:
        """Build the review prompt for a single card"""
        return f"""Review this flashcard for pedagogical effectiveness:

Card {index + 1}:
Question: {card.front.question}
Answer: {card.back.answer}
Explanation: {card.back.explanation}
Example: {card.back.example}
Metadata: {json.dumps(card.metadata, indent=2)}

Evaluate the card based on:
1. Learning Objectives: Does it have clear, measurable learning goals?
2. Bloom's Taxonomy: What cognitive level does it target? Is it appropriate?
3. Cognitive Load: Is the information manageable for learners?
4. Difficulty Progression: Is the difficulty appropriate for the target level?
5. Educational Value: Does it promote deep learning vs. memorization?

Return your assessment as JSON:
{{
    "approved": true/false,
    "cognitive_level": "remember|understand|apply|analyze|evaluate|create",
    "difficulty_rating": 1-5,
    "cognitive_load": "low|medium|high",
    "educational_value": 1-5,
    "feedback": "Detailed pedagogical assessment",
    "improvements": ["specific improvement suggestion 1", "suggestion 2"],
    "learning_objectives": ["clear learning objective 1", "objective 2"]
}}"""


class ContentStructuringAgent(BaseAgentWrapper):
    """Content organization and formatting specialist"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        config_manager = get_config_manager()
        base_config = config_manager.get_agent_config("content_structuring")
        
        if not base_config:
            base_config = AgentConfig(
                name="content_structuring",
                instructions="""You are a content organization specialist focused on consistency and structure.
Format and organize flashcard content for optimal learning with consistent formatting,
proper metadata, clear questions, and appropriate categorization.""",
                model="gpt-4o-mini",
                temperature=0.5
            )
        
        super().__init__(base_config, openai_client)
    
    async def structure_cards(self, cards: List[Card]) -> List[Card]:
        """Structure and format cards for consistency"""
        start_time = datetime.now()
        
        try:
            structured_cards = []
            
            for i, card in enumerate(cards):
                user_input = self._build_structuring_prompt(card, i)
                response = await self.execute(user_input)
                
                try:
                    structured_data = json.loads(response) if isinstance(response, str) else response
                    structured_card = self._parse_structured_card(structured_data, card)
                    structured_cards.append(structured_card)
                except Exception as e:
                    logger.warning(f"Failed to structure card {i}: {e}")
                    structured_cards.append(card)  # Keep original on failure
            
            # Record successful execution
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                metadata={
                    "cards_structured": len(cards),
                    "successful_structures": len([c for c in structured_cards if c != cards[i] for i in range(len(cards))])
                }
            )
            
            return structured_cards
            
        except Exception as e:
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"ContentStructuringAgent failed: {e}")
            raise
    
    def _build_structuring_prompt(self, card: Card, index: int) -> str:
        """Build the structuring prompt for a single card"""
        return f"""Structure and format this flashcard for optimal learning:

Original Card {index + 1}:
Question: {card.front.question}
Answer: {card.back.answer}
Explanation: {card.back.explanation}
Example: {card.back.example}
Type: {card.card_type}
Metadata: {json.dumps(card.metadata, indent=2)}

Improve the card's structure and formatting:
1. Ensure clear, concise, unambiguous question
2. Provide complete, well-structured answer
3. Add comprehensive explanation with reasoning
4. Include relevant, practical example
5. Enhance metadata with appropriate tags and categorization
6. Maintain consistent formatting and style

Return the improved card as JSON:
{{
    "card_type": "basic|cloze",
    "front": {{
        "question": "Improved, clear question"
    }},
    "back": {{
        "answer": "Complete, well-structured answer",
        "explanation": "Comprehensive explanation with reasoning",
        "example": "Relevant, practical example"
    }},
    "metadata": {{
        "topic": "specific topic",
        "subject": "subject area",
        "difficulty": "beginner|intermediate|advanced",
        "tags": ["tag1", "tag2", "tag3"],
        "learning_outcomes": ["outcome1", "outcome2"],
        "prerequisites": ["prereq1", "prereq2"],
        "estimated_time": "time in minutes",
        "category": "category name"
    }}
}}"""
    
    def _parse_structured_card(self, structured_data: Dict[str, Any], original_card: Card) -> Card:
        """Parse structured card data into Card object"""
        try:
            return Card(
                card_type=structured_data.get("card_type", original_card.card_type),
                front=CardFront(
                    question=structured_data["front"]["question"]
                ),
                back=CardBack(
                    answer=structured_data["back"]["answer"],
                    explanation=structured_data["back"].get("explanation", ""),
                    example=structured_data["back"].get("example", "")
                ),
                metadata=structured_data.get("metadata", original_card.metadata)
            )
        except Exception as e:
            logger.warning(f"Failed to parse structured card: {e}")
            return original_card


class GenerationCoordinator(BaseAgentWrapper):
    """Coordinates the multi-agent card generation workflow"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        config_manager = get_config_manager()
        base_config = config_manager.get_agent_config("generation_coordinator")
        
        if not base_config:
            base_config = AgentConfig(
                name="generation_coordinator",
                instructions="""You are the generation workflow coordinator.
Orchestrate the card generation process and manage handoffs between specialized agents.
Make decisions based on content type, user preferences, and system load.""",
                model="gpt-4o-mini",
                temperature=0.3
            )
        
        super().__init__(base_config, openai_client)
        
        # Initialize specialized agents
        self.subject_expert = None
        self.pedagogical = PedagogicalAgent(openai_client)
        self.content_structuring = ContentStructuringAgent(openai_client)
    
    async def coordinate_generation(
        self,
        topic: str,
        subject: str = "general",
        num_cards: int = 5,
        difficulty: str = "intermediate",
        enable_review: bool = True,
        enable_structuring: bool = True,
        context: Dict[str, Any] = None
    ) -> List[Card]:
        """Coordinate the full card generation pipeline"""
        start_time = datetime.now()
        
        try:
            # Initialize subject expert for the specific subject
            if not self.subject_expert or self.subject_expert.subject != subject:
                self.subject_expert = SubjectExpertAgent(self.openai_client, subject)
            
            logger.info(f"Starting coordinated generation: {topic} ({subject})")
            
            # Step 1: Generate initial cards
            cards = await self.subject_expert.generate_cards(
                topic=topic,
                num_cards=num_cards,
                difficulty=difficulty,
                context=context
            )
            
            # Step 2: Pedagogical review (optional)
            if enable_review and cards:
                logger.info("Performing pedagogical review...")
                reviews = await self.pedagogical.review_cards(cards)
                
                # Filter or flag cards based on reviews
                approved_cards = []
                for card, review in zip(cards, reviews):
                    if review.get("approved", True):
                        approved_cards.append(card)
                    else:
                        logger.info(f"Card flagged for revision: {card.front.question[:50]}...")
                
                cards = approved_cards
            
            # Step 3: Content structuring (optional)
            if enable_structuring and cards:
                logger.info("Performing content structuring...")
                cards = await self.content_structuring.structure_cards(cards)
            
            # Record successful coordination
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                metadata={
                    "topic": topic,
                    "subject": subject,
                    "cards_generated": len(cards),
                    "review_enabled": enable_review,
                    "structuring_enabled": enable_structuring
                }
            )
            
            logger.info(f"Generation coordination complete: {len(cards)} cards")
            return cards
            
        except Exception as e:
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metadata={"topic": topic, "subject": subject}
            )
            
            logger.error(f"Generation coordination failed: {e}")
            raise