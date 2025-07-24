# Specialized generator agents for card generation

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from openai import AsyncOpenAI

from ankigen_core.logging import logger
from ankigen_core.models import Card, CardFront, CardBack
from .base import BaseAgentWrapper
from .config import get_config_manager
from .schemas import CardsGenerationSchema, CardSchema


class SubjectExpertAgent(BaseAgentWrapper):
    """Subject matter expert agent for domain-specific card generation"""

    def __init__(self, openai_client: AsyncOpenAI, subject: str = "general"):
        config_manager = get_config_manager()
        base_config = config_manager.get_agent_config("subject_expert")

        if not base_config:
            raise ValueError(
                "subject_expert configuration not found - agent system not properly initialized"
            )

        # Enable structured output for card generation
        base_config.output_type = CardsGenerationSchema

        # Customize instructions for the specific subject
        if subject != "general" and base_config.custom_prompts:
            subject_prompt = base_config.custom_prompts.get(subject.lower(), "")
            if subject_prompt:
                base_config.instructions += (
                    f"\n\nSubject-specific guidance: {subject_prompt}"
                )

        super().__init__(base_config, openai_client)
        self.subject = subject

    async def generate_cards(
        self, topic: str, num_cards: int = 5, context: Optional[Dict[str, Any]] = None
    ) -> List[Card]:
        """Generate flashcards for a given topic"""
        try:
            user_input = f"Generate {num_cards} flashcards for the topic: {topic}"
            if context:
                user_input += f"\n\nAdditional context: {context}"

            response, usage = await self.execute(user_input, context)

            # Log usage information
            if usage and usage.get("total_tokens", 0) > 0:
                logger.info(
                    f"💰 Token Usage: {usage['total_tokens']} tokens (Input: {usage['input_tokens']}, Output: {usage['output_tokens']})"
                )

            return self._parse_cards_response(response, topic)

        except Exception as e:
            logger.error(f"Card generation failed: {e}")
            raise

    def _build_generation_prompt(
        self,
        topic: str,
        num_cards: int,
        difficulty: str,
        prerequisites: List[str],
        context: Dict[str, Any],
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

    def _parse_cards_response(self, response: Any, topic: str) -> List[Card]:
        """Parse the agent response into Card objects"""
        try:
            # Handle structured output from CardsGenerationSchema
            if hasattr(response, "cards"):
                # Response is already a CardsGenerationSchema object
                logger.info(f"✅ STRUCTURED OUTPUT RECEIVED: {type(response)}")
                card_data_list = response.cards
            elif isinstance(response, dict) and "cards" in response:
                # Response is a dict with cards
                card_data_list = response["cards"]
            elif isinstance(response, str):
                # Fallback: Clean up the response - remove markdown code blocks if present
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]  # Remove ```json
                if response.startswith("```"):
                    response = response[3:]  # Remove ```
                if response.endswith("```"):
                    response = response[:-3]  # Remove trailing ```
                response = response.strip()

                data = json.loads(response)
                if "cards" not in data:
                    raise ValueError("Response missing 'cards' field")
                card_data_list = data["cards"]
            else:
                raise ValueError(f"Unexpected response format: {type(response)}")

            cards = []
            for i, card_data in enumerate(card_data_list):
                try:
                    # Handle both Pydantic models and dictionaries
                    if hasattr(card_data, "front"):
                        # Pydantic model
                        front_data = card_data.front
                        back_data = card_data.back
                        metadata = card_data.metadata
                        card_type = card_data.card_type
                    else:
                        # Dictionary
                        if "front" not in card_data or "back" not in card_data:
                            logger.warning(f"Skipping card {i}: missing front or back")
                            continue
                        front_data = card_data["front"]
                        back_data = card_data["back"]
                        metadata = card_data.get("metadata", {})
                        card_type = card_data.get("card_type", "basic")

                    # Extract question and answer
                    if hasattr(front_data, "question"):
                        question = front_data.question
                    else:
                        question = front_data.get("question", "")

                    if hasattr(back_data, "answer"):
                        answer = back_data.answer
                        explanation = back_data.explanation
                        example = back_data.example
                    else:
                        answer = back_data.get("answer", "")
                        explanation = back_data.get("explanation", "")
                        example = back_data.get("example", "")

                    if not question or not answer:
                        logger.warning(f"Skipping card {i}: missing question or answer")
                        continue

                    # Create Card object
                    card = Card(
                        card_type=card_type,
                        front=CardFront(question=question),
                        back=CardBack(
                            answer=answer,
                            explanation=explanation,
                            example=example,
                        ),
                        metadata=metadata
                        if isinstance(metadata, dict)
                        else metadata.dict()
                        if hasattr(metadata, "dict")
                        else {},
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

            logger.info(f"✅ PARSED {len(cards)} CARDS FROM STRUCTURED OUTPUT")
            return cards

        except json.JSONDecodeError as e:
            logger.error(f"💥 JSON DECODE ERROR: {e}")
            logger.error("💥 RAW RESPONSE THAT FAILED TO PARSE:")
            logger.error("---FAILED RESPONSE START---")
            logger.error(f"{response}")
            logger.error("---FAILED RESPONSE END---")
            logger.error(f"💥 RESPONSE TYPE: {type(response)}")
            if isinstance(response, str):
                logger.error(f"💥 RESPONSE LENGTH: {len(response)}")
                logger.error(f"💥 FIRST 200 CHARS: {repr(response[:200])}")
                logger.error(f"💥 LAST 200 CHARS: {repr(response[-200:])}")
            raise ValueError(f"Invalid JSON response from agent: {e}")
        except Exception as e:
            logger.error(f"💥 GENERAL PARSING ERROR: {e}")
            logger.error(f"💥 RESPONSE THAT CAUSED ERROR: {response}")
            raise


class PedagogicalAgent(BaseAgentWrapper):
    """Pedagogical specialist for educational effectiveness"""

    def __init__(self, openai_client: AsyncOpenAI):
        config_manager = get_config_manager()
        base_config = config_manager.get_agent_config("pedagogical")

        if not base_config:
            raise ValueError(
                "pedagogical configuration not found - agent system not properly initialized"
            )

        super().__init__(base_config, openai_client)

    async def review_cards(self, cards: List[Card]) -> List[Dict[str, Any]]:
        """Review cards for pedagogical effectiveness"""
        datetime.now()

        try:
            reviews = []

            for i, card in enumerate(cards):
                user_input = self._build_review_prompt(card, i)
                response, usage = await self.execute(user_input)

                try:
                    review_data = (
                        json.loads(response) if isinstance(response, str) else response
                    )
                    reviews.append(review_data)
                except Exception as e:
                    logger.warning(f"Failed to parse review for card {i}: {e}")
                    reviews.append(
                        {
                            "approved": True,
                            "feedback": f"Review parsing failed: {e}",
                            "improvements": [],
                        }
                    )

            # Record successful execution

            return reviews

        except Exception as e:
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
            required_fields = [
                "pedagogical_quality",
                "clarity",
                "learning_effectiveness",
            ]
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
            raise ValueError(
                "content_structuring configuration not found - agent system not properly initialized"
            )

        # Enable structured output for card structuring
        base_config.output_type = CardSchema

        super().__init__(base_config, openai_client)

    async def structure_cards(self, cards: List[Card]) -> List[Card]:
        """Structure and format cards for consistency"""
        datetime.now()

        try:
            structured_cards = []

            for i, card in enumerate(cards):
                user_input = self._build_structuring_prompt(card, i)
                response, usage = await self.execute(user_input)

                try:
                    # With structured output, response should already be a CardSchema object
                    if hasattr(response, "card_type"):
                        structured_card = self._parse_structured_card(
                            response.__dict__, card
                        )
                    else:
                        # Fallback for backwards compatibility
                        structured_data = (
                            json.loads(response)
                            if isinstance(response, str)
                            else response
                        )
                        structured_card = self._parse_structured_card(
                            structured_data, card
                        )
                    structured_cards.append(structured_card)
                except Exception as e:
                    logger.warning(f"Failed to structure card {i}: {e}")
                    structured_cards.append(card)  # Keep original on failure

            return structured_cards

        except Exception as e:
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

Return the improved card with the structured output format."""

    def _parse_structured_card(
        self, structured_data: Dict[str, Any], original_card: Card
    ) -> Card:
        """Parse structured card data into Card object"""
        try:
            return Card(
                card_type=structured_data.get("card_type", original_card.card_type),
                front=CardFront(question=structured_data["front"]["question"]),
                back=CardBack(
                    answer=structured_data["back"]["answer"],
                    explanation=structured_data["back"].get("explanation", ""),
                    example=structured_data["back"].get("example", ""),
                ),
                metadata=structured_data.get("metadata", original_card.metadata),
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
            raise ValueError(
                "generation_coordinator configuration not found - agent system not properly initialized"
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
        context: Dict[str, Any] = None,
    ) -> List[Card]:
        """Coordinate the full card generation pipeline"""
        datetime.now()

        try:
            # Initialize subject expert for the specific subject
            if not self.subject_expert or self.subject_expert.subject != subject:
                self.subject_expert = SubjectExpertAgent(self.openai_client, subject)

            logger.info(f"Starting coordinated generation: {topic} ({subject})")

            # Step 1: Generate initial cards
            cards = await self.subject_expert.generate_cards(
                topic=topic, num_cards=num_cards, context=context
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
                        logger.info(
                            f"Card flagged for revision: {card.front.question[:50]}..."
                        )

                cards = approved_cards

            # Step 3: Content structuring (optional)
            if enable_structuring and cards:
                logger.info("Performing content structuring...")
                cards = await self.content_structuring.structure_cards(cards)

            # Record successful coordination

            logger.info(f"Generation coordination complete: {len(cards)} cards")
            return cards

        except Exception as e:
            logger.error(f"Generation coordination failed: {e}")
            raise

    async def generate_structured_cards(
        self,
        topic: str,
        num_cards: int = 5,
        difficulty: str = "intermediate",
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Card]:
        """Generate structured flashcards with enhanced metadata"""
        try:
            user_input = f"""Generate {num_cards} structured flashcards for: {topic}

Difficulty: {difficulty}
Requirements:
- Include detailed metadata
- Add learning outcomes
- Specify prerequisites
- Include related concepts
- Estimate study time"""

            response, usage = await self.execute(user_input)

            # Log usage information
            if usage and usage.get("total_tokens", 0) > 0:
                logger.info(
                    f"💰 Token Usage: {usage['total_tokens']} tokens (Input: {usage['input_tokens']}, Output: {usage['output_tokens']})"
                )

            # Parse the structured response directly since it should be a CardsGenerationSchema
            if hasattr(response, "cards") and response.cards:
                return response.cards
            else:
                logger.warning("No cards found in structured response")
                return []

        except Exception as e:
            logger.error(f"Structured card generation failed: {e}")
            raise

    async def generate_adaptive_cards(
        self,
        topic: str,
        learning_style: str = "visual",
        num_cards: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Card]:
        """Generate cards adapted to specific learning styles"""
        try:
            user_input = f"""Generate {num_cards} flashcards for: {topic}

Learning Style: {learning_style}
Adapt the content format and presentation to match this learning style."""

            response, usage = await self.execute(user_input)

            # Log usage information
            if usage and usage.get("total_tokens", 0) > 0:
                logger.info(
                    f"💰 Token Usage: {usage['total_tokens']} tokens (Input: {usage['input_tokens']}, Output: {usage['output_tokens']})"
                )

            # Parse the adaptive response directly since it should be a CardsGenerationSchema
            if hasattr(response, "cards") and response.cards:
                return response.cards
            else:
                logger.warning("No cards found in adaptive response")
                return []

        except Exception as e:
            logger.error(f"Adaptive card generation failed: {e}")
            raise
