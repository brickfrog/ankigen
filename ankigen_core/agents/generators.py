# Specialized generator agents for card generation

import json
from typing import List, Dict, Any, Optional, Tuple

from openai import AsyncOpenAI

from ankigen_core.logging import logger
from ankigen_core.models import Card, CardFront, CardBack
from .base import BaseAgentWrapper, AgentConfig
from .config import get_config_manager
from .schemas import CardsGenerationSchema


def card_dict_to_card(
    card_data: Dict[str, Any],
    default_topic: str,
    default_subject: str,
) -> Card:
    """Convert a dictionary representation of a card into a Card object."""

    if not isinstance(card_data, dict):
        raise ValueError("Card payload must be a dictionary")

    front_data = card_data.get("front")
    back_data = card_data.get("back")

    if not isinstance(front_data, dict) or "question" not in front_data:
        raise ValueError("Card front must include a question field")
    if not isinstance(back_data, dict) or "answer" not in back_data:
        raise ValueError("Card back must include an answer field")

    metadata = card_data.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}

    subject = metadata.get("subject") or default_subject or "general"
    topic = metadata.get("topic") or default_topic or "General Concepts"

    card = Card(
        card_type=str(card_data.get("card_type", "basic")),
        front=CardFront(question=str(front_data.get("question", ""))),
        back=CardBack(
            answer=str(back_data.get("answer", "")),
            explanation=str(back_data.get("explanation", "")),
            example=str(back_data.get("example", "")),
        ),
        metadata=metadata,
    )

    if card.metadata is not None:
        card.metadata.setdefault("subject", subject)
        card.metadata.setdefault("topic", topic)

    return card


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
        """Generate flashcards for a given topic with automatic batching for large requests"""
        try:
            # Use batching for large numbers of cards to avoid LLM limitations
            batch_size = 10  # Generate max 10 cards per batch
            all_cards = []
            total_usage = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}

            cards_remaining = num_cards
            batch_num = 1

            logger.info(
                f"Generating {num_cards} cards for topic '{topic}' using {((num_cards - 1) // batch_size) + 1} batches"
            )

            # Track card topics from previous batches to avoid duplication
            previous_card_topics = []

            while cards_remaining > 0:
                cards_in_this_batch = min(batch_size, cards_remaining)

                logger.info(
                    f"Generating batch {batch_num}: {cards_in_this_batch} cards"
                )

                # Reset agent for each batch to avoid conversation history accumulation
                self.agent = None
                await self.initialize()

                user_input = (
                    f"Generate {cards_in_this_batch} flashcards for the topic: {topic}"
                )
                if context:
                    user_input += f"\n\nAdditional context: {context}"

                # Add previous topics to avoid repetition instead of full conversation history
                if previous_card_topics:
                    topics_summary = ", ".join(
                        previous_card_topics[-20:]
                    )  # Last 20 topics to keep it manageable
                    user_input += f"\n\nAvoid creating cards about these already covered topics: {topics_summary}"

                if batch_num > 1:
                    user_input += f"\n\nThis is batch {batch_num} of cards. Ensure these cards cover different aspects of the topic."

                response, usage = await self.execute(user_input, context)

                # Accumulate usage information
                if usage:
                    for key in total_usage:
                        total_usage[key] += usage.get(key, 0)

                batch_cards = self._parse_cards_response(response, topic)
                all_cards.extend(batch_cards)

                # Extract topics from generated cards to avoid duplication in next batch
                for card in batch_cards:
                    if hasattr(card, "front") and card.front and card.front.question:
                        # Extract key terms from the question for deduplication
                        question_words = card.front.question.lower().split()
                        key_terms = [word for word in question_words if len(word) > 3][
                            :3
                        ]  # First 3 meaningful words
                        if key_terms:
                            previous_card_topics.append(" ".join(key_terms))

                cards_remaining -= len(batch_cards)
                batch_num += 1

                logger.info(
                    f"Batch {batch_num-1} generated {len(batch_cards)} cards. {cards_remaining} cards remaining."
                )

                # Safety check to prevent infinite loops
                if len(batch_cards) == 0:
                    logger.warning(
                        f"No cards generated in batch {batch_num-1}, stopping generation"
                    )
                    break

            # Log final usage information
            if total_usage.get("total_tokens", 0) > 0:
                logger.info(
                    f"💰 Total Token Usage: {total_usage['total_tokens']} tokens (Input: {total_usage['input_tokens']}, Output: {total_usage['output_tokens']})"
                )

            logger.info(
                f"✅ Generated {len(all_cards)} cards total across {batch_num-1} batches for topic '{topic}'"
            )
            return all_cards

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
                    if hasattr(card_data, "dict"):
                        payload = card_data.dict()
                    elif isinstance(card_data, dict):
                        payload = card_data
                    else:
                        logger.warning(
                            f"Skipping card {i}: unsupported payload type {type(card_data)}"
                        )
                        continue

                    card = card_dict_to_card(payload, topic, self.subject)
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


class QualityReviewAgent(BaseAgentWrapper):
    """Single-pass quality review agent for lightweight validation and fixes."""

    def __init__(self, openai_client: AsyncOpenAI, model: str):
        config = AgentConfig(
            name="quality_reviewer",
            instructions=(
                "You are a meticulous flashcard reviewer. Review each card for factual accuracy, clarity,"
                " atomic scope, and answer quality. When needed, revise the card while keeping it concise and"
                " faithful to the original intent. Always respond with a JSON object containing:"
                ' {"approved": bool, "reason": string, "revised_card": object or null}.'
                " The revised card must follow the input schema with fields card_type, front.question,"
                " back.answer/explanation/example, and metadata."
            ),
            model=model,
            temperature=0.2,
            timeout=45.0,
            retry_attempts=2,
            enable_tracing=False,
        )
        super().__init__(config, openai_client)

    async def review_card(self, card: Card) -> Tuple[Optional[Card], bool, str]:
        """Review a card and optionally return a revised version."""

        card_payload = {
            "card_type": card.card_type,
            "front": {"question": card.front.question if card.front else ""},
            "back": {
                "answer": card.back.answer if card.back else "",
                "explanation": card.back.explanation if card.back else "",
                "example": card.back.example if card.back else "",
            },
            "metadata": card.metadata or {},
        }

        user_input = (
            "Review the following flashcard. Approve it if it is accurate, clear, and atomic."
            " If improvements are needed, provide a revised_card with the corrections applied.\n\n"
            "Flashcard JSON:\n"
            f"{json.dumps(card_payload, ensure_ascii=False)}\n\n"
            "Respond with JSON matching this schema:\n"
            '{\n  "approved": true | false,\n  "reason": "short explanation",\n'
            '  "revised_card": { ... } | null\n}'
        )

        try:
            response, _ = await self.execute(user_input)
        except Exception as e:
            logger.error(f"Quality review failed to execute: {e}")
            return card, True, "Review failed; keeping original card"

        try:
            parsed = json.loads(response) if isinstance(response, str) else response
        except Exception as e:
            logger.warning(f"Failed to parse review response as JSON: {e}")
            return card, True, "Reviewer returned invalid JSON; keeping original"

        approved = bool(parsed.get("approved", True))
        reason = str(parsed.get("reason", ""))
        revised_payload = parsed.get("revised_card")

        revised_card: Optional[Card] = None
        if isinstance(revised_payload, dict):
            try:
                metadata = revised_payload.get("metadata", {}) or {}
                revised_subject = metadata.get("subject") or (card.metadata or {}).get(
                    "subject",
                    "general",
                )
                revised_topic = metadata.get("topic") or (card.metadata or {}).get(
                    "topic",
                    "General Concepts",
                )
                revised_card = card_dict_to_card(
                    revised_payload, revised_topic, revised_subject
                )
            except Exception as e:
                logger.warning(f"Failed to build revised card from review payload: {e}")
                revised_card = None

        return revised_card or card, approved, reason or ""
