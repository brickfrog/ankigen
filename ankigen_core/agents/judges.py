# Specialized judge agents for card quality assessment

import json
import asyncio
from typing import List, Dict, Any, Tuple
from datetime import datetime

from openai import AsyncOpenAI

from ankigen_core.logging import logger
from ankigen_core.models import Card
from .base import BaseAgentWrapper, AgentConfig
from .config import get_config_manager
from .metrics import record_agent_execution


class JudgeDecision:
    """Represents a judge's decision on a card"""

    def __init__(
        self,
        approved: bool,
        score: float,
        feedback: str,
        improvements: List[str] = None,
        judge_name: str = "",
        metadata: Dict[str, Any] = None,
    ):
        self.approved = approved
        self.score = score  # 0.0 to 1.0
        self.feedback = feedback
        self.improvements = improvements or []
        self.judge_name = judge_name
        self.metadata = metadata or {}


class ContentAccuracyJudge(BaseAgentWrapper):
    """Judge for factual accuracy and content correctness"""

    def __init__(self, openai_client: AsyncOpenAI):
        config_manager = get_config_manager()
        base_config = config_manager.get_agent_config("content_accuracy_judge")

        if not base_config:
            base_config = AgentConfig(
                name="content_accuracy_judge",
                instructions="""You are a fact-checking and accuracy specialist.
Verify the correctness and accuracy of flashcard content, checking for factual errors,
misconceptions, and ensuring consistency with authoritative sources.""",
                model="gpt-4o",
                temperature=0.3,
            )

        super().__init__(base_config, openai_client)

    async def judge_card(self, card: Card) -> JudgeDecision:
        """Judge a single card for content accuracy"""
        start_time = datetime.now()

        try:
            user_input = self._build_judgment_prompt(card)
            response = await self.execute(user_input)

            # Parse the response
            decision_data = (
                json.loads(response) if isinstance(response, str) else response
            )
            decision = self._parse_decision(decision_data)

            # Record successful execution
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                metadata={
                    "cards_judged": 1,
                    "approved": 1 if decision.approved else 0,
                    "score": decision.score,
                },
            )

            return decision

        except Exception as e:
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
            )

            logger.error(f"ContentAccuracyJudge failed: {e}")
            # Return default approval to avoid blocking workflow
            return JudgeDecision(
                approved=True,
                score=0.5,
                feedback=f"Judgment failed: {str(e)}",
                judge_name=self.config.name,
            )

    def _build_judgment_prompt(self, card: Card) -> str:
        """Build the judgment prompt for content accuracy"""
        return f"""Evaluate this flashcard for factual accuracy and content correctness:

Card:
Question: {card.front.question}
Answer: {card.back.answer}
Explanation: {card.back.explanation}
Example: {card.back.example}
Subject: {card.metadata.get('subject', 'Unknown')}
Topic: {card.metadata.get('topic', 'Unknown')}

Evaluate for:
1. Factual Accuracy: Are all statements factually correct?
2. Source Consistency: Does content align with authoritative sources?
3. Terminology: Is domain-specific terminology used correctly?
4. Misconceptions: Does the card avoid or address common misconceptions?
5. Currency: Is the information up-to-date?

Return your assessment as JSON:
{{
    "approved": true/false,
    "accuracy_score": 0.0-1.0,
    "factual_errors": ["error1", "error2"],
    "terminology_issues": ["issue1", "issue2"],
    "misconceptions": ["misconception1"],
    "suggestions": ["improvement1", "improvement2"],
    "confidence": 0.0-1.0,
    "detailed_feedback": "Comprehensive assessment of content accuracy"
}}"""

    def _parse_decision(self, decision_data: Dict[str, Any]) -> JudgeDecision:
        """Parse the judge response into a JudgeDecision"""
        return JudgeDecision(
            approved=decision_data.get("approved", True),
            score=decision_data.get("accuracy_score", 0.5),
            feedback=decision_data.get("detailed_feedback", "No feedback provided"),
            improvements=decision_data.get("suggestions", []),
            judge_name=self.config.name,
            metadata={
                "factual_errors": decision_data.get("factual_errors", []),
                "terminology_issues": decision_data.get("terminology_issues", []),
                "misconceptions": decision_data.get("misconceptions", []),
                "confidence": decision_data.get("confidence", 0.5),
            },
        )


class PedagogicalJudge(BaseAgentWrapper):
    """Judge for educational effectiveness and pedagogical principles"""

    def __init__(self, openai_client: AsyncOpenAI):
        config_manager = get_config_manager()
        base_config = config_manager.get_agent_config("pedagogical_judge")

        if not base_config:
            base_config = AgentConfig(
                name="pedagogical_judge",
                instructions="""You are an educational assessment specialist.
Evaluate flashcards for pedagogical effectiveness, learning objectives,
cognitive levels, and educational best practices.""",
                model="gpt-4o",
                temperature=0.4,
            )

        super().__init__(base_config, openai_client)

    async def judge_card(self, card: Card) -> JudgeDecision:
        """Judge a single card for pedagogical effectiveness"""
        start_time = datetime.now()

        try:
            user_input = self._build_judgment_prompt(card)
            response = await self.execute(user_input)

            decision_data = (
                json.loads(response) if isinstance(response, str) else response
            )
            decision = self._parse_decision(decision_data)

            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                metadata={
                    "cards_judged": 1,
                    "approved": 1 if decision.approved else 0,
                    "score": decision.score,
                },
            )

            return decision

        except Exception as e:
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
            )

            logger.error(f"PedagogicalJudge failed: {e}")
            return JudgeDecision(
                approved=True,
                score=0.5,
                feedback=f"Judgment failed: {str(e)}",
                judge_name=self.config.name,
            )

    def _build_judgment_prompt(self, card: Card) -> str:
        """Build the judgment prompt for pedagogical effectiveness"""
        return f"""Evaluate this flashcard for pedagogical effectiveness:

Card:
Question: {card.front.question}
Answer: {card.back.answer}
Explanation: {card.back.explanation}
Example: {card.back.example}
Difficulty: {card.metadata.get('difficulty', 'Unknown')}

Evaluate based on:
1. Learning Objectives: Clear, measurable learning goals?
2. Bloom's Taxonomy: Appropriate cognitive level?
3. Cognitive Load: Manageable information load?
4. Motivation: Engaging and relevant content?
5. Assessment: Valid testing of understanding vs memorization?

Return your assessment as JSON:
{{
    "approved": true/false,
    "pedagogical_score": 0.0-1.0,
    "cognitive_level": "remember|understand|apply|analyze|evaluate|create",
    "cognitive_load": "low|medium|high",
    "learning_objectives": ["objective1", "objective2"],
    "engagement_factors": ["factor1", "factor2"],
    "pedagogical_issues": ["issue1", "issue2"],
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "detailed_feedback": "Comprehensive pedagogical assessment"
}}"""

    def _parse_decision(self, decision_data: Dict[str, Any]) -> JudgeDecision:
        """Parse the judge response into a JudgeDecision"""
        return JudgeDecision(
            approved=decision_data.get("approved", True),
            score=decision_data.get("pedagogical_score", 0.5),
            feedback=decision_data.get("detailed_feedback", "No feedback provided"),
            improvements=decision_data.get("improvement_suggestions", []),
            judge_name=self.config.name,
            metadata={
                "cognitive_level": decision_data.get("cognitive_level", "unknown"),
                "cognitive_load": decision_data.get("cognitive_load", "medium"),
                "learning_objectives": decision_data.get("learning_objectives", []),
                "engagement_factors": decision_data.get("engagement_factors", []),
                "pedagogical_issues": decision_data.get("pedagogical_issues", []),
            },
        )


class ClarityJudge(BaseAgentWrapper):
    """Judge for clarity, readability, and communication effectiveness"""

    def __init__(self, openai_client: AsyncOpenAI):
        config_manager = get_config_manager()
        base_config = config_manager.get_agent_config("clarity_judge")

        if not base_config:
            base_config = AgentConfig(
                name="clarity_judge",
                instructions="""You are a communication and clarity specialist.
Ensure flashcards are clear, unambiguous, well-written, and accessible
to the target audience.""",
                model="gpt-4o-mini",
                temperature=0.3,
            )

        super().__init__(base_config, openai_client)

    async def judge_card(self, card: Card) -> JudgeDecision:
        """Judge a single card for clarity and communication"""
        start_time = datetime.now()

        try:
            user_input = self._build_judgment_prompt(card)
            response = await self.execute(user_input)

            decision_data = (
                json.loads(response) if isinstance(response, str) else response
            )
            decision = self._parse_decision(decision_data)

            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                metadata={
                    "cards_judged": 1,
                    "approved": 1 if decision.approved else 0,
                    "score": decision.score,
                },
            )

            return decision

        except Exception as e:
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
            )

            logger.error(f"ClarityJudge failed: {e}")
            return JudgeDecision(
                approved=True,
                score=0.5,
                feedback=f"Judgment failed: {str(e)}",
                judge_name=self.config.name,
            )

    def _build_judgment_prompt(self, card: Card) -> str:
        """Build the judgment prompt for clarity assessment"""
        return f"""Evaluate this flashcard for clarity and communication effectiveness:

Card:
Question: {card.front.question}
Answer: {card.back.answer}
Explanation: {card.back.explanation}
Example: {card.back.example}

Evaluate for:
1. Question Clarity: Is the question clear and unambiguous?
2. Answer Completeness: Is the answer complete and coherent?
3. Language Level: Appropriate for target audience?
4. Readability: Easy to read and understand?
5. Structure: Well-organized and logical flow?

Return your assessment as JSON:
{{
    "approved": true/false,
    "clarity_score": 0.0-1.0,
    "question_clarity": 0.0-1.0,
    "answer_completeness": 0.0-1.0,
    "readability_level": "elementary|middle|high|college",
    "ambiguities": ["ambiguity1", "ambiguity2"],
    "clarity_issues": ["issue1", "issue2"],
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "detailed_feedback": "Comprehensive clarity assessment"
}}"""

    def _parse_decision(self, decision_data: Dict[str, Any]) -> JudgeDecision:
        """Parse the judge response into a JudgeDecision"""
        return JudgeDecision(
            approved=decision_data.get("approved", True),
            score=decision_data.get("clarity_score", 0.5),
            feedback=decision_data.get("detailed_feedback", "No feedback provided"),
            improvements=decision_data.get("improvement_suggestions", []),
            judge_name=self.config.name,
            metadata={
                "question_clarity": decision_data.get("question_clarity", 0.5),
                "answer_completeness": decision_data.get("answer_completeness", 0.5),
                "readability_level": decision_data.get("readability_level", "unknown"),
                "ambiguities": decision_data.get("ambiguities", []),
                "clarity_issues": decision_data.get("clarity_issues", []),
            },
        )


class TechnicalJudge(BaseAgentWrapper):
    """Judge for technical accuracy in programming and technical content"""

    def __init__(self, openai_client: AsyncOpenAI):
        config_manager = get_config_manager()
        base_config = config_manager.get_agent_config("technical_judge")

        if not base_config:
            base_config = AgentConfig(
                name="technical_judge",
                instructions="""You are a technical accuracy specialist for programming and technical content.
Verify code syntax, best practices, security considerations, and technical correctness.""",
                model="gpt-4o",
                temperature=0.2,
            )

        super().__init__(base_config, openai_client)

    async def judge_card(self, card: Card) -> JudgeDecision:
        """Judge a single card for technical accuracy"""
        start_time = datetime.now()

        try:
            # Only judge technical content
            if not self._is_technical_content(card):
                return JudgeDecision(
                    approved=True,
                    score=1.0,
                    feedback="Non-technical content - no technical review needed",
                    judge_name=self.config.name,
                )

            user_input = self._build_judgment_prompt(card)
            response = await self.execute(user_input)

            decision_data = (
                json.loads(response) if isinstance(response, str) else response
            )
            decision = self._parse_decision(decision_data)

            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                metadata={
                    "cards_judged": 1,
                    "approved": 1 if decision.approved else 0,
                    "score": decision.score,
                    "is_technical": True,
                },
            )

            return decision

        except Exception as e:
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
            )

            logger.error(f"TechnicalJudge failed: {e}")
            return JudgeDecision(
                approved=True,
                score=0.5,
                feedback=f"Technical judgment failed: {str(e)}",
                judge_name=self.config.name,
            )

    def _is_technical_content(self, card: Card) -> bool:
        """Determine if card contains technical content requiring technical review"""
        technical_keywords = [
            "code",
            "programming",
            "algorithm",
            "function",
            "class",
            "method",
            "syntax",
            "API",
            "database",
            "SQL",
            "python",
            "javascript",
            "java",
            "framework",
            "library",
            "development",
            "software",
            "technical",
        ]

        content = (
            f"{card.front.question} {card.back.answer} {card.back.explanation}".lower()
        )
        subject = card.metadata.get("subject", "").lower()

        return any(
            keyword in content or keyword in subject for keyword in technical_keywords
        )

    def _build_judgment_prompt(self, card: Card) -> str:
        """Build the judgment prompt for technical accuracy"""
        return f"""Evaluate this technical flashcard for accuracy and best practices:

Card:
Question: {card.front.question}
Answer: {card.back.answer}
Explanation: {card.back.explanation}
Example: {card.back.example}
Subject: {card.metadata.get('subject', 'Unknown')}

Evaluate for:
1. Code Syntax: Is any code syntactically correct?
2. Best Practices: Does it follow established best practices?
3. Security: Are there security considerations addressed?
4. Performance: Are performance implications mentioned where relevant?
5. Tool Accuracy: Are tool/framework references accurate?

Return your assessment as JSON:
{{
    "approved": true/false,
    "technical_score": 0.0-1.0,
    "syntax_errors": ["error1", "error2"],
    "best_practice_violations": ["violation1", "violation2"],
    "security_issues": ["issue1", "issue2"],
    "performance_concerns": ["concern1", "concern2"],
    "tool_inaccuracies": ["inaccuracy1", "inaccuracy2"],
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "detailed_feedback": "Comprehensive technical assessment"
}}"""

    def _parse_decision(self, decision_data: Dict[str, Any]) -> JudgeDecision:
        """Parse the judge response into a JudgeDecision"""
        return JudgeDecision(
            approved=decision_data.get("approved", True),
            score=decision_data.get("technical_score", 0.5),
            feedback=decision_data.get("detailed_feedback", "No feedback provided"),
            improvements=decision_data.get("improvement_suggestions", []),
            judge_name=self.config.name,
            metadata={
                "syntax_errors": decision_data.get("syntax_errors", []),
                "best_practice_violations": decision_data.get(
                    "best_practice_violations", []
                ),
                "security_issues": decision_data.get("security_issues", []),
                "performance_concerns": decision_data.get("performance_concerns", []),
                "tool_inaccuracies": decision_data.get("tool_inaccuracies", []),
            },
        )


class CompletenessJudge(BaseAgentWrapper):
    """Judge for completeness and quality standards"""

    def __init__(self, openai_client: AsyncOpenAI):
        config_manager = get_config_manager()
        base_config = config_manager.get_agent_config("completeness_judge")

        if not base_config:
            base_config = AgentConfig(
                name="completeness_judge",
                instructions="""You are a completeness and quality assurance specialist.
Ensure flashcards meet all requirements, have complete information,
and maintain consistent quality standards.""",
                model="gpt-4o-mini",
                temperature=0.3,
            )

        super().__init__(base_config, openai_client)

    async def judge_card(self, card: Card) -> JudgeDecision:
        """Judge a single card for completeness"""
        start_time = datetime.now()

        try:
            user_input = self._build_judgment_prompt(card)
            response = await self.execute(user_input)

            decision_data = (
                json.loads(response) if isinstance(response, str) else response
            )
            decision = self._parse_decision(decision_data)

            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                metadata={
                    "cards_judged": 1,
                    "approved": 1 if decision.approved else 0,
                    "score": decision.score,
                },
            )

            return decision

        except Exception as e:
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
            )

            logger.error(f"CompletenessJudge failed: {e}")
            return JudgeDecision(
                approved=True,
                score=0.5,
                feedback=f"Completeness judgment failed: {str(e)}",
                judge_name=self.config.name,
            )

    def _build_judgment_prompt(self, card: Card) -> str:
        """Build the judgment prompt for completeness assessment"""
        return f"""Evaluate this flashcard for completeness and quality standards:

Card:
Question: {card.front.question}
Answer: {card.back.answer}
Explanation: {card.back.explanation}
Example: {card.back.example}
Type: {card.card_type}
Metadata: {json.dumps(card.metadata, indent=2)}

Check for:
1. Required Fields: All necessary fields present and filled?
2. Metadata Completeness: Appropriate tags, categorization, difficulty?
3. Content Completeness: Answer, explanation, example present and sufficient?
4. Quality Standards: Consistent formatting and professional quality?
5. Example Relevance: Examples relevant and helpful?

Return your assessment as JSON:
{{
    "approved": true/false,
    "completeness_score": 0.0-1.0,
    "missing_fields": ["field1", "field2"],
    "incomplete_sections": ["section1", "section2"],
    "metadata_issues": ["issue1", "issue2"],
    "quality_concerns": ["concern1", "concern2"],
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "detailed_feedback": "Comprehensive completeness assessment"
}}"""

    def _parse_decision(self, decision_data: Dict[str, Any]) -> JudgeDecision:
        """Parse the judge response into a JudgeDecision"""
        return JudgeDecision(
            approved=decision_data.get("approved", True),
            score=decision_data.get("completeness_score", 0.5),
            feedback=decision_data.get("detailed_feedback", "No feedback provided"),
            improvements=decision_data.get("improvement_suggestions", []),
            judge_name=self.config.name,
            metadata={
                "missing_fields": decision_data.get("missing_fields", []),
                "incomplete_sections": decision_data.get("incomplete_sections", []),
                "metadata_issues": decision_data.get("metadata_issues", []),
                "quality_concerns": decision_data.get("quality_concerns", []),
            },
        )


class JudgeCoordinator(BaseAgentWrapper):
    """Coordinates multiple judges and synthesizes their decisions"""

    def __init__(self, openai_client: AsyncOpenAI):
        config_manager = get_config_manager()
        base_config = config_manager.get_agent_config("judge_coordinator")

        if not base_config:
            base_config = AgentConfig(
                name="judge_coordinator",
                instructions="""You are the quality assurance coordinator.
Orchestrate the judging process and synthesize feedback from specialist judges.
Balance speed with thoroughness in quality assessment.""",
                model="gpt-4o-mini",
                temperature=0.3,
            )

        super().__init__(base_config, openai_client)

        # Initialize specialist judges
        self.content_accuracy = ContentAccuracyJudge(openai_client)
        self.pedagogical = PedagogicalJudge(openai_client)
        self.clarity = ClarityJudge(openai_client)
        self.technical = TechnicalJudge(openai_client)
        self.completeness = CompletenessJudge(openai_client)

    async def coordinate_judgment(
        self,
        cards: List[Card],
        enable_parallel: bool = True,
        min_consensus: float = 0.6,
    ) -> List[Tuple[Card, List[JudgeDecision], bool]]:
        """Coordinate judgment of multiple cards"""
        start_time = datetime.now()

        try:
            results = []

            if enable_parallel:
                # Process all cards in parallel
                tasks = [self._judge_single_card(card, min_consensus) for card in cards]
                card_results = await asyncio.gather(*tasks, return_exceptions=True)

                for card, result in zip(cards, card_results):
                    if isinstance(result, Exception):
                        logger.error(f"Parallel judgment failed for card: {result}")
                        results.append((card, [], False))
                    else:
                        results.append(result)
            else:
                # Process cards sequentially
                for card in cards:
                    try:
                        result = await self._judge_single_card(card, min_consensus)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Sequential judgment failed for card: {e}")
                        results.append((card, [], False))

            # Calculate summary statistics
            total_cards = len(cards)
            approved_cards = len([result for _, _, approved in results if approved])

            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                metadata={
                    "cards_judged": total_cards,
                    "cards_approved": approved_cards,
                    "approval_rate": approved_cards / total_cards
                    if total_cards > 0
                    else 0,
                    "parallel_processing": enable_parallel,
                },
            )

            logger.info(
                f"Judge coordination complete: {approved_cards}/{total_cards} cards approved"
            )
            return results

        except Exception as e:
            record_agent_execution(
                agent_name=self.config.name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
            )

            logger.error(f"Judge coordination failed: {e}")
            raise

    async def _judge_single_card(
        self, card: Card, min_consensus: float
    ) -> Tuple[Card, List[JudgeDecision], bool]:
        """Judge a single card with all relevant judges"""

        # Determine which judges to use based on card content
        judges = [
            self.content_accuracy,
            self.pedagogical,
            self.clarity,
            self.completeness,
        ]

        # Add technical judge only for technical content
        if self.technical._is_technical_content(card):
            judges.append(self.technical)

        # Execute all judges in parallel
        judge_tasks = [judge.judge_card(card) for judge in judges]
        decisions = await asyncio.gather(*judge_tasks, return_exceptions=True)

        # Filter out failed decisions
        valid_decisions = []
        for decision in decisions:
            if isinstance(decision, JudgeDecision):
                valid_decisions.append(decision)
            else:
                logger.warning(f"Judge decision failed: {decision}")

        # Calculate consensus
        if not valid_decisions:
            return (card, [], False)

        approval_votes = len([d for d in valid_decisions if d.approved])
        consensus_score = approval_votes / len(valid_decisions)

        # Determine final approval based on consensus
        final_approval = consensus_score >= min_consensus

        return (card, valid_decisions, final_approval)
