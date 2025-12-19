"""Auto-configuration service for intelligent settings population"""

from typing import Dict, Any
from openai import AsyncOpenAI

from ankigen_core.logging import logger
from ankigen_core.context7 import Context7Client
from ankigen_core.agents.schemas import AutoConfigSchema
from ankigen_core.llm_interface import structured_agent_call


class AutoConfigService:
    """Service for analyzing subjects and auto-configuring flashcard generation settings"""

    def __init__(self):
        self.context7_client = Context7Client()

    async def analyze_subject(
        self,
        subject: str,
        openai_client: AsyncOpenAI,
        target_topic_count: int | None = None,
    ) -> AutoConfigSchema:
        """Analyze a subject string and return configuration settings.

        Args:
            subject: The subject to analyze
            openai_client: OpenAI client for LLM calls
            target_topic_count: If provided, forces exactly this many topics in decomposition
        """

        # Build topic count instruction if override provided
        topic_count_instruction = ""
        if target_topic_count is not None:
            topic_count_instruction = f"""
IMPORTANT OVERRIDE: The user has requested exactly {target_topic_count} topics.
You MUST set topic_number to {target_topic_count} and provide exactly {target_topic_count} items in topics_list.
Choose the {target_topic_count} most important/foundational subtopics for this subject.
"""

        system_prompt = f"""You are an educational content analyzer specializing in spaced repetition learning. Analyze the given subject and determine flashcard generation settings that focus on ESSENTIAL concepts.
{topic_count_instruction}

CRITICAL PRINCIPLE: Quality over quantity. Focus on fundamental concepts that unlock understanding, not trivial facts.

Consider:
1. Extract any library/framework names for Context7 search (e.g., "pandas", "react", "tensorflow")
2. IMPORTANT: Extract the specific documentation focus from the subject
   - "Basic Pandas Dataframe" → documentation_focus: "dataframe basics, creation, indexing"
   - "React hooks tutorial" → documentation_focus: "hooks, useState, useEffect"
   - "Docker networking" → documentation_focus: "networking, network drivers, container communication"
3. Identify the scope: narrow (specific feature), medium (several related topics), broad (full overview)
4. Determine content type: concepts (theory/understanding), syntax (code/commands), api (library usage), practical (hands-on skills)
5. TOPIC DECOMPOSITION: Break down the subject into distinct subtopics that together provide comprehensive coverage
6. Recommend cloze cards for syntax/code, basic cards for concepts
7. Choose model based on complexity: gpt-5.2-thinking for complex topics, gpt-5.2-instant for basic/simple, gpt-5.2-auto for mixed scope
   - Valid model_choice values: "gpt-5.2-auto", "gpt-5.2-instant", "gpt-5.2-thinking"

TOPIC DECOMPOSITION (topics_list):
You MUST provide a topics_list - a list of distinct subtopics that together cover the subject comprehensively.
- Each topic should be specific and non-overlapping
- Order topics from foundational to advanced (learning progression)
- The number of topics should match topic_number

Examples:
- "React Hooks" → topics_list: ["useState fundamentals", "useEffect and lifecycle", "useRef and useContext", "custom hooks patterns", "performance with useMemo/useCallback", "testing hooks"]
- "Docker basics" → topics_list: ["containers vs VMs", "images and Dockerfile", "container lifecycle", "volumes and persistence", "networking fundamentals", "docker-compose basics"]
- "Machine Learning" → topics_list: ["supervised vs unsupervised", "regression models", "classification models", "model evaluation metrics", "overfitting and regularization", "feature engineering", "cross-validation"]

IMPORTANT - Focus on HIGH-VALUE topics:
- GOOD topics: Core concepts, fundamental principles, mental models, design patterns, key abstractions
- AVOID topics: Trivial commands (like "docker ps"), basic syntax that's easily googled, minor API details

Guidelines for settings (MINIMUM 30 cards total):
- Narrow/specific scope: 4-5 essential topics with 8-10 cards each (32-50 cards)
- Medium scope: 5-7 core topics with 7-9 cards each (35-63 cards)
- Broad scope: 6-8 fundamental topics with 6-8 cards each (36-64 cards)
- "Basic"/"Introduction" keywords: Start with fundamentals, 40-50 cards total
- "Complex" keywords: Deep dive into critical concepts, 45-60 cards

Learning preference suggestions:
- For basics: "Focus on fundamental concepts and mental models that form the foundation"
- For practical: "Emphasize core patterns and principles with real-world applications"
- For theory: "Build deep conceptual understanding with progressive complexity"

Return a JSON object matching the AutoConfigSchema."""

        user_prompt = f"""Analyze this subject for flashcard generation: "{subject}"

Extract:
1. The library name if mentioned
2. The specific documentation focus (what aspects of the library to focus on)
3. Suggested settings for effective learning

Provide a brief rationale for your choices."""

        try:
            config = await structured_agent_call(
                openai_client=openai_client,
                model="gpt-5.2",
                instructions=system_prompt,
                user_input=user_prompt,
                output_type=AutoConfigSchema,
                temperature=0.3,  # Lower temperature for more consistent analysis
            )

            logger.info(
                f"Subject analysis complete: library='{config.library_search_term}', "
                f"topics={config.topic_number}, cards/topic={config.cards_per_topic}"
            )
            return config

        except Exception as e:
            logger.error(f"Failed to analyze subject: {e}")
            # Return sensible defaults on error (still aim for good card count)
            # Use the subject as a single topic as fallback
            return AutoConfigSchema(
                library_search_term="",
                documentation_focus=None,
                topic_number=6,
                topics_list=[
                    f"{subject} - fundamentals",
                    f"{subject} - core concepts",
                    f"{subject} - practical applications",
                    f"{subject} - common patterns",
                    f"{subject} - best practices",
                    f"{subject} - advanced topics",
                ],
                cards_per_topic=8,
                learning_preferences="Focus on fundamental concepts and core principles with practical examples",
                generate_cloze=False,
                model_choice="gpt-5.2-auto",
                subject_type="concepts",
                scope="medium",
                rationale="Using default settings due to analysis error",
            )

    async def auto_configure(
        self,
        subject: str,
        openai_client: AsyncOpenAI,
        target_topic_count: int | None = None,
    ) -> Dict[str, Any]:
        """
        Complete auto-configuration pipeline:
        1. Analyze subject with AI
        2. Search Context7 for library if detected
        3. Return complete configuration for UI

        Args:
            subject: The subject to analyze
            openai_client: OpenAI client for LLM calls
            target_topic_count: If provided, forces exactly this many topics
        """

        if not subject or not subject.strip():
            logger.warning("Empty subject provided to auto_configure")
            return {}

        logger.info(f"Starting auto-configuration for subject: '{subject}'")

        # Step 1: Analyze the subject
        config = await self.analyze_subject(
            subject, openai_client, target_topic_count=target_topic_count
        )

        # Step 2: Search Context7 for library if one was detected
        library_id = None
        if config.library_search_term:
            logger.info(
                f"Searching Context7 for library: '{config.library_search_term}'"
            )
            try:
                library_id = await self.context7_client.resolve_library_id(
                    config.library_search_term
                )
                if library_id:
                    logger.info(f"Resolved library to Context7 ID: {library_id}")
                else:
                    logger.warning(
                        f"Could not find library '{config.library_search_term}' in Context7"
                    )
            except Exception as e:
                logger.error(f"Context7 search failed: {e}")

        # Step 3: Build complete configuration dict for UI
        ui_config = {
            "library_name": config.library_search_term if library_id else "",
            "library_topic": config.documentation_focus or "",
            "topic_number": config.topic_number,
            "topics_list": config.topics_list,
            "cards_per_topic": config.cards_per_topic,
            "preference_prompt": config.learning_preferences,
            "generate_cloze_checkbox": config.generate_cloze,
            "model_choice": config.model_choice,
            # Metadata for display
            "analysis_metadata": {
                "subject_type": config.subject_type,
                "scope": config.scope,
                "rationale": config.rationale,
                "library_found": library_id is not None,
                "context7_id": library_id,
            },
        }

        logger.info(
            f"Auto-configuration complete: library={'found' if library_id else 'not found'}, "
            f"topics={config.topic_number}, model={config.model_choice}"
        )

        return ui_config
