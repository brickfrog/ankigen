"""Auto-configuration service for intelligent settings population"""

from typing import Dict, Any
from openai import AsyncOpenAI

from ankigen_core.logging import logger
from ankigen_core.context7 import Context7Client
from ankigen_core.agents.schemas import AutoConfigSchema


class AutoConfigService:
    """Service for analyzing subjects and auto-configuring flashcard generation settings"""

    def __init__(self):
        self.context7_client = Context7Client()

    async def analyze_subject(
        self, subject: str, openai_client: AsyncOpenAI
    ) -> AutoConfigSchema:
        """Analyze a subject string and return optimal configuration settings"""

        system_prompt = """You are an expert educational content analyzer specializing in spaced repetition learning. Analyze the given subject and determine optimal flashcard generation settings that focus on ESSENTIAL, HIGH-VALUE concepts.

CRITICAL PRINCIPLE: Quality over quantity. Focus on fundamental concepts that unlock understanding, not trivial facts.

Consider:
1. Extract any library/framework names for Context7 search (e.g., "pandas", "react", "tensorflow")
2. IMPORTANT: Extract the specific documentation focus from the subject
   - "Basic Pandas Dataframe" → documentation_focus: "dataframe basics, creation, indexing"
   - "React hooks tutorial" → documentation_focus: "hooks, useState, useEffect"
   - "Docker networking" → documentation_focus: "networking, network drivers, container communication"
3. Identify the scope: narrow (specific feature), medium (several related topics), broad (comprehensive overview)
4. Determine content type: concepts (theory/understanding), syntax (code/commands), api (library usage), practical (hands-on skills)
5. Suggest optimal number of topics and cards - aim for comprehensive learning (30-60 total cards minimum)
6. Recommend cloze cards for syntax/code, basic cards for concepts
7. Choose model based on complexity: gpt-4.1 for complex/advanced, gpt-4.1-nano for basic/simple

IMPORTANT - Focus on HIGH-VALUE topics:
- GOOD topics: Core concepts, fundamental principles, mental models, design patterns, key abstractions
- AVOID topics: Trivial commands (like "docker ps"), basic syntax that's easily googled, minor API details
- Example: For Docker, focus on "container lifecycle", "image layers", "networking models" NOT "list of docker commands"

Guidelines for settings (MINIMUM 30 cards total):
- Narrow/specific scope: 4-5 essential topics with 8-10 cards each (32-50 cards)
- Medium scope: 5-7 core topics with 7-9 cards each (35-63 cards)
- Broad scope: 6-8 fundamental topics with 6-8 cards each (36-64 cards)
- "Basic"/"Introduction" keywords: Start with fundamentals, 40-50 cards total
- "Advanced"/"Complex" keywords: Deep dive into critical concepts, 45-60 cards

Learning preference suggestions:
- For basics: "Focus on fundamental concepts and mental models that form the foundation"
- For practical: "Emphasize core patterns and principles with real-world applications"
- For theory: "Build deep conceptual understanding with progressive complexity"

Documentation focus examples (be specific and comprehensive):
- "Basic Pandas Dataframe" → "dataframe creation, indexing, selection, basic operations, data types"
- "React hooks" → "useState, useEffect, custom hooks, hook rules, common patterns"
- "Docker basics" → "containers, images, Dockerfile, volumes, basic networking"
- "Advanced TypeScript" → "generics, conditional types, mapped types, utility types, type inference"

Return a JSON object matching the AutoConfigSchema."""

        user_prompt = f"""Analyze this subject for flashcard generation: "{subject}"

Extract:
1. The library name if mentioned
2. The specific documentation focus (what aspects of the library to focus on)
3. Optimal settings for effective learning

Provide a brief rationale for your choices."""

        try:
            response = await openai_client.beta.chat.completions.parse(
                model="gpt-4.1-nano",  # Use nano for this analysis task
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=AutoConfigSchema,
                temperature=0.3,  # Lower temperature for more consistent analysis
            )

            if not response.choices or not response.choices[0].message.parsed:
                raise ValueError("Failed to get valid response from OpenAI")

            config = response.choices[0].message.parsed
            logger.info(
                f"Subject analysis complete: library='{config.library_search_term}', "
                f"topics={config.topic_number}, cards/topic={config.cards_per_topic}"
            )
            return config

        except Exception as e:
            logger.error(f"Failed to analyze subject: {e}")
            # Return sensible defaults on error (still aim for good card count)
            return AutoConfigSchema(
                library_search_term="",
                documentation_focus=None,
                topic_number=6,
                cards_per_topic=8,
                learning_preferences="Focus on fundamental concepts and core principles with practical examples",
                generate_cloze=False,
                model_choice="gpt-4.1-nano",
                subject_type="concepts",
                scope="medium",
                rationale="Using default settings due to analysis error",
            )

    async def auto_configure(
        self, subject: str, openai_client: AsyncOpenAI
    ) -> Dict[str, Any]:
        """
        Complete auto-configuration pipeline:
        1. Analyze subject with AI
        2. Search Context7 for library if detected
        3. Return complete configuration for UI
        """

        if not subject or not subject.strip():
            logger.warning("Empty subject provided to auto_configure")
            return {}

        logger.info(f"Starting auto-configuration for subject: '{subject}'")

        # Step 1: Analyze the subject
        config = await self.analyze_subject(subject, openai_client)

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
