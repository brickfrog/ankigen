# Agent configuration management system

import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict

from ankigen_core.logging import logger
from .base import AgentConfig


@dataclass
class AgentPromptTemplate:
    """Template for agent prompts with variables"""

    system_prompt: str
    user_prompt_template: str
    variables: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.variables is None:
            self.variables = {}

    def render_system_prompt(self, **kwargs) -> str:
        """Render system prompt with provided variables"""
        try:
            variables = self.variables or {}
            return self.system_prompt.format(**{**variables, **kwargs})
        except KeyError as e:
            logger.error(f"Missing variable in system prompt template: {e}")
            return self.system_prompt

    def render_user_prompt(self, **kwargs) -> str:
        """Render user prompt template with provided variables"""
        try:
            variables = self.variables or {}
            return self.user_prompt_template.format(**{**variables, **kwargs})
        except KeyError as e:
            logger.error(f"Missing variable in user prompt template: {e}")
            return self.user_prompt_template


class AgentConfigManager:
    """Manages agent configurations from files and runtime updates"""

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config/agents")
        self.configs: Dict[str, AgentConfig] = {}
        self.prompt_templates: Dict[str, AgentPromptTemplate] = {}
        self._ensure_config_dir()
        self._load_default_configs()

    def _ensure_config_dir(self):
        """Ensure config directory exists"""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Create default config files if they don't exist
        defaults_dir = self.config_dir / "defaults"
        defaults_dir.mkdir(exist_ok=True)

        if not (defaults_dir / "generators.yaml").exists():
            self._create_default_generator_configs()

        if not (defaults_dir / "judges.yaml").exists():
            self._create_default_judge_configs()

        if not (defaults_dir / "enhancers.yaml").exists():
            self._create_default_enhancer_configs()

    def _load_default_configs(self):
        """Load all default configurations"""
        try:
            self._load_configs_from_file("defaults/generators.yaml")
            self._load_configs_from_file("defaults/judges.yaml")
            self._load_configs_from_file("defaults/enhancers.yaml")
            logger.info(f"Loaded {len(self.configs)} agent configurations")
        except Exception as e:
            logger.error(f"Failed to load default agent configurations: {e}")

    def _load_configs_from_file(self, filename: str):
        """Load configurations from a YAML/JSON file"""
        file_path = self.config_dir / filename

        if not file_path.exists():
            logger.warning(f"Agent config file not found: {file_path}")
            return

        try:
            with open(file_path, "r") as f:
                if filename.endswith(".yaml") or filename.endswith(".yml"):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            # Load agent configs
            if "agents" in data:
                for agent_name, agent_data in data["agents"].items():
                    config = AgentConfig(
                        name=agent_name,
                        instructions=agent_data.get("instructions", ""),
                        model=agent_data.get("model", "gpt-4o"),
                        temperature=agent_data.get("temperature", 0.7),
                        max_tokens=agent_data.get("max_tokens"),
                        timeout=agent_data.get("timeout", 30.0),
                        retry_attempts=agent_data.get("retry_attempts", 3),
                        enable_tracing=agent_data.get("enable_tracing", True),
                        custom_prompts=agent_data.get("custom_prompts", {}),
                    )
                    self.configs[agent_name] = config

            # Load prompt templates
            if "prompt_templates" in data:
                for template_name, template_data in data["prompt_templates"].items():
                    template = AgentPromptTemplate(
                        system_prompt=template_data.get("system_prompt", ""),
                        user_prompt_template=template_data.get(
                            "user_prompt_template", ""
                        ),
                        variables=template_data.get("variables", {}),
                    )
                    self.prompt_templates[template_name] = template

        except Exception as e:
            logger.error(f"Failed to load agent config from {file_path}: {e}")

    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent"""
        return self.configs.get(agent_name)

    def get_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Alias for get_agent_config for compatibility"""
        return self.get_agent_config(agent_name)

    def get_prompt_template(self, template_name: str) -> Optional[AgentPromptTemplate]:
        """Get a prompt template by name"""
        return self.prompt_templates.get(template_name)

    def update_agent_config(self, agent_name: str, **kwargs):
        """Update an agent's configuration at runtime"""
        if agent_name in self.configs:
            config = self.configs[agent_name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    logger.info(f"Updated {agent_name} config: {key} = {value}")

    def update_config(
        self, agent_name: str, updates: Dict[str, Any]
    ) -> Optional[AgentConfig]:
        """Update agent configuration with a dictionary of updates"""
        if agent_name not in self.configs:
            return None

        config = self.configs[agent_name]
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def list_configs(self) -> List[str]:
        """List all agent configuration names"""
        return list(self.configs.keys())

    def list_prompt_templates(self) -> List[str]:
        """List all prompt template names"""
        return list(self.prompt_templates.keys())

    def load_config_from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from a dictionary"""
        # Load agent configs
        if "agents" in config_dict:
            for agent_name, agent_data in config_dict["agents"].items():
                config = AgentConfig(
                    name=agent_name,
                    instructions=agent_data.get("instructions", ""),
                    model=agent_data.get("model", "gpt-4o"),
                    temperature=agent_data.get("temperature", 0.7),
                    max_tokens=agent_data.get("max_tokens"),
                    timeout=agent_data.get("timeout", 30.0),
                    retry_attempts=agent_data.get("retry_attempts", 3),
                    enable_tracing=agent_data.get("enable_tracing", True),
                    custom_prompts=agent_data.get("custom_prompts", {}),
                )
                self.configs[agent_name] = config

        # Load prompt templates
        if "prompt_templates" in config_dict:
            for template_name, template_data in config_dict["prompt_templates"].items():
                template = AgentPromptTemplate(
                    system_prompt=template_data.get("system_prompt", ""),
                    user_prompt_template=template_data.get("user_prompt_template", ""),
                    variables=template_data.get("variables", {}),
                )
                self.prompt_templates[template_name] = template

    def _validate_config(self, config_data: Dict[str, Any]) -> bool:
        """Validate agent configuration data"""
        # Check required fields
        if "name" not in config_data or "instructions" not in config_data:
            return False

        # Check temperature range
        temperature = config_data.get("temperature", 0.7)
        if not 0.0 <= temperature <= 2.0:
            return False

        # Check timeout is positive
        timeout = config_data.get("timeout", 30.0)
        if timeout <= 0:
            return False

        return True

    def save_config_to_file(self, filename: str, agents: List[str] = None):
        """Save current configurations to a file"""
        file_path = self.config_dir / filename

        # Prepare data structure
        data = {"agents": {}, "prompt_templates": {}}

        # Add agent configs
        agents_to_save = agents if agents else list(self.configs.keys())
        for agent_name in agents_to_save:
            if agent_name in self.configs:
                config = self.configs[agent_name]
                data["agents"][agent_name] = asdict(config)

        # Add prompt templates
        for template_name, template in self.prompt_templates.items():
            data["prompt_templates"][template_name] = asdict(template)

        try:
            with open(file_path, "w") as f:
                if filename.endswith(".yaml") or filename.endswith(".yml"):
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(data, f, indent=2)
            logger.info(f"Saved agent configurations to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save agent config to {file_path}: {e}")

    def _create_default_generator_configs(self):
        """Create default configuration for generator agents"""
        config = {
            "agents": {
                "subject_expert": {
                    "instructions": """You are a world-class expert in {subject} with deep pedagogical knowledge. 
Your role is to generate high-quality flashcards that demonstrate mastery of {subject} concepts.

Key responsibilities:
- Ensure technical accuracy and depth appropriate for the target level
- Use domain-specific terminology correctly
- Include practical applications and real-world examples
- Connect concepts to prerequisite knowledge
- Avoid oversimplification while maintaining clarity

Generate cards that test understanding, not just memorization.""",
                    "model": "gpt-4o",
                    "temperature": 0.7,
                    "timeout": 45.0,
                    "custom_prompts": {
                        "math": "Focus on problem-solving strategies and mathematical reasoning",
                        "science": "Emphasize experimental design and scientific method",
                        "history": "Connect events to broader historical patterns and causation",
                        "programming": "Include executable examples and best practices",
                    },
                },
                "pedagogical": {
                    "instructions": """You are an educational specialist focused on learning theory and instructional design.
Your role is to ensure all flashcards follow educational best practices.

Apply these frameworks:
- Bloom's Taxonomy: Ensure questions target appropriate cognitive levels
- Spaced Repetition: Design cards for optimal retention
- Cognitive Load Theory: Avoid overwhelming learners
- Active Learning: Encourage engagement and application

Review cards for:
- Clear learning objectives
- Appropriate difficulty progression
- Effective use of examples and analogies
- Prerequisite knowledge alignment""",
                    "model": "gpt-4o",
                    "temperature": 0.6,
                    "timeout": 30.0,
                },
                "content_structuring": {
                    "instructions": """You are a content organization specialist focused on consistency and structure.
Your role is to format and organize flashcard content for optimal learning.

Ensure all cards have:
- Consistent formatting and style
- Proper metadata and tagging
- Clear, unambiguous questions
- Complete, well-structured answers
- Appropriate examples and explanations
- Relevant categorization and difficulty levels

Maintain high standards for readability and accessibility.""",
                    "model": "gpt-4o-mini",
                    "temperature": 0.5,
                    "timeout": 25.0,
                },
                "generation_coordinator": {
                    "instructions": """You are the generation workflow coordinator. 
Your role is to orchestrate the card generation process and manage handoffs between specialized agents.

Responsibilities:
- Route requests to appropriate specialist agents
- Coordinate parallel generation tasks
- Manage workflow state and progress
- Handle errors and fallback strategies
- Optimize generation pipelines

Make decisions based on content type, user preferences, and system load.""",
                    "model": "gpt-4o-mini",
                    "temperature": 0.3,
                    "timeout": 20.0,
                },
            },
            "prompt_templates": {
                "subject_generation": {
                    "system_prompt": "You are an expert in {subject}. Generate {num_cards} flashcards covering key concepts.",
                    "user_prompt_template": "Topic: {topic}\nDifficulty: {difficulty}\nPrerequisites: {prerequisites}\n\nGenerate cards that help learners master this topic.",
                    "variables": {
                        "subject": "general",
                        "num_cards": "5",
                        "difficulty": "intermediate",
                        "prerequisites": "none",
                    },
                }
            },
        }

        with open(self.config_dir / "defaults" / "generators.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    def _create_default_judge_configs(self):
        """Create default configuration for judge agents"""
        config = {
            "agents": {
                "content_accuracy_judge": {
                    "instructions": """You are a fact-checking and accuracy specialist.
Your role is to verify the correctness and accuracy of flashcard content.

Evaluate cards for:
- Factual accuracy and up-to-date information
- Proper use of terminology and definitions
- Absence of misconceptions or errors
- Appropriate level of detail for the target audience
- Consistency with authoritative sources

Rate each card's accuracy and provide specific feedback on any issues found.""",
                    "model": "gpt-4o",
                    "temperature": 0.3,
                    "timeout": 25.0,
                },
                "pedagogical_judge": {
                    "instructions": """You are an educational assessment specialist.
Your role is to evaluate flashcards for pedagogical effectiveness.

Assess cards for:
- Alignment with learning objectives
- Appropriate difficulty level and cognitive load
- Effective use of educational principles
- Clear prerequisite knowledge requirements
- Potential for promoting deep learning

Provide detailed feedback on educational effectiveness and improvement suggestions.""",
                    "model": "gpt-4o",
                    "temperature": 0.4,
                    "timeout": 30.0,
                },
                "clarity_judge": {
                    "instructions": """You are a communication and clarity specialist.
Your role is to ensure flashcards are clear, unambiguous, and well-written.

Evaluate cards for:
- Question clarity and specificity
- Answer completeness and coherence
- Absence of ambiguity or confusion
- Appropriate language level for target audience
- Effective use of examples and explanations

Rate clarity and provide specific suggestions for improvement.""",
                    "model": "gpt-4o-mini",
                    "temperature": 0.3,
                    "timeout": 20.0,
                },
                "technical_judge": {
                    "instructions": """You are a technical accuracy specialist for programming and technical content.
Your role is to verify technical correctness and best practices.

For technical cards, check:
- Code syntax and functionality
- Best practices and conventions
- Security considerations
- Performance implications
- Tool and framework accuracy

Provide detailed technical feedback and corrections.""",
                    "model": "gpt-4o",
                    "temperature": 0.2,
                    "timeout": 35.0,
                },
                "completeness_judge": {
                    "instructions": """You are a completeness and quality assurance specialist.
Your role is to ensure flashcards meet all requirements and quality standards.

Verify cards have:
- All required fields and metadata
- Proper formatting and structure
- Appropriate tags and categorization
- Complete explanations and examples
- Consistent quality across the set

Rate completeness and identify missing elements.""",
                    "model": "gpt-4o-mini",
                    "temperature": 0.3,
                    "timeout": 20.0,
                },
                "judge_coordinator": {
                    "instructions": """You are the quality assurance coordinator.
Your role is to orchestrate the judging process and synthesize feedback from specialist judges.

Responsibilities:
- Route cards to appropriate specialist judges
- Coordinate parallel judging tasks
- Synthesize feedback from multiple judges
- Make final accept/reject/revise decisions
- Manage judge workload and performance

Balance speed with thoroughness in quality assessment.""",
                    "model": "gpt-4o-mini",
                    "temperature": 0.3,
                    "timeout": 15.0,
                },
            }
        }

        with open(self.config_dir / "defaults" / "judges.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    def _create_default_enhancer_configs(self):
        """Create default configuration for enhancement agents"""
        config = {
            "agents": {
                "revision_agent": {
                    "instructions": """You are a content revision specialist.
Your role is to improve flashcards based on feedback from quality judges.

For each revision request:
- Analyze specific feedback provided
- Make targeted improvements to address issues
- Maintain the card's educational intent
- Preserve correct information while fixing problems
- Improve clarity, accuracy, and pedagogical value

Focus on iterative improvement rather than complete rewrites.""",
                    "model": "gpt-4o",
                    "temperature": 0.6,
                    "timeout": 40.0,
                },
                "enhancement_agent": {
                    "instructions": """You are a content enhancement specialist.
Your role is to add missing elements and enrich flashcard content.

Enhancement tasks:
- Add missing explanations or examples
- Improve metadata and tagging
- Generate additional context or background
- Create connections to related concepts
- Enhance visual or structural elements

Ensure enhancements add value without overwhelming the learner.""",
                    "model": "gpt-4o",
                    "temperature": 0.7,
                    "timeout": 35.0,
                },
            }
        }

        with open(self.config_dir / "defaults" / "enhancers.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)


# Global config manager instance
_global_config_manager: Optional[AgentConfigManager] = None


def get_config_manager() -> AgentConfigManager:
    """Get the global agent configuration manager"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = AgentConfigManager()
    return _global_config_manager
