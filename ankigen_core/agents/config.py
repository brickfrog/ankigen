# Agent configuration management system

import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from jinja2 import Environment, FileSystemLoader

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
    """Manages agent configurations using Jinja templates and runtime updates"""

    def __init__(
        self,
        model_overrides: Optional[Dict[str, str]] = None,
        template_vars: Optional[Dict[str, Any]] = None,
    ):
        self.model_overrides = model_overrides or {}
        self.template_vars = template_vars or {}
        self.configs: Dict[str, AgentConfig] = {}
        self.prompt_templates: Dict[str, AgentPromptTemplate] = {}

        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        self._load_default_configs()

    def update_models(self, model_overrides: Dict[str, str]):
        """Update model selections and regenerate configs"""
        self.model_overrides = model_overrides
        self._load_default_configs()
        logger.info(f"Updated model overrides: {model_overrides}")

    def update_template_vars(self, template_vars: Dict[str, Any]):
        logger.info(
            "Template vars are no longer used in the simplified agent pipeline."
        )

    def _load_default_configs(self):
        """Load all default configurations from Jinja templates"""
        try:
            self._load_configs_from_template("generators.j2")
            self.prompt_templates.clear()
            logger.info(
                f"Loaded {len(self.configs)} agent configurations from Jinja templates"
            )
        except Exception as e:
            logger.error(f"Failed to load agent configurations from templates: {e}")

    def _get_model_for_agent(self, agent_name: str, default_model: str) -> str:
        """Get model for agent, using override if available"""
        return self.model_overrides.get(agent_name, default_model)

    def _load_configs_from_template(self, template_name: str):
        """Load agent configurations from a Jinja template"""
        try:
            template = self.jinja_env.get_template(template_name)

            # Default models for each agent type
            default_models = {
                "subject_expert_model": "gpt-5.1",
            }

            # Simple mapping: agent_name -> agent_name_model
            model_vars = {}
            for agent_name, model in self.model_overrides.items():
                model_vars[f"{agent_name}_model"] = model

            # Merge all template variables with defaults
            render_vars = {**default_models, **self.template_vars, **model_vars}

            logger.info(f"Rendering template {template_name} with vars: {render_vars}")
            rendered_json = template.render(**render_vars)
            config_data = json.loads(rendered_json)

            # Create AgentConfig objects from the rendered data
            for agent_name, agent_data in config_data.items():
                config = AgentConfig(
                    name=agent_data.get("name", agent_name),
                    instructions=agent_data.get("instructions", ""),
                    model=agent_data.get("model", "gpt-5.1"),
                    temperature=agent_data.get("temperature", 0.7),
                    max_tokens=agent_data.get("max_tokens"),
                    timeout=agent_data.get("timeout", 30.0),
                    retry_attempts=agent_data.get("retry_attempts", 3),
                    enable_tracing=agent_data.get("enable_tracing", True),
                    custom_prompts=agent_data.get("custom_prompts", {}),
                )
                self.configs[agent_name] = config
                logger.info(f"Loaded config for {agent_name}: model={config.model}")

        except Exception as e:
            logger.error(f"Failed to load configs from template {template_name}: {e}")

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
                    model=agent_data.get("model", "gpt-5.1"),
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
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved agent configurations to {filename}")
        except Exception as e:
            logger.error(f"Failed to save agent config to {filename}: {e}")


# Global config manager instance
_global_config_manager: Optional[AgentConfigManager] = None


def get_config_manager(
    model_overrides: Optional[Dict[str, str]] = None,
    template_vars: Optional[Dict[str, Any]] = None,
) -> AgentConfigManager:
    """Get the global agent configuration manager"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = AgentConfigManager(model_overrides, template_vars)
    else:
        if model_overrides:
            _global_config_manager.update_models(model_overrides)
        if template_vars:
            _global_config_manager.update_template_vars(template_vars)
    return _global_config_manager
