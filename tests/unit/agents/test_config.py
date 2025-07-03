# Tests for ankigen_core/agents/config.py

import pytest
import json
import yaml
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import asdict

from ankigen_core.agents.config import AgentPromptTemplate, AgentConfigManager
from ankigen_core.agents.base import AgentConfig


# Test AgentPromptTemplate
def test_agent_prompt_template_creation():
    """Test basic AgentPromptTemplate creation"""
    template = AgentPromptTemplate(
        system_prompt="You are a {role} expert.",
        user_prompt_template="Please analyze: {content}",
        variables={"role": "mathematics"}
    )
    
    assert template.system_prompt == "You are a {role} expert."
    assert template.user_prompt_template == "Please analyze: {content}"
    assert template.variables == {"role": "mathematics"}


def test_agent_prompt_template_defaults():
    """Test AgentPromptTemplate with default values"""
    template = AgentPromptTemplate(
        system_prompt="System prompt",
        user_prompt_template="User prompt"
    )
    
    assert template.variables == {}


def test_agent_prompt_template_render_system_prompt():
    """Test rendering system prompt with variables"""
    template = AgentPromptTemplate(
        system_prompt="You are a {role} expert specializing in {subject}.",
        user_prompt_template="User prompt",
        variables={"role": "mathematics"}
    )
    
    rendered = template.render_system_prompt(subject="calculus")
    assert rendered == "You are a mathematics expert specializing in calculus."


def test_agent_prompt_template_render_system_prompt_override():
    """Test rendering system prompt with variable override"""
    template = AgentPromptTemplate(
        system_prompt="You are a {role} expert.",
        user_prompt_template="User prompt",
        variables={"role": "mathematics"}
    )
    
    rendered = template.render_system_prompt(role="physics")
    assert rendered == "You are a physics expert."


def test_agent_prompt_template_render_system_prompt_missing_variable():
    """Test rendering system prompt with missing variable"""
    template = AgentPromptTemplate(
        system_prompt="You are a {role} expert in {missing_var}.",
        user_prompt_template="User prompt"
    )
    
    with patch('ankigen_core.logging.logger') as mock_logger:
        rendered = template.render_system_prompt(role="mathematics")
        
        # Should return original prompt and log error
        assert rendered == "You are a {role} expert in {missing_var}."
        mock_logger.error.assert_called_once()


def test_agent_prompt_template_render_user_prompt():
    """Test rendering user prompt with variables"""
    template = AgentPromptTemplate(
        system_prompt="System prompt",
        user_prompt_template="Analyze this {content_type}: {content}",
        variables={"content_type": "text"}
    )
    
    rendered = template.render_user_prompt(content="Sample content")
    assert rendered == "Analyze this text: Sample content"


def test_agent_prompt_template_render_user_prompt_missing_variable():
    """Test rendering user prompt with missing variable"""
    template = AgentPromptTemplate(
        system_prompt="System prompt",
        user_prompt_template="Analyze {content} for {missing_var}"
    )
    
    with patch('ankigen_core.logging.logger') as mock_logger:
        rendered = template.render_user_prompt(content="test")
        
        # Should return original prompt and log error
        assert rendered == "Analyze {content} for {missing_var}"
        mock_logger.error.assert_called_once()


# Test AgentConfigManager
@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def agent_config_manager(temp_config_dir):
    """Create AgentConfigManager with temporary directory"""
    return AgentConfigManager(config_dir=temp_config_dir)


def test_agent_config_manager_init(temp_config_dir):
    """Test AgentConfigManager initialization"""
    manager = AgentConfigManager(config_dir=temp_config_dir)
    
    assert manager.config_dir == Path(temp_config_dir)
    assert isinstance(manager.configs, dict)
    assert isinstance(manager.prompt_templates, dict)
    
    # Check that default directories are created
    assert (Path(temp_config_dir) / "defaults").exists()


def test_agent_config_manager_init_default_dir():
    """Test AgentConfigManager initialization with default directory"""
    with patch('pathlib.Path.mkdir') as mock_mkdir:
        manager = AgentConfigManager()
        
        assert manager.config_dir == Path("config/agents")
        mock_mkdir.assert_called()


def test_agent_config_manager_ensure_config_dir(temp_config_dir):
    """Test _ensure_config_dir method"""
    manager = AgentConfigManager(config_dir=temp_config_dir)
    
    # Should create defaults directory
    defaults_dir = Path(temp_config_dir) / "defaults"
    assert defaults_dir.exists()


def test_agent_config_manager_load_configs_from_yaml(agent_config_manager):
    """Test loading configurations from YAML file"""
    config_data = {
        "agents": {
            "test_agent": {
                "instructions": "Test instructions",
                "model": "gpt-4o",
                "temperature": 0.8,
                "timeout": 45.0
            }
        },
        "prompt_templates": {
            "test_template": {
                "system_prompt": "System: {role}",
                "user_prompt_template": "User: {input}",
                "variables": {"role": "assistant"}
            }
        }
    }
    
    config_file = agent_config_manager.config_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)
    
    agent_config_manager._load_configs_from_file("test_config.yaml")
    
    # Check agent config was loaded
    assert "test_agent" in agent_config_manager.configs
    config = agent_config_manager.configs["test_agent"]
    assert config.name == "test_agent"
    assert config.instructions == "Test instructions"
    assert config.model == "gpt-4o"
    assert config.temperature == 0.8
    assert config.timeout == 45.0
    
    # Check prompt template was loaded
    assert "test_template" in agent_config_manager.prompt_templates
    template = agent_config_manager.prompt_templates["test_template"]
    assert template.system_prompt == "System: {role}"
    assert template.user_prompt_template == "User: {input}"
    assert template.variables == {"role": "assistant"}


def test_agent_config_manager_load_configs_from_json(agent_config_manager):
    """Test loading configurations from JSON file"""
    config_data = {
        "agents": {
            "json_agent": {
                "instructions": "JSON instructions",
                "model": "gpt-3.5-turbo",
                "temperature": 0.5
            }
        }
    }
    
    config_file = agent_config_manager.config_dir / "test_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f)
    
    agent_config_manager._load_configs_from_file("test_config.json")
    
    # Check agent config was loaded
    assert "json_agent" in agent_config_manager.configs
    config = agent_config_manager.configs["json_agent"]
    assert config.name == "json_agent"
    assert config.instructions == "JSON instructions"
    assert config.model == "gpt-3.5-turbo"
    assert config.temperature == 0.5


def test_agent_config_manager_load_nonexistent_file(agent_config_manager):
    """Test loading from non-existent file"""
    with patch('ankigen_core.logging.logger') as mock_logger:
        agent_config_manager._load_configs_from_file("nonexistent.yaml")
        
        mock_logger.warning.assert_called_once()
        assert "not found" in mock_logger.warning.call_args[0][0]


def test_agent_config_manager_load_invalid_yaml(agent_config_manager):
    """Test loading from invalid YAML file"""
    config_file = agent_config_manager.config_dir / "invalid.yaml"
    with open(config_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    
    with patch('ankigen_core.logging.logger') as mock_logger:
        agent_config_manager._load_configs_from_file("invalid.yaml")
        
        mock_logger.error.assert_called_once()


def test_agent_config_manager_get_config(agent_config_manager):
    """Test getting agent configuration"""
    # Add a test config
    test_config = AgentConfig(
        name="test_agent",
        instructions="Test instructions",
        model="gpt-4o"
    )
    agent_config_manager.configs["test_agent"] = test_config
    
    # Test getting existing config
    retrieved_config = agent_config_manager.get_config("test_agent")
    assert retrieved_config == test_config
    
    # Test getting non-existent config
    missing_config = agent_config_manager.get_config("missing_agent")
    assert missing_config is None


def test_agent_config_manager_get_prompt_template(agent_config_manager):
    """Test getting prompt template"""
    # Add a test template
    test_template = AgentPromptTemplate(
        system_prompt="Test system",
        user_prompt_template="Test user",
        variables={"var": "value"}
    )
    agent_config_manager.prompt_templates["test_template"] = test_template
    
    # Test getting existing template
    retrieved_template = agent_config_manager.get_prompt_template("test_template")
    assert retrieved_template == test_template
    
    # Test getting non-existent template
    missing_template = agent_config_manager.get_prompt_template("missing_template")
    assert missing_template is None


def test_agent_config_manager_list_configs(agent_config_manager):
    """Test listing all agent configurations"""
    # Add test configs
    config1 = AgentConfig(name="agent1", instructions="Instructions 1")
    config2 = AgentConfig(name="agent2", instructions="Instructions 2")
    agent_config_manager.configs["agent1"] = config1
    agent_config_manager.configs["agent2"] = config2
    
    config_names = agent_config_manager.list_configs()
    assert set(config_names) == {"agent1", "agent2"}


def test_agent_config_manager_list_prompt_templates(agent_config_manager):
    """Test listing all prompt templates"""
    # Add test templates
    template1 = AgentPromptTemplate(system_prompt="S1", user_prompt_template="U1")
    template2 = AgentPromptTemplate(system_prompt="S2", user_prompt_template="U2")
    agent_config_manager.prompt_templates["template1"] = template1
    agent_config_manager.prompt_templates["template2"] = template2
    
    template_names = agent_config_manager.list_prompt_templates()
    assert set(template_names) == {"template1", "template2"}


def test_agent_config_manager_update_config(agent_config_manager):
    """Test updating agent configuration"""
    # Add initial config
    initial_config = AgentConfig(
        name="test_agent",
        instructions="Initial instructions",
        temperature=0.7
    )
    agent_config_manager.configs["test_agent"] = initial_config
    
    # Update config
    updates = {"temperature": 0.9, "timeout": 60.0}
    updated_config = agent_config_manager.update_config("test_agent", updates)
    
    assert updated_config.temperature == 0.9
    assert updated_config.timeout == 60.0
    assert updated_config.instructions == "Initial instructions"  # Unchanged
    
    # Verify it's stored
    assert agent_config_manager.configs["test_agent"] == updated_config


def test_agent_config_manager_update_nonexistent_config(agent_config_manager):
    """Test updating non-existent agent configuration"""
    updates = {"temperature": 0.9}
    updated_config = agent_config_manager.update_config("missing_agent", updates)
    
    assert updated_config is None


def test_agent_config_manager_save_config_to_file(agent_config_manager):
    """Test saving configuration to file"""
    # Add test configs
    config1 = AgentConfig(name="agent1", instructions="Instructions 1", temperature=0.7)
    config2 = AgentConfig(name="agent2", instructions="Instructions 2", model="gpt-3.5-turbo")
    agent_config_manager.configs["agent1"] = config1
    agent_config_manager.configs["agent2"] = config2
    
    # Save to file
    output_file = "test_output.yaml"
    agent_config_manager.save_config_to_file(output_file)
    
    # Verify file was created
    saved_file_path = agent_config_manager.config_dir / output_file
    assert saved_file_path.exists()
    
    # Verify content
    with open(saved_file_path, 'r') as f:
        saved_data = yaml.safe_load(f)
    
    assert "agents" in saved_data
    assert "agent1" in saved_data["agents"]
    assert "agent2" in saved_data["agents"]
    assert saved_data["agents"]["agent1"]["instructions"] == "Instructions 1"
    assert saved_data["agents"]["agent1"]["temperature"] == 0.7
    assert saved_data["agents"]["agent2"]["model"] == "gpt-3.5-turbo"


def test_agent_config_manager_load_config_from_dict(agent_config_manager):
    """Test loading configuration from dictionary"""
    config_dict = {
        "agents": {
            "dict_agent": {
                "instructions": "From dict",
                "model": "gpt-4",
                "temperature": 0.3,
                "max_tokens": 1000,
                "timeout": 25.0,
                "retry_attempts": 2,
                "enable_tracing": False
            }
        },
        "prompt_templates": {
            "dict_template": {
                "system_prompt": "Dict system",
                "user_prompt_template": "Dict user",
                "variables": {"key": "value"}
            }
        }
    }
    
    agent_config_manager.load_config_from_dict(config_dict)
    
    # Check agent config
    assert "dict_agent" in agent_config_manager.configs
    config = agent_config_manager.configs["dict_agent"]
    assert config.name == "dict_agent"
    assert config.instructions == "From dict"
    assert config.model == "gpt-4"
    assert config.temperature == 0.3
    assert config.max_tokens == 1000
    assert config.timeout == 25.0
    assert config.retry_attempts == 2
    assert config.enable_tracing is False
    
    # Check prompt template
    assert "dict_template" in agent_config_manager.prompt_templates
    template = agent_config_manager.prompt_templates["dict_template"]
    assert template.system_prompt == "Dict system"
    assert template.user_prompt_template == "Dict user"
    assert template.variables == {"key": "value"}


def test_agent_config_manager_validate_config():
    """Test configuration validation"""
    manager = AgentConfigManager()
    
    # Valid config
    valid_config = {
        "name": "test_agent",
        "instructions": "Test instructions",
        "model": "gpt-4o",
        "temperature": 0.7
    }
    assert manager._validate_config(valid_config) is True
    
    # Invalid config - missing required fields
    invalid_config = {
        "name": "test_agent"
        # Missing instructions
    }
    assert manager._validate_config(invalid_config) is False
    
    # Invalid config - invalid temperature
    invalid_temp_config = {
        "name": "test_agent",
        "instructions": "Test instructions",
        "temperature": 2.0  # > 1.0
    }
    assert manager._validate_config(invalid_temp_config) is False


def test_agent_config_manager_create_default_generator_configs(temp_config_dir):
    """Test creation of default generator configurations"""
    manager = AgentConfigManager(config_dir=temp_config_dir)
    
    # Should create defaults/generators.yaml
    generators_file = Path(temp_config_dir) / "defaults" / "generators.yaml"
    assert generators_file.exists()
    
    # Check content
    with open(generators_file, 'r') as f:
        data = yaml.safe_load(f)
    
    assert "agents" in data
    # Should have at least the subject expert agent
    assert any("subject_expert" in name.lower() for name in data["agents"].keys())


def test_agent_config_manager_create_default_judge_configs(temp_config_dir):
    """Test creation of default judge configurations"""
    manager = AgentConfigManager(config_dir=temp_config_dir)
    
    # Should create defaults/judges.yaml
    judges_file = Path(temp_config_dir) / "defaults" / "judges.yaml"
    assert judges_file.exists()
    
    # Check content
    with open(judges_file, 'r') as f:
        data = yaml.safe_load(f)
    
    assert "agents" in data
    # Should have judge agents
    assert any("judge" in name.lower() for name in data["agents"].keys())


def test_agent_config_manager_create_default_enhancer_configs(temp_config_dir):
    """Test creation of default enhancer configurations"""
    manager = AgentConfigManager(config_dir=temp_config_dir)
    
    # Should create defaults/enhancers.yaml
    enhancers_file = Path(temp_config_dir) / "defaults" / "enhancers.yaml"
    assert enhancers_file.exists()
    
    # Check content
    with open(enhancers_file, 'r') as f:
        data = yaml.safe_load(f)
    
    assert "agents" in data
    # Should have enhancement agents
    assert any("enhancement" in name.lower() or "revision" in name.lower() for name in data["agents"].keys())


# Integration tests
def test_agent_config_manager_full_workflow(temp_config_dir):
    """Test complete configuration management workflow"""
    manager = AgentConfigManager(config_dir=temp_config_dir)
    
    # 1. Load configs from dict
    config_data = {
        "agents": {
            "workflow_agent": {
                "instructions": "Workflow instructions",
                "model": "gpt-4o",
                "temperature": 0.8
            }
        },
        "prompt_templates": {
            "workflow_template": {
                "system_prompt": "You are {role}",
                "user_prompt_template": "Process: {content}",
                "variables": {"role": "assistant"}
            }
        }
    }
    manager.load_config_from_dict(config_data)
    
    # 2. Update config
    manager.update_config("workflow_agent", {"timeout": 45.0})
    
    # 3. Get config and template
    config = manager.get_config("workflow_agent")
    template = manager.get_prompt_template("workflow_template")
    
    assert config.timeout == 45.0
    assert template.variables["role"] == "assistant"
    
    # 4. Save to file
    manager.save_config_to_file("workflow_output.yaml")
    
    # 5. Verify saved content
    saved_file = Path(temp_config_dir) / "workflow_output.yaml"
    with open(saved_file, 'r') as f:
        saved_data = yaml.safe_load(f)
    
    assert saved_data["agents"]["workflow_agent"]["timeout"] == 45.0
    assert saved_data["prompt_templates"]["workflow_template"]["variables"]["role"] == "assistant"