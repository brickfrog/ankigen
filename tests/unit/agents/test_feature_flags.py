# Tests for ankigen_core/agents/feature_flags.py

import pytest
import os
from unittest.mock import patch, Mock
from dataclasses import dataclass

from ankigen_core.agents.feature_flags import (
    AgentMode,
    AgentFeatureFlags,
    _env_bool,
    get_feature_flags,
    set_feature_flags,
    reset_feature_flags
)


# Test AgentMode enum
def test_agent_mode_values():
    """Test AgentMode enum values"""
    assert AgentMode.LEGACY.value == "legacy"
    assert AgentMode.AGENT_ONLY.value == "agent_only"
    assert AgentMode.HYBRID.value == "hybrid"
    assert AgentMode.A_B_TEST.value == "a_b_test"


# Test AgentFeatureFlags
def test_agent_feature_flags_defaults():
    """Test AgentFeatureFlags with default values"""
    flags = AgentFeatureFlags()
    
    assert flags.mode == AgentMode.LEGACY
    assert flags.enable_subject_expert_agent is False
    assert flags.enable_pedagogical_agent is False
    assert flags.enable_content_structuring_agent is False
    assert flags.enable_generation_coordinator is False
    
    assert flags.enable_content_accuracy_judge is False
    assert flags.enable_pedagogical_judge is False
    assert flags.enable_clarity_judge is False
    assert flags.enable_technical_judge is False
    assert flags.enable_completeness_judge is False
    assert flags.enable_judge_coordinator is False
    
    assert flags.enable_revision_agent is False
    assert flags.enable_enhancement_agent is False
    
    assert flags.enable_multi_agent_generation is False
    assert flags.enable_parallel_judging is False
    assert flags.enable_agent_handoffs is False
    assert flags.enable_agent_tracing is True
    
    assert flags.ab_test_ratio == 0.5
    assert flags.ab_test_user_hash is None
    
    assert flags.agent_timeout == 30.0
    assert flags.max_agent_retries == 3
    assert flags.enable_agent_caching is True
    
    assert flags.min_judge_consensus == 0.6
    assert flags.max_revision_iterations == 3


def test_agent_feature_flags_custom_values():
    """Test AgentFeatureFlags with custom values"""
    flags = AgentFeatureFlags(
        mode=AgentMode.AGENT_ONLY,
        enable_subject_expert_agent=True,
        enable_pedagogical_agent=True,
        enable_content_accuracy_judge=True,
        enable_multi_agent_generation=True,
        ab_test_ratio=0.7,
        agent_timeout=60.0,
        max_agent_retries=5,
        min_judge_consensus=0.8
    )
    
    assert flags.mode == AgentMode.AGENT_ONLY
    assert flags.enable_subject_expert_agent is True
    assert flags.enable_pedagogical_agent is True
    assert flags.enable_content_accuracy_judge is True
    assert flags.enable_multi_agent_generation is True
    assert flags.ab_test_ratio == 0.7
    assert flags.agent_timeout == 60.0
    assert flags.max_agent_retries == 5
    assert flags.min_judge_consensus == 0.8


@patch.dict(os.environ, {
    'ANKIGEN_AGENT_MODE': 'agent_only',
    'ANKIGEN_ENABLE_SUBJECT_EXPERT': 'true',
    'ANKIGEN_ENABLE_PEDAGOGICAL_AGENT': '1',
    'ANKIGEN_ENABLE_CONTENT_JUDGE': 'yes',
    'ANKIGEN_ENABLE_MULTI_AGENT_GEN': 'on',
    'ANKIGEN_AB_TEST_RATIO': '0.3',
    'ANKIGEN_AGENT_TIMEOUT': '45.0',
    'ANKIGEN_MAX_AGENT_RETRIES': '5',
    'ANKIGEN_MIN_JUDGE_CONSENSUS': '0.7'
}, clear=False)
def test_agent_feature_flags_from_env():
    """Test loading AgentFeatureFlags from environment variables"""
    flags = AgentFeatureFlags.from_env()
    
    assert flags.mode == AgentMode.AGENT_ONLY
    assert flags.enable_subject_expert_agent is True
    assert flags.enable_pedagogical_agent is True
    assert flags.enable_content_accuracy_judge is True
    assert flags.enable_multi_agent_generation is True
    assert flags.ab_test_ratio == 0.3
    assert flags.agent_timeout == 45.0
    assert flags.max_agent_retries == 5
    assert flags.min_judge_consensus == 0.7


@patch.dict(os.environ, {}, clear=True)
def test_agent_feature_flags_from_env_defaults():
    """Test loading AgentFeatureFlags from environment with defaults"""
    flags = AgentFeatureFlags.from_env()
    
    assert flags.mode == AgentMode.LEGACY
    assert flags.enable_subject_expert_agent is False
    assert flags.ab_test_ratio == 0.5
    assert flags.agent_timeout == 30.0
    assert flags.max_agent_retries == 3


def test_should_use_agents_legacy_mode():
    """Test should_use_agents() in LEGACY mode"""
    flags = AgentFeatureFlags(mode=AgentMode.LEGACY)
    assert flags.should_use_agents() is False


def test_should_use_agents_agent_only_mode():
    """Test should_use_agents() in AGENT_ONLY mode"""
    flags = AgentFeatureFlags(mode=AgentMode.AGENT_ONLY)
    assert flags.should_use_agents() is True


def test_should_use_agents_hybrid_mode_no_agents():
    """Test should_use_agents() in HYBRID mode with no agents enabled"""
    flags = AgentFeatureFlags(mode=AgentMode.HYBRID)
    assert flags.should_use_agents() is False


def test_should_use_agents_hybrid_mode_with_generation_agent():
    """Test should_use_agents() in HYBRID mode with generation agent enabled"""
    flags = AgentFeatureFlags(
        mode=AgentMode.HYBRID,
        enable_subject_expert_agent=True
    )
    assert flags.should_use_agents() is True


def test_should_use_agents_hybrid_mode_with_judge_agent():
    """Test should_use_agents() in HYBRID mode with judge agent enabled"""
    flags = AgentFeatureFlags(
        mode=AgentMode.HYBRID,
        enable_content_accuracy_judge=True
    )
    assert flags.should_use_agents() is True


def test_should_use_agents_ab_test_mode_with_hash():
    """Test should_use_agents() in A_B_TEST mode with user hash"""
    # Test hash that should result in False (< 50%)
    flags = AgentFeatureFlags(
        mode=AgentMode.A_B_TEST,
        ab_test_ratio=0.5,
        ab_test_user_hash="test_user_1"  # This should hash to a value < 50%
    )
    
    # Hash is deterministic, so we can test specific values
    import hashlib
    hash_value = int(hashlib.md5("test_user_1".encode()).hexdigest(), 16)
    expected_result = (hash_value % 100) < 50
    
    assert flags.should_use_agents() == expected_result


def test_should_use_agents_ab_test_mode_without_hash():
    """Test should_use_agents() in A_B_TEST mode without user hash (random)"""
    flags = AgentFeatureFlags(
        mode=AgentMode.A_B_TEST,
        ab_test_ratio=0.5
    )
    
    # Since it's random, we can't test the exact result, but we can test that it returns a boolean
    with patch('random.random') as mock_random:
        mock_random.return_value = 0.3  # < 0.5, should return True
        assert flags.should_use_agents() is True
        
        mock_random.return_value = 0.7  # > 0.5, should return False
        assert flags.should_use_agents() is False


def test_get_enabled_agents():
    """Test get_enabled_agents() method"""
    flags = AgentFeatureFlags(
        enable_subject_expert_agent=True,
        enable_pedagogical_agent=False,
        enable_content_accuracy_judge=True,
        enable_revision_agent=True
    )
    
    enabled = flags.get_enabled_agents()
    
    assert enabled["subject_expert"] is True
    assert enabled["pedagogical"] is False
    assert enabled["content_accuracy_judge"] is True
    assert enabled["revision_agent"] is True
    assert enabled["enhancement_agent"] is False  # Default false


def test_to_dict():
    """Test to_dict() method"""
    flags = AgentFeatureFlags(
        mode=AgentMode.HYBRID,
        enable_subject_expert_agent=True,
        enable_multi_agent_generation=True,
        enable_agent_tracing=False,
        ab_test_ratio=0.3,
        agent_timeout=45.0,
        max_agent_retries=5,
        min_judge_consensus=0.7,
        max_revision_iterations=2
    )
    
    result = flags.to_dict()
    
    assert result["mode"] == "hybrid"
    assert result["enabled_agents"]["subject_expert"] is True
    assert result["workflow_features"]["multi_agent_generation"] is True
    assert result["workflow_features"]["agent_tracing"] is False
    assert result["ab_test_ratio"] == 0.3
    assert result["performance_config"]["timeout"] == 45.0
    assert result["performance_config"]["max_retries"] == 5
    assert result["quality_thresholds"]["min_judge_consensus"] == 0.7
    assert result["quality_thresholds"]["max_revision_iterations"] == 2


# Test _env_bool helper function
def test_env_bool_true_values():
    """Test _env_bool() with various true values"""
    true_values = ["true", "True", "TRUE", "1", "yes", "Yes", "YES", "on", "On", "ON", "enabled", "ENABLED"]
    
    for value in true_values:
        with patch.dict(os.environ, {'TEST_VAR': value}):
            assert _env_bool('TEST_VAR') is True


def test_env_bool_false_values():
    """Test _env_bool() with various false values"""
    false_values = ["false", "False", "FALSE", "0", "no", "No", "NO", "off", "Off", "OFF", "disabled", "DISABLED", "random"]
    
    for value in false_values:
        with patch.dict(os.environ, {'TEST_VAR': value}):
            assert _env_bool('TEST_VAR') is False


def test_env_bool_default_true():
    """Test _env_bool() with default=True"""
    with patch.dict(os.environ, {}, clear=True):
        assert _env_bool('NON_EXISTENT_VAR', default=True) is True


def test_env_bool_default_false():
    """Test _env_bool() with default=False"""
    with patch.dict(os.environ, {}, clear=True):
        assert _env_bool('NON_EXISTENT_VAR', default=False) is False


# Test global flag management functions
def test_get_feature_flags_first_call():
    """Test get_feature_flags() on first call"""
    # Reset the global flag
    reset_feature_flags()
    
    with patch('ankigen_core.agents.feature_flags.AgentFeatureFlags.from_env') as mock_from_env:
        mock_flags = AgentFeatureFlags(mode=AgentMode.AGENT_ONLY)
        mock_from_env.return_value = mock_flags
        
        flags = get_feature_flags()
        
        assert flags == mock_flags
        mock_from_env.assert_called_once()


def test_get_feature_flags_subsequent_calls():
    """Test get_feature_flags() on subsequent calls (should use cached value)"""
    # Set a known flag first
    test_flags = AgentFeatureFlags(mode=AgentMode.HYBRID)
    set_feature_flags(test_flags)
    
    with patch('ankigen_core.agents.feature_flags.AgentFeatureFlags.from_env') as mock_from_env:
        flags1 = get_feature_flags()
        flags2 = get_feature_flags()
        
        assert flags1 == test_flags
        assert flags2 == test_flags
        # from_env should not be called since we already have cached flags
        mock_from_env.assert_not_called()


def test_set_feature_flags():
    """Test set_feature_flags() function"""
    test_flags = AgentFeatureFlags(
        mode=AgentMode.AGENT_ONLY,
        enable_subject_expert_agent=True
    )
    
    set_feature_flags(test_flags)
    
    retrieved_flags = get_feature_flags()
    assert retrieved_flags == test_flags
    assert retrieved_flags.mode == AgentMode.AGENT_ONLY
    assert retrieved_flags.enable_subject_expert_agent is True


def test_reset_feature_flags():
    """Test reset_feature_flags() function"""
    # Set some flags first
    test_flags = AgentFeatureFlags(mode=AgentMode.AGENT_ONLY)
    set_feature_flags(test_flags)
    
    # Verify they're set
    assert get_feature_flags() == test_flags
    
    # Reset
    reset_feature_flags()
    
    # Next call should reload from environment
    with patch('ankigen_core.agents.feature_flags.AgentFeatureFlags.from_env') as mock_from_env:
        mock_flags = AgentFeatureFlags(mode=AgentMode.HYBRID)
        mock_from_env.return_value = mock_flags
        
        flags = get_feature_flags()
        
        assert flags == mock_flags
        mock_from_env.assert_called_once()


# Integration tests for specific use cases
def test_feature_flags_production_config():
    """Test typical production configuration"""
    flags = AgentFeatureFlags(
        mode=AgentMode.HYBRID,
        enable_subject_expert_agent=True,
        enable_pedagogical_agent=True,
        enable_content_accuracy_judge=True,
        enable_judge_coordinator=True,
        enable_multi_agent_generation=True,
        enable_parallel_judging=True,
        agent_timeout=60.0,
        max_agent_retries=3,
        min_judge_consensus=0.7
    )
    
    assert flags.should_use_agents() is True
    enabled = flags.get_enabled_agents()
    assert enabled["subject_expert"] is True
    assert enabled["pedagogical"] is True
    assert enabled["content_accuracy_judge"] is True


def test_feature_flags_development_config():
    """Test typical development configuration"""
    flags = AgentFeatureFlags(
        mode=AgentMode.AGENT_ONLY,
        enable_subject_expert_agent=True,
        enable_pedagogical_agent=True,
        enable_content_accuracy_judge=True,
        enable_pedagogical_judge=True,
        enable_revision_agent=True,
        enable_multi_agent_generation=True,
        enable_agent_tracing=True,
        agent_timeout=30.0,
        max_agent_retries=2
    )
    
    assert flags.should_use_agents() is True
    config_dict = flags.to_dict()
    assert config_dict["mode"] == "agent_only"
    assert config_dict["workflow_features"]["agent_tracing"] is True


def test_feature_flags_ab_test_consistency():
    """Test A/B test consistency with same user hash"""
    flags = AgentFeatureFlags(
        mode=AgentMode.A_B_TEST,
        ab_test_ratio=0.5,
        ab_test_user_hash="consistent_user"
    )
    
    # Multiple calls with same hash should return same result
    result1 = flags.should_use_agents()
    result2 = flags.should_use_agents()
    result3 = flags.should_use_agents()
    
    assert result1 == result2 == result3