# Integration tests for agent workflows

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from ankigen_core.agents.integration import AgentOrchestrator, integrate_with_existing_workflow
from ankigen_core.agents.feature_flags import AgentFeatureFlags, AgentMode
from ankigen_core.agents.config import AgentConfigManager
from ankigen_core.llm_interface import OpenAIClientManager
from ankigen_core.models import Card, CardFront, CardBack


# Test fixtures
@pytest.fixture
def temp_config_dir():
    """Create temporary config directory for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_cards():
    """Sample cards for testing workflows"""
    return [
        Card(
            card_type="basic",
            front=CardFront(question="What is a Python function?"),
            back=CardBack(
                answer="A reusable block of code",
                explanation="Functions help organize code into reusable components",
                example="def hello(): print('hello')"
            ),
            metadata={
                "difficulty": "beginner",
                "subject": "programming",
                "topic": "Python Functions",
                "learning_outcomes": ["understanding functions"],
                "quality_score": 8.5
            }
        ),
        Card(
            card_type="basic",
            front=CardFront(question="How do you call a function in Python?"),
            back=CardBack(
                answer="By using the function name followed by parentheses",
                explanation="Function calls execute the code inside the function",
                example="hello()"
            ),
            metadata={
                "difficulty": "beginner",
                "subject": "programming",
                "topic": "Python Functions",
                "learning_outcomes": ["function execution"],
                "quality_score": 7.8
            }
        )
    ]


@pytest.fixture
def mock_openai_responses():
    """Mock OpenAI API responses for different agents"""
    return {
        "generation": {
            "cards": [
                {
                    "card_type": "basic",
                    "front": {"question": "What is a Python function?"},
                    "back": {
                        "answer": "A reusable block of code",
                        "explanation": "Functions help organize code",
                        "example": "def hello(): print('hello')"
                    },
                    "metadata": {
                        "difficulty": "beginner",
                        "subject": "programming",
                        "topic": "Functions"
                    }
                }
            ]
        },
        "judgment": {
            "approved": True,
            "quality_score": 8.5,
            "feedback": "Good question with clear answer",
            "suggestions": []
        },
        "enhancement": {
            "enhanced_explanation": "Functions help organize code into reusable, testable components",
            "enhanced_example": "def greet(name): return f'Hello, {name}!'",
            "additional_metadata": {
                "complexity": "low",
                "estimated_study_time": "5 minutes"
            }
        }
    }


# Test complete agent workflow
@patch('ankigen_core.agents.integration.get_feature_flags')
@patch('ankigen_core.agents.integration.record_agent_execution')
async def test_complete_agent_workflow_success(mock_record, mock_get_flags, sample_cards, mock_openai_responses):
    """Test complete agent workflow from generation to enhancement"""
    
    # Setup feature flags for full agent mode
    feature_flags = AgentFeatureFlags(
        mode=AgentMode.AGENT_ONLY,
        enable_generation_coordinator=True,
        enable_judge_coordinator=True,
        enable_revision_agent=True,
        enable_enhancement_agent=True,
        enable_parallel_judging=True,
        min_judge_consensus=0.6
    )
    mock_get_flags.return_value = feature_flags
    
    # Mock client manager
    mock_client_manager = MagicMock(spec=OpenAIClientManager)
    mock_client_manager.initialize_client = AsyncMock()
    mock_openai_client = MagicMock()
    mock_client_manager.get_client.return_value = mock_openai_client
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(mock_client_manager)
    
    # Mock all agent components
    with patch('ankigen_core.agents.integration.GenerationCoordinator') as mock_gen_coord, \
         patch('ankigen_core.agents.integration.JudgeCoordinator') as mock_judge_coord, \
         patch('ankigen_core.agents.integration.RevisionAgent') as mock_revision, \
         patch('ankigen_core.agents.integration.EnhancementAgent') as mock_enhancement:
        
        # Setup generation coordinator
        mock_gen_instance = MagicMock()
        mock_gen_instance.coordinate_generation = AsyncMock(return_value=sample_cards)
        mock_gen_coord.return_value = mock_gen_instance
        
        # Setup judge coordinator (approve all cards)
        mock_judge_instance = MagicMock()
        judge_results = [(card, ["positive feedback"], True) for card in sample_cards]
        mock_judge_instance.coordinate_judgment = AsyncMock(return_value=judge_results)
        mock_judge_coord.return_value = mock_judge_instance
        
        # Setup enhancement agent
        enhanced_cards = sample_cards.copy()
        for card in enhanced_cards:
            card.metadata["enhanced"] = True
        mock_enhancement_instance = MagicMock()
        mock_enhancement_instance.enhance_card_batch = AsyncMock(return_value=enhanced_cards)
        mock_enhancement.return_value = mock_enhancement_instance
        
        # Initialize and run workflow
        await orchestrator.initialize("test-api-key")
        
        cards, metadata = await orchestrator.generate_cards_with_agents(
            topic="Python Functions",
            subject="programming",
            num_cards=2,
            difficulty="beginner",
            enable_quality_pipeline=True
        )
        
        # Verify results
        assert len(cards) == 2
        assert all(isinstance(card, Card) for card in cards)
        assert all(card.metadata.get("enhanced") for card in cards)
        
        # Verify metadata
        assert metadata["generation_method"] == "agent_system"
        assert metadata["cards_generated"] == 2
        assert metadata["topic"] == "Python Functions"
        assert metadata["subject"] == "programming"
        assert "quality_results" in metadata
        
        # Verify all phases were executed
        mock_gen_instance.coordinate_generation.assert_called_once()
        mock_judge_instance.coordinate_judgment.assert_called_once()
        mock_enhancement_instance.enhance_card_batch.assert_called_once()
        
        # Verify execution was recorded
        mock_record.assert_called()


@patch('ankigen_core.agents.integration.get_feature_flags')
async def test_agent_workflow_with_card_rejection_and_revision(mock_get_flags, sample_cards):
    """Test workflow when cards are rejected and need revision"""
    
    feature_flags = AgentFeatureFlags(
        mode=AgentMode.AGENT_ONLY,
        enable_generation_coordinator=True,
        enable_judge_coordinator=True,
        enable_revision_agent=True,
        max_revision_iterations=2
    )
    mock_get_flags.return_value = feature_flags
    
    # Mock client manager
    mock_client_manager = MagicMock(spec=OpenAIClientManager)
    mock_client_manager.initialize_client = AsyncMock()
    mock_openai_client = MagicMock()
    mock_client_manager.get_client.return_value = mock_openai_client
    
    orchestrator = AgentOrchestrator(mock_client_manager)
    
    with patch('ankigen_core.agents.integration.GenerationCoordinator') as mock_gen_coord, \
         patch('ankigen_core.agents.integration.JudgeCoordinator') as mock_judge_coord, \
         patch('ankigen_core.agents.integration.RevisionAgent') as mock_revision:
        
        # Setup generation coordinator
        mock_gen_instance = MagicMock()
        mock_gen_instance.coordinate_generation = AsyncMock(return_value=sample_cards)
        mock_gen_coord.return_value = mock_gen_instance
        
        # Setup judge coordinator (reject first card, approve second)
        judge_results_initial = [
            (sample_cards[0], ["unclear question"], False),  # Rejected
            (sample_cards[1], ["good question"], True)       # Approved
        ]
        
        # Create revised card
        revised_card = Card(
            card_type="basic",
            front=CardFront(question="What is a Python function and how is it used?"),
            back=CardBack(
                answer="A reusable block of code that performs a specific task",
                explanation="Functions are fundamental building blocks in programming",
                example="def add(a, b): return a + b"
            ),
            metadata={"difficulty": "beginner", "revised": True}
        )
        
        # Judge approves revised card
        judge_results_revision = [(revised_card, ["much improved"], True)]
        
        mock_judge_instance = MagicMock()
        mock_judge_instance.coordinate_judgment = AsyncMock(
            side_effect=[judge_results_initial, judge_results_revision]
        )
        mock_judge_coord.return_value = mock_judge_instance
        
        # Setup revision agent
        mock_revision_instance = MagicMock()
        mock_revision_instance.revise_card = AsyncMock(return_value=revised_card)
        mock_revision.return_value = mock_revision_instance
        
        # Initialize and run workflow
        await orchestrator.initialize("test-api-key")
        
        cards, metadata = await orchestrator.generate_cards_with_agents(
            topic="Python Functions",
            subject="programming",
            num_cards=2,
            difficulty="beginner"
        )
        
        # Verify results
        assert len(cards) == 2  # Original approved card + revised card
        assert sample_cards[1] in cards  # Originally approved card
        assert revised_card in cards      # Revised card
        
        # Verify quality results
        quality_results = metadata["quality_results"]
        assert quality_results["initially_approved"] == 1
        assert quality_results["initially_rejected"] == 1
        assert quality_results["successfully_revised"] == 1
        assert quality_results["final_approval_rate"] == 1.0
        
        # Verify revision was called
        mock_revision_instance.revise_card.assert_called_once()


@patch('ankigen_core.agents.integration.get_feature_flags')
async def test_agent_workflow_hybrid_mode(mock_get_flags, sample_cards):
    """Test workflow in hybrid mode with selective agent usage"""
    
    feature_flags = AgentFeatureFlags(
        mode=AgentMode.HYBRID,
        enable_subject_expert_agent=True,
        enable_content_accuracy_judge=True,
        enable_generation_coordinator=False,  # Not enabled
        enable_enhancement_agent=False        # Not enabled
    )
    mock_get_flags.return_value = feature_flags
    
    mock_client_manager = MagicMock(spec=OpenAIClientManager)
    mock_client_manager.initialize_client = AsyncMock()
    mock_openai_client = MagicMock()
    mock_client_manager.get_client.return_value = mock_openai_client
    
    orchestrator = AgentOrchestrator(mock_client_manager)
    
    with patch('ankigen_core.agents.integration.SubjectExpertAgent') as mock_subject_expert:
        
        # Setup subject expert agent (fallback when coordinator is disabled)
        mock_expert_instance = MagicMock()
        mock_expert_instance.generate_cards = AsyncMock(return_value=sample_cards)
        mock_subject_expert.return_value = mock_expert_instance
        
        # Initialize orchestrator (should only create enabled agents)
        await orchestrator.initialize("test-api-key")
        
        # Verify only enabled agents were created
        assert orchestrator.generation_coordinator is None    # Disabled
        assert orchestrator.judge_coordinator is None         # Not enabled in flags
        assert orchestrator.enhancement_agent is None         # Disabled
        
        # Run generation
        cards, metadata = await orchestrator.generate_cards_with_agents(
            topic="Python Functions",
            subject="programming",
            num_cards=2
        )
        
        # Verify results
        assert len(cards) == 2
        assert metadata["generation_method"] == "agent_system"
        
        # Verify subject expert was used
        mock_subject_expert.assert_called_once_with(mock_openai_client, "programming")
        mock_expert_instance.generate_cards.assert_called_once()


@patch('ankigen_core.agents.integration.get_feature_flags')
async def test_integrate_with_existing_workflow_function(mock_get_flags, sample_cards):
    """Test the integrate_with_existing_workflow function"""
    
    feature_flags = AgentFeatureFlags(mode=AgentMode.AGENT_ONLY, enable_subject_expert_agent=True)
    mock_get_flags.return_value = feature_flags
    
    mock_client_manager = MagicMock(spec=OpenAIClientManager)
    
    with patch('ankigen_core.agents.integration.AgentOrchestrator') as mock_orchestrator_class:
        
        # Mock orchestrator instance
        mock_orchestrator = MagicMock()
        mock_orchestrator.initialize = AsyncMock()
        mock_orchestrator.generate_cards_with_agents = AsyncMock(
            return_value=(sample_cards, {"method": "agent_system"})
        )
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Call integration function
        cards, metadata = await integrate_with_existing_workflow(
            client_manager=mock_client_manager,
            api_key="test-key",
            topic="Python Basics",
            subject="programming",
            num_cards=2,
            difficulty="beginner"
        )
        
        # Verify results
        assert cards == sample_cards
        assert metadata == {"method": "agent_system"}
        
        # Verify orchestrator was used correctly
        mock_orchestrator_class.assert_called_once_with(mock_client_manager)
        mock_orchestrator.initialize.assert_called_once_with("test-key")
        mock_orchestrator.generate_cards_with_agents.assert_called_once_with(
            topic="Python Basics",
            subject="programming",
            num_cards=2,
            difficulty="beginner"
        )


@patch('ankigen_core.agents.integration.get_feature_flags')
async def test_integrate_with_existing_workflow_legacy_fallback(mock_get_flags):
    """Test integration function with legacy fallback"""
    
    feature_flags = AgentFeatureFlags(mode=AgentMode.LEGACY)
    mock_get_flags.return_value = feature_flags
    
    mock_client_manager = MagicMock(spec=OpenAIClientManager)
    
    # Should raise NotImplementedError for legacy fallback
    with pytest.raises(NotImplementedError, match="Legacy fallback not implemented"):
        await integrate_with_existing_workflow(
            client_manager=mock_client_manager,
            api_key="test-key",
            topic="Test"
        )


async def test_agent_workflow_error_handling():
    """Test agent workflow error handling and recovery"""
    
    mock_client_manager = MagicMock(spec=OpenAIClientManager)
    mock_client_manager.initialize_client = AsyncMock(side_effect=Exception("API key invalid"))
    
    orchestrator = AgentOrchestrator(mock_client_manager)
    
    # Should raise initialization error
    with pytest.raises(Exception, match="API key invalid"):
        await orchestrator.initialize("invalid-key")


async def test_agent_workflow_timeout_handling():
    """Test agent workflow timeout handling"""
    
    feature_flags = AgentFeatureFlags(
        mode=AgentMode.AGENT_ONLY,
        enable_generation_coordinator=True,
        agent_timeout=0.1  # Very short timeout
    )
    
    mock_client_manager = MagicMock(spec=OpenAIClientManager)
    mock_client_manager.initialize_client = AsyncMock()
    mock_client_manager.get_client.return_value = MagicMock()
    
    orchestrator = AgentOrchestrator(mock_client_manager)
    orchestrator.feature_flags = feature_flags
    
    with patch('ankigen_core.agents.integration.GenerationCoordinator') as mock_gen_coord:
        
        # Setup generation coordinator with slow response
        mock_gen_instance = MagicMock()
        mock_gen_instance.coordinate_generation = AsyncMock()
        
        async def slow_generation(*args, **kwargs):
            await asyncio.sleep(1)  # Longer than timeout
            return []
        
        mock_gen_instance.coordinate_generation.side_effect = slow_generation
        mock_gen_coord.return_value = mock_gen_instance
        
        await orchestrator.initialize("test-key")
        
        # Should handle timeout gracefully (depends on implementation)
        # This tests the timeout mechanism in the base agent wrapper
        with pytest.raises(Exception):  # Could be TimeoutError or other exception
            await orchestrator.generate_cards_with_agents(
                topic="Test",
                subject="test",
                num_cards=1
            )


def test_agent_config_integration_with_workflow(temp_config_dir):
    """Test agent configuration integration with workflow"""
    
    # Create test configuration
    config_manager = AgentConfigManager(config_dir=temp_config_dir)
    
    test_config = {
        "agents": {
            "subject_expert": {
                "instructions": "You are a subject matter expert",
                "model": "gpt-4o",
                "temperature": 0.8,
                "timeout": 45.0,
                "custom_prompts": {
                    "programming": "Focus on code examples and best practices"
                }
            }
        }
    }
    
    config_manager.load_config_from_dict(test_config)
    
    # Verify config was loaded
    subject_config = config_manager.get_config("subject_expert")
    assert subject_config is not None
    assert subject_config.temperature == 0.8
    assert subject_config.timeout == 45.0
    assert "programming" in subject_config.custom_prompts


@patch('ankigen_core.agents.integration.get_feature_flags')
async def test_agent_performance_metrics_collection(mock_get_flags, sample_cards):
    """Test that performance metrics are collected during workflow"""
    
    feature_flags = AgentFeatureFlags(
        mode=AgentMode.AGENT_ONLY,
        enable_generation_coordinator=True,
        enable_agent_tracing=True
    )
    mock_get_flags.return_value = feature_flags
    
    mock_client_manager = MagicMock(spec=OpenAIClientManager)
    mock_client_manager.initialize_client = AsyncMock()
    mock_client_manager.get_client.return_value = MagicMock()
    
    orchestrator = AgentOrchestrator(mock_client_manager)
    
    with patch('ankigen_core.agents.integration.GenerationCoordinator') as mock_gen_coord, \
         patch('ankigen_core.agents.integration.get_metrics') as mock_get_metrics:
        
        # Setup generation coordinator
        mock_gen_instance = MagicMock()
        mock_gen_instance.coordinate_generation = AsyncMock(return_value=sample_cards)
        mock_gen_coord.return_value = mock_gen_instance
        
        # Setup metrics
        mock_metrics = MagicMock()
        mock_metrics.get_performance_report.return_value = {"avg_response_time": 1.5}
        mock_metrics.get_quality_metrics.return_value = {"avg_quality": 8.2}
        mock_get_metrics.return_value = mock_metrics
        
        await orchestrator.initialize("test-key")
        
        # Generate cards
        await orchestrator.generate_cards_with_agents(
            topic="Test",
            subject="test",
            num_cards=1
        )
        
        # Get performance metrics
        performance = orchestrator.get_performance_metrics()
        
        # Verify metrics structure
        assert "agent_performance" in performance
        assert "quality_metrics" in performance
        assert "feature_flags" in performance
        assert "enabled_agents" in performance
        
        # Verify metrics were retrieved
        mock_metrics.get_performance_report.assert_called_once_with(hours=24)
        mock_metrics.get_quality_metrics.assert_called_once()


# Stress test for concurrent agent operations
@patch('ankigen_core.agents.integration.get_feature_flags')
async def test_concurrent_agent_operations(mock_get_flags, sample_cards):
    """Test concurrent agent operations"""
    
    feature_flags = AgentFeatureFlags(
        mode=AgentMode.AGENT_ONLY,
        enable_generation_coordinator=True,
        enable_parallel_judging=True
    )
    mock_get_flags.return_value = feature_flags
    
    mock_client_manager = MagicMock(spec=OpenAIClientManager)
    mock_client_manager.initialize_client = AsyncMock()
    mock_client_manager.get_client.return_value = MagicMock()
    
    # Create multiple orchestrators for concurrent operations
    orchestrators = [AgentOrchestrator(mock_client_manager) for _ in range(3)]
    
    with patch('ankigen_core.agents.integration.GenerationCoordinator') as mock_gen_coord:
        
        # Setup generation coordinator
        mock_gen_instance = MagicMock()
        mock_gen_instance.coordinate_generation = AsyncMock(return_value=sample_cards)
        mock_gen_coord.return_value = mock_gen_instance
        
        # Initialize all orchestrators
        await asyncio.gather(*[orch.initialize("test-key") for orch in orchestrators])
        
        # Run concurrent card generation
        tasks = [
            orch.generate_cards_with_agents(
                topic=f"Topic {i}",
                subject="test",
                num_cards=1
            )
            for i, orch in enumerate(orchestrators)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all operations completed successfully
        assert len(results) == 3
        for cards, metadata in results:
            assert len(cards) == 2  # sample_cards has 2 cards
            assert metadata["generation_method"] == "agent_system"