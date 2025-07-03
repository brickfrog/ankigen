# Tests for ankigen_core/agents/integration.py

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Tuple

from ankigen_core.agents.integration import AgentOrchestrator, integrate_with_existing_workflow
from ankigen_core.agents.feature_flags import AgentFeatureFlags, AgentMode
from ankigen_core.llm_interface import OpenAIClientManager
from ankigen_core.models import Card, CardFront, CardBack


# Test fixtures
@pytest.fixture
def mock_client_manager():
    """Mock OpenAI client manager"""
    manager = MagicMock(spec=OpenAIClientManager)
    manager.initialize_client = AsyncMock()
    manager.get_client = MagicMock()
    return manager


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    return MagicMock()


@pytest.fixture
def sample_cards():
    """Sample cards for testing"""
    return [
        Card(
            front=CardFront(question="What is Python?"),
            back=CardBack(answer="A programming language", explanation="High-level language", example="print('hello')"),
            metadata={"subject": "programming", "difficulty": "beginner"}
        ),
        Card(
            front=CardFront(question="What is a function?"),
            back=CardBack(answer="A reusable block of code", explanation="Functions help organize code", example="def hello(): pass"),
            metadata={"subject": "programming", "difficulty": "intermediate"}
        )
    ]


@pytest.fixture
def enabled_feature_flags():
    """Feature flags with agents enabled"""
    return AgentFeatureFlags(
        mode=AgentMode.AGENT_ONLY,
        enable_subject_expert_agent=True,
        enable_pedagogical_agent=True,
        enable_content_structuring_agent=True,
        enable_generation_coordinator=True,
        enable_content_accuracy_judge=True,
        enable_pedagogical_judge=True,
        enable_judge_coordinator=True,
        enable_revision_agent=True,
        enable_enhancement_agent=True,
        enable_multi_agent_generation=True,
        enable_parallel_judging=True,
        min_judge_consensus=0.6,
        max_revision_iterations=2
    )


# Test AgentOrchestrator initialization
def test_agent_orchestrator_init(mock_client_manager):
    """Test AgentOrchestrator initialization"""
    orchestrator = AgentOrchestrator(mock_client_manager)
    
    assert orchestrator.client_manager == mock_client_manager
    assert orchestrator.openai_client is None
    assert orchestrator.generation_coordinator is None
    assert orchestrator.judge_coordinator is None
    assert orchestrator.revision_agent is None
    assert orchestrator.enhancement_agent is None
    assert orchestrator.feature_flags is not None


@patch('ankigen_core.agents.integration.get_feature_flags')
async def test_agent_orchestrator_initialize_success(mock_get_flags, mock_client_manager, mock_openai_client, enabled_feature_flags):
    """Test successful agent orchestrator initialization"""
    mock_get_flags.return_value = enabled_feature_flags
    mock_client_manager.get_client.return_value = mock_openai_client
    
    with patch('ankigen_core.agents.integration.GenerationCoordinator') as mock_gen_coord, \
         patch('ankigen_core.agents.integration.JudgeCoordinator') as mock_judge_coord, \
         patch('ankigen_core.agents.integration.RevisionAgent') as mock_revision, \
         patch('ankigen_core.agents.integration.EnhancementAgent') as mock_enhancement:
        
        orchestrator = AgentOrchestrator(mock_client_manager)
        await orchestrator.initialize("test-api-key")
        
        mock_client_manager.initialize_client.assert_called_once_with("test-api-key")
        mock_client_manager.get_client.assert_called_once()
        
        # Verify agents were initialized based on feature flags
        mock_gen_coord.assert_called_once_with(mock_openai_client)
        mock_judge_coord.assert_called_once_with(mock_openai_client)
        mock_revision.assert_called_once_with(mock_openai_client)
        mock_enhancement.assert_called_once_with(mock_openai_client)
        
        assert orchestrator.openai_client == mock_openai_client


@patch('ankigen_core.agents.integration.get_feature_flags')
async def test_agent_orchestrator_initialize_partial_flags(mock_get_flags, mock_client_manager, mock_openai_client):
    """Test agent orchestrator initialization with partial feature flags"""
    partial_flags = AgentFeatureFlags(
        mode=AgentMode.HYBRID,
        enable_generation_coordinator=True,
        enable_judge_coordinator=False,  # This should not be initialized
        enable_revision_agent=True,
        enable_enhancement_agent=False   # This should not be initialized
    )
    mock_get_flags.return_value = partial_flags
    mock_client_manager.get_client.return_value = mock_openai_client
    
    with patch('ankigen_core.agents.integration.GenerationCoordinator') as mock_gen_coord, \
         patch('ankigen_core.agents.integration.JudgeCoordinator') as mock_judge_coord, \
         patch('ankigen_core.agents.integration.RevisionAgent') as mock_revision, \
         patch('ankigen_core.agents.integration.EnhancementAgent') as mock_enhancement:
        
        orchestrator = AgentOrchestrator(mock_client_manager)
        await orchestrator.initialize("test-api-key")
        
        # Only enabled agents should be initialized
        mock_gen_coord.assert_called_once()
        mock_judge_coord.assert_not_called()
        mock_revision.assert_called_once()
        mock_enhancement.assert_not_called()


async def test_agent_orchestrator_initialize_client_error(mock_client_manager):
    """Test agent orchestrator initialization with client error"""
    mock_client_manager.initialize_client.side_effect = Exception("API key invalid")
    
    orchestrator = AgentOrchestrator(mock_client_manager)
    
    with pytest.raises(Exception, match="API key invalid"):
        await orchestrator.initialize("invalid-key")


# Test generate_cards_with_agents
@patch('ankigen_core.agents.integration.get_feature_flags')
@patch('ankigen_core.agents.integration.record_agent_execution')
async def test_generate_cards_with_agents_success(mock_record, mock_get_flags, mock_client_manager, sample_cards, enabled_feature_flags):
    """Test successful card generation with agents"""
    mock_get_flags.return_value = enabled_feature_flags
    
    orchestrator = AgentOrchestrator(mock_client_manager)
    orchestrator.openai_client = MagicMock()
    
    # Mock the phase methods
    orchestrator._generation_phase = AsyncMock(return_value=sample_cards)
    orchestrator._quality_phase = AsyncMock(return_value=(sample_cards, {"quality": "good"}))
    orchestrator._enhancement_phase = AsyncMock(return_value=sample_cards)
    
    start_time = datetime.now()
    with patch('ankigen_core.agents.integration.datetime') as mock_dt:
        mock_dt.now.return_value = start_time
        
        cards, metadata = await orchestrator.generate_cards_with_agents(
            topic="Python Basics",
            subject="programming",
            num_cards=2,
            difficulty="beginner",
            enable_quality_pipeline=True,
            context={"source": "test"}
        )
    
    assert cards == sample_cards
    assert metadata["generation_method"] == "agent_system"
    assert metadata["cards_generated"] == 2
    assert metadata["topic"] == "Python Basics"
    assert metadata["subject"] == "programming"
    assert metadata["difficulty"] == "beginner"
    assert metadata["quality_results"] == {"quality": "good"}
    
    # Verify phases were called
    orchestrator._generation_phase.assert_called_once_with(
        topic="Python Basics",
        subject="programming",
        num_cards=2,
        difficulty="beginner",
        context={"source": "test"}
    )
    orchestrator._quality_phase.assert_called_once_with(sample_cards)
    orchestrator._enhancement_phase.assert_called_once_with(sample_cards)
    
    # Verify execution was recorded
    mock_record.assert_called()


@patch('ankigen_core.agents.integration.get_feature_flags')
async def test_generate_cards_with_agents_not_enabled(mock_get_flags, mock_client_manager):
    """Test card generation when agents are not enabled"""
    legacy_flags = AgentFeatureFlags(mode=AgentMode.LEGACY)
    mock_get_flags.return_value = legacy_flags
    
    orchestrator = AgentOrchestrator(mock_client_manager)
    
    with pytest.raises(ValueError, match="Agent mode not enabled"):
        await orchestrator.generate_cards_with_agents(topic="Test", subject="test")


async def test_generate_cards_with_agents_not_initialized(mock_client_manager):
    """Test card generation when orchestrator is not initialized"""
    orchestrator = AgentOrchestrator(mock_client_manager)
    
    with pytest.raises(ValueError, match="Agent system not initialized"):
        await orchestrator.generate_cards_with_agents(topic="Test", subject="test")


@patch('ankigen_core.agents.integration.get_feature_flags')
@patch('ankigen_core.agents.integration.record_agent_execution')
async def test_generate_cards_with_agents_error(mock_record, mock_get_flags, mock_client_manager, enabled_feature_flags):
    """Test card generation with error"""
    mock_get_flags.return_value = enabled_feature_flags
    
    orchestrator = AgentOrchestrator(mock_client_manager)
    orchestrator.openai_client = MagicMock()
    orchestrator._generation_phase = AsyncMock(side_effect=Exception("Generation failed"))
    
    with pytest.raises(Exception, match="Generation failed"):
        await orchestrator.generate_cards_with_agents(topic="Test", subject="test")
    
    # Verify error was recorded
    mock_record.assert_called()
    assert mock_record.call_args[1]["success"] is False


# Test _generation_phase
@patch('ankigen_core.agents.integration.SubjectExpertAgent')
async def test_generation_phase_with_coordinator(mock_subject_expert, mock_client_manager, sample_cards, enabled_feature_flags):
    """Test generation phase with generation coordinator"""
    orchestrator = AgentOrchestrator(mock_client_manager)
    orchestrator.feature_flags = enabled_feature_flags
    orchestrator.openai_client = MagicMock()
    
    # Mock generation coordinator
    mock_coordinator = MagicMock()
    mock_coordinator.coordinate_generation = AsyncMock(return_value=sample_cards)
    orchestrator.generation_coordinator = mock_coordinator
    
    result = await orchestrator._generation_phase(
        topic="Python",
        subject="programming",
        num_cards=2,
        difficulty="beginner",
        context={"test": "context"}
    )
    
    assert result == sample_cards
    mock_coordinator.coordinate_generation.assert_called_once_with(
        topic="Python",
        subject="programming",
        num_cards=2,
        difficulty="beginner",
        enable_review=True,  # pedagogical agent enabled
        enable_structuring=True,  # content structuring enabled
        context={"test": "context"}
    )


@patch('ankigen_core.agents.integration.SubjectExpertAgent')
async def test_generation_phase_with_subject_expert(mock_subject_expert, mock_client_manager, sample_cards):
    """Test generation phase with subject expert agent only"""
    flags = AgentFeatureFlags(
        mode=AgentMode.AGENT_ONLY,
        enable_subject_expert_agent=True,
        enable_generation_coordinator=False
    )
    
    orchestrator = AgentOrchestrator(mock_client_manager)
    orchestrator.feature_flags = flags
    orchestrator.openai_client = MagicMock()
    orchestrator.generation_coordinator = None
    
    # Mock subject expert
    mock_expert_instance = MagicMock()
    mock_expert_instance.generate_cards = AsyncMock(return_value=sample_cards)
    mock_subject_expert.return_value = mock_expert_instance
    
    result = await orchestrator._generation_phase(
        topic="Python",
        subject="programming",
        num_cards=2,
        difficulty="beginner"
    )
    
    assert result == sample_cards
    mock_subject_expert.assert_called_once_with(orchestrator.openai_client, "programming")
    mock_expert_instance.generate_cards.assert_called_once_with(
        topic="Python",
        num_cards=2,
        difficulty="beginner",
        context=None
    )


async def test_generation_phase_no_agents_enabled(mock_client_manager):
    """Test generation phase with no generation agents enabled"""
    flags = AgentFeatureFlags(mode=AgentMode.LEGACY)
    
    orchestrator = AgentOrchestrator(mock_client_manager)
    orchestrator.feature_flags = flags
    orchestrator.openai_client = MagicMock()
    orchestrator.generation_coordinator = None
    
    with pytest.raises(ValueError, match="No generation agents enabled"):
        await orchestrator._generation_phase(
            topic="Python",
            subject="programming",
            num_cards=2,
            difficulty="beginner"
        )


# Test _quality_phase
async def test_quality_phase_success(mock_client_manager, sample_cards, enabled_feature_flags):
    """Test successful quality phase"""
    orchestrator = AgentOrchestrator(mock_client_manager)
    orchestrator.feature_flags = enabled_feature_flags
    
    # Mock judge coordinator
    mock_judge_coordinator = MagicMock()
    judge_results = [
        (sample_cards[0], ["decision1"], True),   # Approved
        (sample_cards[1], ["decision2"], False)  # Rejected
    ]
    mock_judge_coordinator.coordinate_judgment = AsyncMock(return_value=judge_results)
    orchestrator.judge_coordinator = mock_judge_coordinator
    
    # Mock revision agent
    revised_card = Card(
        front=CardFront(question="Revised question"),
        back=CardBack(answer="Revised answer", explanation="Revised explanation", example="Revised example")
    )
    mock_revision_agent = MagicMock()
    mock_revision_agent.revise_card = AsyncMock(return_value=revised_card)
    orchestrator.revision_agent = mock_revision_agent
    
    # Mock re-judging of revised card (approved)
    re_judge_results = [(revised_card, ["new_decision"], True)]
    mock_judge_coordinator.coordinate_judgment.side_effect = [judge_results, re_judge_results]
    
    result_cards, quality_results = await orchestrator._quality_phase(sample_cards)
    
    # Should have original approved card + revised card
    assert len(result_cards) == 2
    assert sample_cards[0] in result_cards
    assert revised_card in result_cards
    
    # Check quality results
    assert quality_results["total_cards_judged"] == 2
    assert quality_results["initially_approved"] == 1
    assert quality_results["initially_rejected"] == 1
    assert quality_results["successfully_revised"] == 1
    assert quality_results["final_approval_rate"] == 1.0
    
    # Verify calls
    assert mock_judge_coordinator.coordinate_judgment.call_count == 2
    mock_revision_agent.revise_card.assert_called_once()


async def test_quality_phase_no_judge_coordinator(mock_client_manager, sample_cards):
    """Test quality phase without judge coordinator"""
    orchestrator = AgentOrchestrator(mock_client_manager)
    orchestrator.judge_coordinator = None
    
    result_cards, quality_results = await orchestrator._quality_phase(sample_cards)
    
    assert result_cards == sample_cards
    assert quality_results["message"] == "Judge coordinator not available"


async def test_quality_phase_revision_fails(mock_client_manager, sample_cards, enabled_feature_flags):
    """Test quality phase when card revision fails"""
    orchestrator = AgentOrchestrator(mock_client_manager)
    orchestrator.feature_flags = enabled_feature_flags
    
    # Mock judge coordinator - all cards rejected
    mock_judge_coordinator = MagicMock()
    judge_results = [
        (sample_cards[0], ["decision1"], False),  # Rejected
        (sample_cards[1], ["decision2"], False)  # Rejected
    ]
    mock_judge_coordinator.coordinate_judgment = AsyncMock(return_value=judge_results)
    orchestrator.judge_coordinator = mock_judge_coordinator
    
    # Mock revision agent that fails
    mock_revision_agent = MagicMock()
    mock_revision_agent.revise_card = AsyncMock(side_effect=Exception("Revision failed"))
    orchestrator.revision_agent = mock_revision_agent
    
    result_cards, quality_results = await orchestrator._quality_phase(sample_cards)
    
    # Should have no cards (all rejected, none revised)
    assert len(result_cards) == 0
    assert quality_results["initially_approved"] == 0
    assert quality_results["initially_rejected"] == 2
    assert quality_results["successfully_revised"] == 0
    assert quality_results["final_approval_rate"] == 0.0


# Test _enhancement_phase
async def test_enhancement_phase_success(mock_client_manager, sample_cards):
    """Test successful enhancement phase"""
    orchestrator = AgentOrchestrator(mock_client_manager)
    
    enhanced_cards = [
        Card(
            front=CardFront(question="Enhanced question 1"),
            back=CardBack(answer="Enhanced answer 1", explanation="Enhanced explanation", example="Enhanced example")
        ),
        Card(
            front=CardFront(question="Enhanced question 2"),
            back=CardBack(answer="Enhanced answer 2", explanation="Enhanced explanation", example="Enhanced example")
        )
    ]
    
    mock_enhancement_agent = MagicMock()
    mock_enhancement_agent.enhance_card_batch = AsyncMock(return_value=enhanced_cards)
    orchestrator.enhancement_agent = mock_enhancement_agent
    
    result = await orchestrator._enhancement_phase(sample_cards)
    
    assert result == enhanced_cards
    mock_enhancement_agent.enhance_card_batch.assert_called_once_with(
        cards=sample_cards,
        enhancement_targets=["explanation", "example", "metadata"]
    )


async def test_enhancement_phase_no_agent(mock_client_manager, sample_cards):
    """Test enhancement phase without enhancement agent"""
    orchestrator = AgentOrchestrator(mock_client_manager)
    orchestrator.enhancement_agent = None
    
    result = await orchestrator._enhancement_phase(sample_cards)
    
    assert result == sample_cards


# Test get_performance_metrics
@patch('ankigen_core.agents.integration.get_metrics')
def test_get_performance_metrics(mock_get_metrics, mock_client_manager, enabled_feature_flags):
    """Test getting performance metrics"""
    mock_metrics = MagicMock()
    mock_metrics.get_performance_report.return_value = {"performance": "data"}
    mock_metrics.get_quality_metrics.return_value = {"quality": "data"}
    mock_get_metrics.return_value = mock_metrics
    
    orchestrator = AgentOrchestrator(mock_client_manager)
    orchestrator.feature_flags = enabled_feature_flags
    
    metrics = orchestrator.get_performance_metrics()
    
    assert "agent_performance" in metrics
    assert "quality_metrics" in metrics
    assert "feature_flags" in metrics
    assert "enabled_agents" in metrics
    
    mock_metrics.get_performance_report.assert_called_once_with(hours=24)
    mock_metrics.get_quality_metrics.assert_called_once()


# Test integrate_with_existing_workflow
@patch('ankigen_core.agents.integration.get_feature_flags')
@patch('ankigen_core.agents.integration.AgentOrchestrator')
async def test_integrate_with_existing_workflow_agents_enabled(mock_orchestrator_class, mock_get_flags, mock_client_manager, sample_cards, enabled_feature_flags):
    """Test integration with existing workflow when agents are enabled"""
    mock_get_flags.return_value = enabled_feature_flags
    
    mock_orchestrator = MagicMock()
    mock_orchestrator.initialize = AsyncMock()
    mock_orchestrator.generate_cards_with_agents = AsyncMock(return_value=(sample_cards, {"test": "metadata"}))
    mock_orchestrator_class.return_value = mock_orchestrator
    
    cards, metadata = await integrate_with_existing_workflow(
        client_manager=mock_client_manager,
        api_key="test-key",
        topic="Python",
        subject="programming"
    )
    
    assert cards == sample_cards
    assert metadata == {"test": "metadata"}
    
    mock_orchestrator_class.assert_called_once_with(mock_client_manager)
    mock_orchestrator.initialize.assert_called_once_with("test-key")
    mock_orchestrator.generate_cards_with_agents.assert_called_once_with(
        topic="Python",
        subject="programming"
    )


@patch('ankigen_core.agents.integration.get_feature_flags')
async def test_integrate_with_existing_workflow_agents_disabled(mock_get_flags, mock_client_manager):
    """Test integration with existing workflow when agents are disabled"""
    legacy_flags = AgentFeatureFlags(mode=AgentMode.LEGACY)
    mock_get_flags.return_value = legacy_flags
    
    with pytest.raises(NotImplementedError, match="Legacy fallback not implemented"):
        await integrate_with_existing_workflow(
            client_manager=mock_client_manager,
            api_key="test-key",
            topic="Python"
        )


# Integration tests
@patch('ankigen_core.agents.integration.get_feature_flags')
async def test_full_agent_workflow_integration(mock_get_flags, mock_client_manager, sample_cards, enabled_feature_flags):
    """Test complete agent workflow integration"""
    mock_get_flags.return_value = enabled_feature_flags
    mock_client_manager.get_client.return_value = MagicMock()
    
    with patch('ankigen_core.agents.integration.GenerationCoordinator') as mock_gen_coord, \
         patch('ankigen_core.agents.integration.JudgeCoordinator') as mock_judge_coord, \
         patch('ankigen_core.agents.integration.RevisionAgent') as mock_revision, \
         patch('ankigen_core.agents.integration.EnhancementAgent') as mock_enhancement, \
         patch('ankigen_core.agents.integration.record_agent_execution') as mock_record:
        
        # Mock coordinator behavior
        mock_gen_instance = MagicMock()
        mock_gen_instance.coordinate_generation = AsyncMock(return_value=sample_cards)
        mock_gen_coord.return_value = mock_gen_instance
        
        mock_judge_instance = MagicMock()
        judge_results = [(card, ["decision"], True) for card in sample_cards]  # All approved
        mock_judge_instance.coordinate_judgment = AsyncMock(return_value=judge_results)
        mock_judge_coord.return_value = mock_judge_instance
        
        mock_enhancement_instance = MagicMock()
        mock_enhancement_instance.enhance_card_batch = AsyncMock(return_value=sample_cards)
        mock_enhancement.return_value = mock_enhancement_instance
        
        # Test complete workflow
        orchestrator = AgentOrchestrator(mock_client_manager)
        await orchestrator.initialize("test-key")
        
        cards, metadata = await orchestrator.generate_cards_with_agents(
            topic="Python Functions",
            subject="programming",
            num_cards=2,
            difficulty="intermediate",
            enable_quality_pipeline=True
        )
        
        # Verify results
        assert len(cards) == 2
        assert metadata["generation_method"] == "agent_system"
        assert metadata["cards_generated"] == 2
        
        # Verify all phases were executed
        mock_gen_instance.coordinate_generation.assert_called_once()
        mock_judge_instance.coordinate_judgment.assert_called_once()
        mock_enhancement_instance.enhance_card_batch.assert_called_once()
        
        # Verify execution recording
        assert mock_record.call_count == 1
        assert mock_record.call_args[1]["success"] is True


# Error handling tests
async def test_orchestrator_handles_generation_timeout(mock_client_manager, enabled_feature_flags):
    """Test orchestrator handling of generation timeout"""
    orchestrator = AgentOrchestrator(mock_client_manager)
    orchestrator.feature_flags = enabled_feature_flags
    orchestrator.openai_client = MagicMock()
    orchestrator._generation_phase = AsyncMock(side_effect=asyncio.TimeoutError("Generation timed out"))
    
    with pytest.raises(asyncio.TimeoutError):
        await orchestrator.generate_cards_with_agents(topic="Test", subject="test")


async def test_orchestrator_handles_quality_phase_error(mock_client_manager, sample_cards, enabled_feature_flags):
    """Test orchestrator handling of quality phase error"""
    orchestrator = AgentOrchestrator(mock_client_manager)
    orchestrator.feature_flags = enabled_feature_flags
    orchestrator.openai_client = MagicMock()
    orchestrator._generation_phase = AsyncMock(return_value=sample_cards)
    orchestrator._quality_phase = AsyncMock(side_effect=Exception("Quality check failed"))
    
    with pytest.raises(Exception, match="Quality check failed"):
        await orchestrator.generate_cards_with_agents(topic="Test", subject="test")


async def test_orchestrator_handles_enhancement_error(mock_client_manager, sample_cards, enabled_feature_flags):
    """Test orchestrator handling of enhancement error"""
    orchestrator = AgentOrchestrator(mock_client_manager)
    orchestrator.feature_flags = enabled_feature_flags
    orchestrator.openai_client = MagicMock()
    orchestrator._generation_phase = AsyncMock(return_value=sample_cards)
    orchestrator._quality_phase = AsyncMock(return_value=(sample_cards, {}))
    orchestrator._enhancement_phase = AsyncMock(side_effect=Exception("Enhancement failed"))
    
    with pytest.raises(Exception, match="Enhancement failed"):
        await orchestrator.generate_cards_with_agents(topic="Test", subject="test")