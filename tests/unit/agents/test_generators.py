# Tests for ankigen_core/agents/generators.py

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from ankigen_core.agents.generators import SubjectExpertAgent, PedagogicalAgent
from ankigen_core.agents.base import AgentConfig
from ankigen_core.models import Card, CardFront, CardBack


# Test fixtures
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    return MagicMock()


@pytest.fixture
def sample_card():
    """Sample card for testing"""
    return Card(
        card_type="basic",
        front=CardFront(question="What is Python?"),
        back=CardBack(
            answer="A programming language",
            explanation="Python is a high-level, interpreted programming language",
            example="print('Hello, World!')",
        ),
        metadata={
            "difficulty": "beginner",
            "subject": "programming",
            "topic": "Python Basics",
        },
    )


@pytest.fixture
def sample_cards_json():
    """Sample JSON response for card generation"""
    return {
        "cards": [
            {
                "card_type": "basic",
                "front": {"question": "What is a Python function?"},
                "back": {
                    "answer": "A reusable block of code",
                    "explanation": "Functions help organize code into reusable components",
                    "example": "def hello(): print('hello')",
                },
                "metadata": {
                    "difficulty": "beginner",
                    "prerequisites": ["variables"],
                    "topic": "Functions",
                    "subject": "programming",
                    "learning_outcomes": ["understanding functions"],
                    "common_misconceptions": ["functions are variables"],
                },
            },
            {
                "card_type": "basic",
                "front": {"question": "How do you define a function in Python?"},
                "back": {
                    "answer": "Using the 'def' keyword",
                    "explanation": "The 'def' keyword starts a function definition",
                    "example": "def my_function(): pass",
                },
                "metadata": {
                    "difficulty": "beginner",
                    "prerequisites": ["functions"],
                    "topic": "Functions",
                    "subject": "programming",
                },
            },
        ]
    }


# Test SubjectExpertAgent
@patch("ankigen_core.agents.generators.get_config_manager")
def test_subject_expert_agent_init_with_config(
    mock_get_config_manager, mock_openai_client
):
    """Test SubjectExpertAgent initialization with existing config"""
    mock_config_manager = MagicMock()
    mock_config = AgentConfig(
        name="subject_expert", instructions="Test instructions", model="gpt-4o"
    )
    mock_config_manager.get_agent_config.return_value = mock_config
    mock_get_config_manager.return_value = mock_config_manager

    agent = SubjectExpertAgent(mock_openai_client, subject="mathematics")

    assert agent.subject == "mathematics"
    assert agent.config == mock_config
    mock_config_manager.get_agent_config.assert_called_once_with("subject_expert")


@patch("ankigen_core.agents.generators.get_config_manager")
def test_subject_expert_agent_init_fallback_config(
    mock_get_config_manager, mock_openai_client
):
    """Test SubjectExpertAgent initialization with fallback config"""
    mock_config_manager = MagicMock()
    mock_config_manager.get_agent_config.return_value = None  # No config found
    mock_get_config_manager.return_value = mock_config_manager

    agent = SubjectExpertAgent(mock_openai_client, subject="physics")

    assert agent.subject == "physics"
    assert agent.config.name == "subject_expert"
    assert "physics" in agent.config.instructions
    assert agent.config.model == "gpt-4o"


@patch("ankigen_core.agents.generators.get_config_manager")
def test_subject_expert_agent_init_with_custom_prompts(
    mock_get_config_manager, mock_openai_client
):
    """Test SubjectExpertAgent initialization with custom prompts"""
    mock_config_manager = MagicMock()
    mock_config = AgentConfig(
        name="subject_expert",
        instructions="Base instructions",
        model="gpt-4o",
        custom_prompts={"mathematics": "Focus on mathematical rigor"},
    )
    mock_config_manager.get_agent_config.return_value = mock_config
    mock_get_config_manager.return_value = mock_config_manager

    agent = SubjectExpertAgent(mock_openai_client, subject="mathematics")

    assert "Focus on mathematical rigor" in agent.config.instructions


def test_subject_expert_agent_build_generation_prompt():
    """Test building generation prompt"""
    with patch("ankigen_core.agents.generators.get_config_manager"):
        agent = SubjectExpertAgent(MagicMock(), subject="programming")

        prompt = agent._build_generation_prompt(
            topic="Python Functions",
            num_cards=3,
            difficulty="intermediate",
            prerequisites=["variables", "basic syntax"],
            context={"source_text": "Some source material about functions"},
        )

        assert "Python Functions" in prompt
        assert "3" in prompt
        assert "intermediate" in prompt
        assert "programming" in prompt
        assert "variables, basic syntax" in prompt
        assert "Some source material" in prompt


def test_subject_expert_agent_parse_cards_response_success(sample_cards_json):
    """Test successful card parsing"""
    with patch("ankigen_core.agents.generators.get_config_manager"):
        agent = SubjectExpertAgent(MagicMock(), subject="programming")

        # Test with JSON string
        json_string = json.dumps(sample_cards_json)
        cards = agent._parse_cards_response(json_string, "Functions")

        assert len(cards) == 2
        assert cards[0].front.question == "What is a Python function?"
        assert cards[0].back.answer == "A reusable block of code"
        assert cards[0].metadata["subject"] == "programming"
        assert cards[0].metadata["topic"] == "Functions"

        # Test with dict object
        cards = agent._parse_cards_response(sample_cards_json, "Functions")
        assert len(cards) == 2


def test_subject_expert_agent_parse_cards_response_invalid_json():
    """Test parsing invalid JSON response"""
    with patch("ankigen_core.agents.generators.get_config_manager"):
        agent = SubjectExpertAgent(MagicMock(), subject="programming")

        with pytest.raises(ValueError, match="Invalid JSON response"):
            agent._parse_cards_response("invalid json {", "topic")


def test_subject_expert_agent_parse_cards_response_missing_cards_field():
    """Test parsing response missing cards field"""
    with patch("ankigen_core.agents.generators.get_config_manager"):
        agent = SubjectExpertAgent(MagicMock(), subject="programming")

        invalid_response = {"wrong_field": []}
        with pytest.raises(ValueError, match="Response missing 'cards' field"):
            agent._parse_cards_response(invalid_response, "topic")


def test_subject_expert_agent_parse_cards_response_invalid_card_data():
    """Test parsing response with invalid card data"""
    with patch("ankigen_core.agents.generators.get_config_manager"):
        agent = SubjectExpertAgent(MagicMock(), subject="programming")

        invalid_cards = {
            "cards": [
                {
                    "front": {"question": "Valid question"},
                    "back": {"answer": "Valid answer"},
                },
                {
                    "front": {},  # Missing question
                    "back": {"answer": "Answer"},
                },
                {
                    "front": {"question": "Question"},
                    "back": {},  # Missing answer
                },
                "invalid_card_data",  # Not a dict
            ]
        }

        with patch("ankigen_core.logging.logger") as mock_logger:
            cards = agent._parse_cards_response(invalid_cards, "topic")

            # Should only get the valid card
            assert len(cards) == 1
            assert cards[0].front.question == "Valid question"

            # Should have logged warnings for invalid cards
            assert mock_logger.warning.call_count >= 3


@patch("ankigen_core.agents.generators.record_agent_execution")
@patch("ankigen_core.agents.generators.get_config_manager")
async def test_subject_expert_agent_generate_cards_success(
    mock_get_config_manager, mock_record, sample_cards_json, mock_openai_client
):
    """Test successful card generation"""
    mock_config_manager = MagicMock()
    mock_config_manager.get_agent_config.return_value = None
    mock_get_config_manager.return_value = mock_config_manager

    agent = SubjectExpertAgent(mock_openai_client, subject="programming")

    # Mock the execute method to return our sample response
    agent.execute = AsyncMock(return_value=json.dumps(sample_cards_json))

    cards = await agent.generate_cards(
        topic="Python Functions",
        num_cards=2,
        difficulty="beginner",
        prerequisites=["variables"],
        context={"source": "test"},
    )

    assert len(cards) == 2
    assert cards[0].front.question == "What is a Python function?"
    assert cards[0].metadata["subject"] == "programming"
    assert cards[0].metadata["topic"] == "Python Functions"

    # Verify execution was recorded
    mock_record.assert_called()
    assert mock_record.call_args[1]["success"] is True
    assert mock_record.call_args[1]["metadata"]["cards_generated"] == 2


@patch("ankigen_core.agents.generators.record_agent_execution")
@patch("ankigen_core.agents.generators.get_config_manager")
async def test_subject_expert_agent_generate_cards_error(
    mock_get_config_manager, mock_record, mock_openai_client
):
    """Test card generation with error"""
    mock_config_manager = MagicMock()
    mock_config_manager.get_agent_config.return_value = None
    mock_get_config_manager.return_value = mock_config_manager

    agent = SubjectExpertAgent(mock_openai_client, subject="programming")

    # Mock the execute method to raise an error
    agent.execute = AsyncMock(side_effect=Exception("Generation failed"))

    with pytest.raises(Exception, match="Generation failed"):
        await agent.generate_cards(topic="Test", num_cards=1)

    # Verify error was recorded
    mock_record.assert_called()
    assert mock_record.call_args[1]["success"] is False
    assert "Generation failed" in mock_record.call_args[1]["error_message"]


# Test PedagogicalAgent
@patch("ankigen_core.agents.generators.get_config_manager")
def test_pedagogical_agent_init_with_config(
    mock_get_config_manager, mock_openai_client
):
    """Test PedagogicalAgent initialization with existing config"""
    mock_config_manager = MagicMock()
    mock_config = AgentConfig(
        name="pedagogical", instructions="Pedagogical instructions", model="gpt-4o"
    )
    mock_config_manager.get_agent_config.return_value = mock_config
    mock_get_config_manager.return_value = mock_config_manager

    agent = PedagogicalAgent(mock_openai_client)

    assert agent.config == mock_config
    mock_config_manager.get_agent_config.assert_called_once_with("pedagogical")


@patch("ankigen_core.agents.generators.get_config_manager")
def test_pedagogical_agent_init_fallback_config(
    mock_get_config_manager, mock_openai_client
):
    """Test PedagogicalAgent initialization with fallback config"""
    mock_config_manager = MagicMock()
    mock_config_manager.get_agent_config.return_value = None
    mock_get_config_manager.return_value = mock_config_manager

    agent = PedagogicalAgent(mock_openai_client)

    assert agent.config.name == "pedagogical"
    assert "educational specialist" in agent.config.instructions.lower()
    assert agent.config.temperature == 0.6


@patch("ankigen_core.agents.generators.record_agent_execution")
@patch("ankigen_core.agents.generators.get_config_manager")
async def test_pedagogical_agent_review_cards_success(
    mock_get_config_manager, mock_record, mock_openai_client, sample_card
):
    """Test successful card review"""
    mock_config_manager = MagicMock()
    mock_config_manager.get_agent_config.return_value = None
    mock_get_config_manager.return_value = mock_config_manager

    agent = PedagogicalAgent(mock_openai_client)

    # Mock review response
    review_response = json.dumps(
        {
            "pedagogical_quality": 8,
            "clarity": 9,
            "learning_effectiveness": 7,
            "suggestions": ["Add more examples"],
            "cognitive_load": "appropriate",
            "bloom_taxonomy_level": "application",
        }
    )

    agent.execute = AsyncMock(return_value=review_response)

    reviews = await agent.review_cards([sample_card])

    assert len(reviews) == 1
    assert reviews[0]["pedagogical_quality"] == 8
    assert reviews[0]["clarity"] == 9
    assert "Add more examples" in reviews[0]["suggestions"]

    # Verify execution was recorded
    mock_record.assert_called()
    assert mock_record.call_args[1]["success"] is True


@patch("ankigen_core.agents.generators.get_config_manager")
def test_pedagogical_agent_build_review_prompt(
    mock_get_config_manager, mock_openai_client, sample_card
):
    """Test building review prompt"""
    mock_config_manager = MagicMock()
    mock_config_manager.get_agent_config.return_value = None
    mock_get_config_manager.return_value = mock_config_manager

    agent = PedagogicalAgent(mock_openai_client)

    prompt = agent._build_review_prompt(sample_card, 0)

    assert "What is Python?" in prompt
    assert "A programming language" in prompt
    assert "pedagogical quality" in prompt.lower()
    assert "bloom's taxonomy" in prompt.lower()
    assert "cognitive load" in prompt.lower()


@patch("ankigen_core.agents.generators.get_config_manager")
def test_pedagogical_agent_parse_review_response_success(
    mock_get_config_manager, mock_openai_client
):
    """Test successful review response parsing"""
    mock_config_manager = MagicMock()
    mock_config_manager.get_agent_config.return_value = None
    mock_get_config_manager.return_value = mock_config_manager

    agent = PedagogicalAgent(mock_openai_client)

    review_data = {
        "pedagogical_quality": 8,
        "clarity": 9,
        "learning_effectiveness": 7,
        "suggestions": ["Add more examples", "Improve explanation"],
        "cognitive_load": "appropriate",
        "bloom_taxonomy_level": "application",
    }

    # Test with JSON string
    result = agent._parse_review_response(json.dumps(review_data))
    assert result == review_data

    # Test with dict
    result = agent._parse_review_response(review_data)
    assert result == review_data


@patch("ankigen_core.agents.generators.get_config_manager")
def test_pedagogical_agent_parse_review_response_invalid_json(
    mock_get_config_manager, mock_openai_client
):
    """Test parsing invalid review response"""
    mock_config_manager = MagicMock()
    mock_config_manager.get_agent_config.return_value = None
    mock_get_config_manager.return_value = mock_config_manager

    agent = PedagogicalAgent(mock_openai_client)

    # Test invalid JSON
    with pytest.raises(ValueError, match="Invalid review response"):
        agent._parse_review_response("invalid json {")

    # Test response without required fields
    incomplete_response = {"pedagogical_quality": 8}  # Missing other required fields
    with pytest.raises(ValueError, match="Invalid review response"):
        agent._parse_review_response(incomplete_response)


@patch("ankigen_core.agents.generators.record_agent_execution")
@patch("ankigen_core.agents.generators.get_config_manager")
async def test_pedagogical_agent_review_cards_error(
    mock_get_config_manager, mock_record, mock_openai_client, sample_card
):
    """Test card review with error"""
    mock_config_manager = MagicMock()
    mock_config_manager.get_agent_config.return_value = None
    mock_get_config_manager.return_value = mock_config_manager

    agent = PedagogicalAgent(mock_openai_client)

    # Mock the execute method to raise an error
    agent.execute = AsyncMock(side_effect=Exception("Review failed"))

    with pytest.raises(Exception, match="Review failed"):
        await agent.review_cards([sample_card])

    # Verify error was recorded
    mock_record.assert_called()
    assert mock_record.call_args[1]["success"] is False


# Integration tests
@patch("ankigen_core.agents.generators.get_config_manager")
async def test_subject_expert_agent_end_to_end(
    mock_get_config_manager, mock_openai_client, sample_cards_json
):
    """Test end-to-end SubjectExpertAgent workflow"""
    mock_config_manager = MagicMock()
    mock_config_manager.get_agent_config.return_value = None
    mock_get_config_manager.return_value = mock_config_manager

    agent = SubjectExpertAgent(mock_openai_client, subject="programming")

    # Mock initialization and execution
    with patch.object(agent, "initialize") as mock_init, patch.object(
        agent, "_run_agent"
    ) as mock_run:
        mock_run.return_value = json.dumps(sample_cards_json)

        cards = await agent.generate_cards(
            topic="Python Functions",
            num_cards=2,
            difficulty="beginner",
            prerequisites=["variables"],
            context={"source_text": "Function tutorial content"},
        )

        # Verify results
        assert len(cards) == 2
        assert all(isinstance(card, Card) for card in cards)
        assert cards[0].front.question == "What is a Python function?"
        assert cards[0].metadata["subject"] == "programming"
        assert cards[0].metadata["topic"] == "Python Functions"

        # Verify agent was called correctly
        mock_init.assert_called_once()
        mock_run.assert_called_once()

        # Check that the prompt was built correctly
        call_args = mock_run.call_args[0][0]
        assert "Python Functions" in call_args
        assert "2" in call_args
        assert "beginner" in call_args
        assert "variables" in call_args
        assert "Function tutorial content" in call_args


@patch("ankigen_core.agents.generators.get_config_manager")
async def test_pedagogical_agent_end_to_end(
    mock_get_config_manager, mock_openai_client, sample_card
):
    """Test end-to-end PedagogicalAgent workflow"""
    mock_config_manager = MagicMock()
    mock_config_manager.get_agent_config.return_value = None
    mock_get_config_manager.return_value = mock_config_manager

    agent = PedagogicalAgent(mock_openai_client)

    review_response = {
        "pedagogical_quality": 8,
        "clarity": 9,
        "learning_effectiveness": 7,
        "suggestions": ["Add more practical examples"],
        "cognitive_load": "appropriate",
        "bloom_taxonomy_level": "knowledge",
    }

    # Mock initialization and execution
    with patch.object(agent, "initialize") as mock_init, patch.object(
        agent, "_run_agent"
    ) as mock_run:
        mock_run.return_value = json.dumps(review_response)

        reviews = await agent.review_cards([sample_card])

        # Verify results
        assert len(reviews) == 1
        assert reviews[0]["pedagogical_quality"] == 8
        assert reviews[0]["clarity"] == 9
        assert "Add more practical examples" in reviews[0]["suggestions"]

        # Verify agent was called correctly
        mock_init.assert_called_once()
        mock_run.assert_called_once()

        # Check that the prompt was built correctly
        call_args = mock_run.call_args[0][0]
        assert sample_card.front.question in call_args
        assert sample_card.back.answer in call_args
