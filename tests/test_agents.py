import pytest
from unittest.mock import MagicMock, AsyncMock
from openai import AsyncOpenAI
from ankigen.agents.generators import (
    SubjectExpertAgent,
    QualityReviewAgent,
    card_dict_to_card,
)
from ankigen.models import Card, CardFront, CardBack

# --- card_dict_to_card Tests ---


def test_card_dict_to_card_success():
    data = {
        "card_type": "basic",
        "front": {"question": "Q?"},
        "back": {"answer": "A", "explanation": "E", "example": "Ex"},
        "metadata": {"subject": "math"},
    }
    card = card_dict_to_card(data, "topic", "subject")
    assert card.card_type == "basic"
    assert card.front.question == "Q?"
    assert card.back.answer == "A"
    assert card.metadata["subject"] == "math"


def test_card_dict_to_card_defaults():
    data = {"front": {"question": "Q?"}, "back": {"answer": "A"}}
    card = card_dict_to_card(data, "Default Topic", "Default Subject")
    assert card.card_type == "basic"
    assert card.metadata["topic"] == "Default Topic"
    assert card.metadata["subject"] == "Default Subject"


def test_card_dict_to_card_invalid_payload():
    with pytest.raises(ValueError, match="Card payload must be a dictionary"):
        card_dict_to_card("not a dict", "t", "s")


def test_card_dict_to_card_missing_front():
    with pytest.raises(ValueError, match="Card front must include a question field"):
        card_dict_to_card({"back": {"answer": "A"}}, "t", "s")


# --- SubjectExpertAgent Tests ---


@pytest.fixture
def mock_openai_client():
    return MagicMock(spec=AsyncOpenAI)


@pytest.fixture
def subject_expert_agent(mock_openai_client, mocker):
    # Mock config manager
    mock_config = MagicMock()
    mock_config.instructions = "base instructions"
    mock_config.custom_prompts = {"math": "math prompt"}
    mocker.patch(
        "ankigen.agents.generators.get_config_manager"
    ).return_value.get_agent_config.return_value = mock_config

    return SubjectExpertAgent(mock_openai_client, subject="math")


def test_subject_expert_agent_init(subject_expert_agent):
    assert subject_expert_agent.subject == "math"
    assert "math prompt" in subject_expert_agent.config.instructions


def test_subject_expert_agent_build_batch_prompt(subject_expert_agent):
    prompt = subject_expert_agent._build_batch_prompt(
        topic="Addition",
        cards_in_batch=5,
        batch_num=1,
        context={"generate_cloze": True, "learning_preferences": "Focus on speed"},
        previous_topics=["Numbers"],
    )
    assert "Generate 5 flashcards" in prompt
    assert "Addition" in prompt
    assert "cloze cards" in prompt
    assert "Focus on speed" in prompt
    assert "Numbers" in prompt


def test_subject_expert_agent_extract_topics_for_dedup(subject_expert_agent):
    card1 = Card(
        front=CardFront(question="What is 1+1?"),
        back=CardBack(answer="2", explanation="E", example="X"),
    )
    card2 = Card(
        front=CardFront(question="How to multiply numbers?"),
        back=CardBack(answer="A", explanation="E", example="X"),
    )

    topics = subject_expert_agent._extract_topics_for_dedup([card1, card2])
    assert len(topics) == 2
    assert "what" in topics[0].lower() or "multiply" in topics[1].lower()


def test_subject_expert_agent_accumulate_usage(subject_expert_agent):
    total = {"total_tokens": 100}
    batch = {"total_tokens": 50}
    subject_expert_agent._accumulate_usage(total, batch)
    assert total["total_tokens"] == 150


@pytest.mark.anyio
async def test_subject_expert_agent_generate_cards_success(
    subject_expert_agent, mocker
):
    mocker.patch.object(subject_expert_agent, "initialize", new_callable=AsyncMock)

    mock_response = MagicMock()
    mock_response.cards = [{"front": {"question": "Q1"}, "back": {"answer": "A1"}}]
    mocker.patch.object(
        subject_expert_agent,
        "execute",
        new_callable=AsyncMock,
        return_value=(mock_response, {"total_tokens": 10}),
    )

    cards = await subject_expert_agent.generate_cards("Algebra", num_cards=1)
    assert len(cards) == 1
    assert cards[0].front.question == "Q1"


# --- QualityReviewAgent Tests ---


@pytest.fixture
def quality_review_agent(mock_openai_client):
    return QualityReviewAgent(mock_openai_client, model="gpt-4")


@pytest.mark.anyio
async def test_quality_review_agent_approve(quality_review_agent, mocker):
    mock_card = Card(
        card_type="basic",
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="X"),
    )

    mock_response = '{"approved": true, "reason": "Good card"}'
    mocker.patch.object(
        quality_review_agent,
        "execute",
        new_callable=AsyncMock,
        return_value=(mock_response, {}),
    )

    revised_card, approved, reason = await quality_review_agent.review_card(mock_card)
    assert approved is True
    assert reason == "Good card"
    assert revised_card is mock_card


@pytest.mark.anyio
async def test_quality_review_agent_revise(quality_review_agent, mocker):
    mock_card = Card(
        card_type="basic",
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="X"),
        metadata={"subject": "math", "topic": "arithmetic"},
    )

    mock_response = {
        "approved": False,
        "reason": "Clarify",
        "revised_card": {
            "card_type": "basic",
            "front": {"question": "Revised Q"},
            "back": {"answer": "Revised A", "explanation": "E", "example": "X"},
            "metadata": {"subject": "math", "topic": "arithmetic"},
        },
    }
    mocker.patch.object(
        quality_review_agent,
        "execute",
        new_callable=AsyncMock,
        return_value=(mock_response, {}),
    )

    revised_card, approved, reason = await quality_review_agent.review_card(mock_card)
    assert approved is False
    assert revised_card.front.question == "Revised Q"
