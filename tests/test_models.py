import pytest
from pydantic import ValidationError
from ankigen_core.models import (
    Card,
    CardFront,
    CardBack,
    CardList,
    ConceptBreakdown,
    CardGeneration,
    LearningSequence,
)


def test_card_front_model():
    front = CardFront(question="What is Python?")
    assert front.question == "What is Python?"

    empty_front = CardFront()
    assert empty_front.question is None


def test_card_back_model():
    back = CardBack(answer="A language", explanation="Details", example="print()")
    assert back.answer == "A language"
    assert back.explanation == "Details"
    assert back.example == "print()"


def test_card_model():
    front = CardFront(question="Q")
    back = CardBack(answer="A", explanation="E", example="Ex")
    card = Card(front=front, back=back, card_type="cloze")

    assert card.front.question == "Q"
    assert card.card_type == "cloze"
    assert card.metadata is None


def test_card_list_model():
    card = Card(
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="Ex"),
    )
    card_list = CardList(topic="Test", cards=[card])
    assert card_list.topic == "Test"
    assert len(card_list.cards) == 1


def test_concept_breakdown_model():
    concept = ConceptBreakdown(
        main_concept="Python",
        prerequisites=["Math"],
        learning_outcomes=["Coding"],
        difficulty_level="beginner",
    )
    assert concept.main_concept == "Python"
    assert concept.difficulty_level == "beginner"


def test_concept_breakdown_invalid_difficulty():
    # pydantic doesn't have Literal here, but let's check if it exists
    # If not defined as Literal, it will accept anything.
    # Looking at models.py, it's just a str.
    concept = ConceptBreakdown(
        main_concept="P", prerequisites=[], learning_outcomes=[], difficulty_level="pro"
    )
    assert concept.difficulty_level == "pro"


def test_card_generation_model():
    card = Card(
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="Ex"),
    )
    gen = CardGeneration(
        concept="C", thought_process="T", verification_steps=["V"], card=card
    )
    assert gen.concept == "C"
    assert gen.card.front.question == "Q"


def test_learning_sequence_model():
    ls = LearningSequence(
        topic="T",
        concepts=[],
        cards=[],
        suggested_study_order=[],
        review_recommendations=[],
    )
    assert ls.topic == "T"


def test_pydantic_validation_error():
    with pytest.raises(ValidationError):
        # Missing required field 'explanation' in CardBack
        CardBack(answer="A", example="Ex")
