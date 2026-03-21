import pytest
from pydantic import ValidationError
from ankigen_core.models import (
    Step,
    Subtopics,
    Topics,
    CardFront,
    CardBack,
    Card,
    CardList,
    ConceptBreakdown,
    CardGeneration,
    LearningSequence,
)


def test_step_model():
    step = Step(explanation="Step 1 explanation", output="Step 1 output")
    assert step.explanation == "Step 1 explanation"
    assert step.output == "Step 1 output"


def test_subtopics_model():
    step = Step(explanation="Exp", output="Out")
    subtopics = Subtopics(steps=[step], result=["Topic 1", "Topic 2"])
    assert len(subtopics.steps) == 1
    assert subtopics.result == ["Topic 1", "Topic 2"]


def test_topics_model():
    step = Step(explanation="Exp", output="Out")
    subtopics = Subtopics(steps=[step], result=["Topic 1"])
    topics = Topics(result=[subtopics])
    assert len(topics.result) == 1
    assert topics.result[0].result == ["Topic 1"]


def test_card_front_model():
    front = CardFront(question="What is Python?")
    assert front.question == "What is Python?"

    empty_front = CardFront()
    assert empty_front.question is None


def test_card_back_model():
    back = CardBack(
        answer="A programming language",
        explanation="Python is high-level.",
        example="print('Hello')",
    )
    assert back.answer == "A programming language"
    assert back.explanation == "Python is high-level."
    assert back.example == "print('Hello')"


def test_card_model_defaults():
    front = CardFront(question="Q")
    back = CardBack(answer="A", explanation="E", example="Ex")
    card = Card(front=front, back=back)
    assert card.card_type == "basic"
    assert card.metadata is None


def test_card_model_custom():
    front = CardFront(question="Q")
    back = CardBack(answer="A", explanation="E", example="Ex")
    metadata = {"source": "web"}
    card = Card(front=front, back=back, card_type="cloze", metadata=metadata)
    assert card.card_type == "cloze"
    assert card.metadata == metadata


def test_card_list_model():
    front = CardFront(question="Q")
    back = CardBack(answer="A", explanation="E", example="Ex")
    card = Card(front=front, back=back)
    card_list = CardList(topic="Python", cards=[card])
    assert card_list.topic == "Python"
    assert len(card_list.cards) == 1


def test_concept_breakdown_model():
    concept = ConceptBreakdown(
        main_concept="Loops",
        prerequisites=["Variables"],
        learning_outcomes=["Iterate over lists"],
        difficulty_level="beginner",
    )
    assert concept.main_concept == "Loops"
    assert "Variables" in concept.prerequisites
    assert concept.difficulty_level == "beginner"


def test_card_generation_model():
    front = CardFront(question="Q")
    back = CardBack(answer="A", explanation="E", example="Ex")
    card = Card(front=front, back=back)
    gen = CardGeneration(
        concept="Loops",
        thought_process="Thinking...",
        verification_steps=["Step 1"],
        card=card,
    )
    assert gen.concept == "Loops"
    assert gen.card.front.question == "Q"


def test_learning_sequence_model():
    concept = ConceptBreakdown(
        main_concept="Loops",
        prerequisites=[],
        learning_outcomes=[],
        difficulty_level="beginner",
    )
    front = CardFront(question="Q")
    back = CardBack(answer="A", explanation="E", example="Ex")
    card = Card(front=front, back=back)
    gen = CardGeneration(
        concept="Loops", thought_process="Thinking...", verification_steps=[], card=card
    )
    seq = LearningSequence(
        topic="Python",
        concepts=[concept],
        cards=[gen],
        suggested_study_order=["Loops"],
        review_recommendations=["Review tomorrow"],
    )
    assert seq.topic == "Python"
    assert len(seq.concepts) == 1
    assert len(seq.cards) == 1


def test_validation_error():
    # Test that invalid input raises ValidationError
    with pytest.raises(ValidationError):
        Step(explanation="Missing output")
