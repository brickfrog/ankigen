import pytest
from pydantic import ValidationError
from ankigen.models import (
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
    step = Step(explanation="expl", output="out")
    assert step.explanation == "expl"
    assert step.output == "out"

    with pytest.raises(ValidationError):
        Step(explanation="expl")  # missing output


def test_subtopics_model():
    sub = Subtopics(steps=[Step(explanation="e", output="o")], result=["r1"])
    assert len(sub.steps) == 1
    assert sub.result == ["r1"]


def test_topics_model():
    topics = Topics(result=[Subtopics(steps=[], result=[])])
    assert len(topics.result) == 1


def test_card_front_model():
    cf = CardFront(question="Q?")
    assert cf.question == "Q?"
    assert CardFront().question is None


def test_card_back_model():
    cb = CardBack(answer="A", explanation="E", example="Ex")
    assert cb.answer == "A"
    assert cb.explanation == "E"
    assert cb.example == "Ex"


def test_card_model():
    cf = CardFront(question="Q")
    cb = CardBack(answer="A", explanation="E", example="Ex")
    card = Card(front=cf, back=cb, metadata={"key": "val"}, card_type="cloze")
    assert card.front.question == "Q"
    assert card.card_type == "cloze"
    assert card.metadata["key"] == "val"


def test_card_list_model():
    cf = CardFront(question="Q")
    cb = CardBack(answer="A", explanation="E", example="Ex")
    card = Card(front=cf, back=cb)
    cl = CardList(topic="test", cards=[card])
    assert cl.topic == "test"
    assert len(cl.cards) == 1


def test_concept_breakdown_model():
    cb = ConceptBreakdown(
        main_concept="C",
        prerequisites=["P"],
        learning_outcomes=["L"],
        difficulty_level="beginner",
    )
    assert cb.difficulty_level == "beginner"


def test_card_generation_model():
    cf = CardFront(question="Q")
    cb = CardBack(answer="A", explanation="E", example="Ex")
    card = Card(front=cf, back=cb)
    cg = CardGeneration(
        concept="C", thought_process="T", verification_steps=["V"], card=card
    )
    assert cg.concept == "C"
    assert cg.card.front.question == "Q"


def test_learning_sequence_model():
    ls = LearningSequence(
        topic="T",
        concepts=[],
        cards=[],
        suggested_study_order=[],
        review_recommendations=[],
    )
    assert ls.topic == "T"
