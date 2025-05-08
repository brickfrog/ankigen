# Tests for ankigen_core/models.py
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


# Tests for Step model
def test_step_creation():
    step = Step(explanation="Test explanation", output="Test output")
    assert step.explanation == "Test explanation"
    assert step.output == "Test output"


def test_step_missing_fields():
    with pytest.raises(ValidationError):
        Step(output="Test output")  # Missing explanation
    with pytest.raises(ValidationError):
        Step(explanation="Test explanation")  # Missing output


# Tests for Subtopics model
def test_subtopics_creation():
    step1 = Step(explanation="Expl1", output="Out1")
    step2 = Step(explanation="Expl2", output="Out2")
    subtopics = Subtopics(steps=[step1, step2], result=["Res1", "Res2"])
    assert len(subtopics.steps) == 2
    assert subtopics.steps[0].explanation == "Expl1"
    assert subtopics.result == ["Res1", "Res2"]


def test_subtopics_missing_fields():
    with pytest.raises(ValidationError):
        Subtopics(result=["Res1"])  # Missing steps
    with pytest.raises(ValidationError):
        Subtopics(steps=[Step(explanation="e", output="o")])  # Missing result


def test_subtopics_incorrect_types():
    with pytest.raises(ValidationError):
        Subtopics(steps="not a list", result=["Res1"])
    with pytest.raises(ValidationError):
        Subtopics(steps=[Step(explanation="e", output="o")], result="not a list")
    with pytest.raises(ValidationError):
        Subtopics(steps=["not a step"], result=["Res1"])


# Tests for Topics model
def test_topics_creation():
    step = Step(explanation="e", output="o")
    subtopic1 = Subtopics(steps=[step], result=["R1"])
    topics = Topics(result=[subtopic1])
    assert len(topics.result) == 1
    assert topics.result[0].steps[0].explanation == "e"


def test_topics_missing_fields():
    with pytest.raises(ValidationError):
        Topics()


def test_topics_incorrect_types():
    with pytest.raises(ValidationError):
        Topics(result="not a list")
    with pytest.raises(ValidationError):
        Topics(result=["not a subtopic"])


# Tests for CardFront model
def test_card_front_creation():
    card_front = CardFront(question="What is Pydantic?")
    assert card_front.question == "What is Pydantic?"
    card_front_none = CardFront()
    assert card_front_none.question is None


# Tests for CardBack model
def test_card_back_creation():
    card_back = CardBack(
        answer="A data validation library",
        explanation="It uses Python type hints",
        example="class Model(BaseModel): ...",
    )
    assert card_back.answer == "A data validation library"
    assert card_back.explanation == "It uses Python type hints"
    assert card_back.example == "class Model(BaseModel): ..."

    # Test with optional answer
    card_back_no_answer = CardBack(
        explanation="Explanation only", example="Example only"
    )
    assert card_back_no_answer.answer is None
    assert card_back_no_answer.explanation == "Explanation only"
    assert card_back_no_answer.example == "Example only"


def test_card_back_missing_fields():
    with pytest.raises(ValidationError):
        CardBack(answer="A", explanation="B")  # Missing example
    with pytest.raises(ValidationError):
        CardBack(answer="A", example="C")  # Missing explanation
    # Removed the case that expected a ValidationError when 'answer' was missing,
    # as 'answer' is Optional.


# Tests for Card model
def test_card_creation():
    front = CardFront(question="Q")
    back = CardBack(answer="A", explanation="E", example="Ex")
    card = Card(front=front, back=back, metadata={"source": "test"}, card_type="basic")
    assert card.front.question == "Q"
    assert card.back.answer == "A"
    assert card.metadata == {"source": "test"}
    assert card.card_type == "basic"

    card_default_type = Card(front=front, back=back)
    assert card_default_type.card_type == "basic"


def test_card_missing_fields():
    front = CardFront(question="Q")
    back = CardBack(answer="A", explanation="E", example="Ex")
    with pytest.raises(ValidationError):
        Card(front=front)
    with pytest.raises(ValidationError):
        Card(back=back)


def test_card_incorrect_types():
    back = CardBack(answer="A", explanation="E", example="Ex")
    with pytest.raises(ValidationError):
        Card(front="not a CardFront", back=back)


# Tests for CardList model
def test_card_list_creation():
    front = CardFront(question="Q")
    back = CardBack(answer="A", explanation="E", example="Ex")
    card1 = Card(front=front, back=back)
    card_list = CardList(topic="Python Basics", cards=[card1])
    assert card_list.topic == "Python Basics"
    assert len(card_list.cards) == 1
    assert card_list.cards[0].front.question == "Q"


def test_card_list_missing_fields():
    with pytest.raises(ValidationError):
        CardList(cards=[])  # Missing topic
    with pytest.raises(ValidationError):
        CardList(topic="Topic")  # Missing cards


def test_card_list_incorrect_types():
    with pytest.raises(ValidationError):
        CardList(topic=123, cards=[])
    with pytest.raises(ValidationError):
        CardList(topic="Topic", cards="not a list")
    with pytest.raises(ValidationError):
        CardList(topic="Topic", cards=["not a card"])


# Tests for ConceptBreakdown model
def test_concept_breakdown_creation():
    cb = ConceptBreakdown(
        main_concept="Loops",
        prerequisites=["Variables"],
        learning_outcomes=["Understand for/while loops"],
        common_misconceptions=["Off-by-one errors"],
        difficulty_level="beginner",
    )
    assert cb.main_concept == "Loops"
    assert cb.prerequisites == ["Variables"]
    assert cb.learning_outcomes == ["Understand for/while loops"]
    assert cb.common_misconceptions == ["Off-by-one errors"]
    assert cb.difficulty_level == "beginner"


def test_concept_breakdown_missing_fields():
    with pytest.raises(ValidationError):
        ConceptBreakdown(
            prerequisites=[]
        )  # Missing main_concept, learning_outcomes, common_misconceptions, difficulty_level
    with pytest.raises(ValidationError):
        ConceptBreakdown(main_concept="Test")  # Missing other required


# Tests for CardGeneration model
def test_card_generation_creation():
    front = CardFront(question="What is a for loop?")
    back = CardBack(answer="A control flow statement", explanation="...", example="...")
    card = Card(front=front, back=back)
    cg = CardGeneration(
        concept="For Loops",
        thought_process="Break down the concept...",
        verification_steps=["Check for clarity"],
        card=card,
    )
    assert cg.concept == "For Loops"
    assert cg.thought_process == "Break down the concept..."
    assert cg.verification_steps == ["Check for clarity"]
    assert cg.card.front.question == "What is a for loop?"


def test_card_generation_missing_fields():
    front = CardFront(question="Q")
    back = CardBack(answer="A", explanation="E", example="Ex")
    card = Card(front=front, back=back)
    with pytest.raises(ValidationError):
        CardGeneration(
            concept="Test", thought_process="Test", verification_steps=[]
        )  # Missing card
    with pytest.raises(ValidationError):
        CardGeneration(
            concept="Test", thought_process="Test", card=card
        )  # Missing verification_steps etc.


# Tests for LearningSequence model
def test_learning_sequence_creation():
    concept = ConceptBreakdown(
        main_concept="C",
        prerequisites=["P"],
        learning_outcomes=["L"],
        common_misconceptions=["M"],
        difficulty_level="D",
    )
    front = CardFront(question="Q")
    back = CardBack(answer="A", explanation="E", example="Ex")
    card_obj = Card(front=front, back=back)
    card_gen = CardGeneration(
        concept="C", thought_process="T", verification_steps=["V"], card=card_obj
    )
    ls = LearningSequence(
        topic="Advanced Python",
        concepts=[concept],
        cards=[card_gen],
        suggested_study_order=["C"],
        review_recommendations=["Review daily"],
    )
    assert ls.topic == "Advanced Python"
    assert len(ls.concepts) == 1
    assert ls.concepts[0].main_concept == "C"
    assert len(ls.cards) == 1
    assert ls.cards[0].concept == "C"
    assert ls.suggested_study_order == ["C"]
    assert ls.review_recommendations == ["Review daily"]


def test_learning_sequence_missing_fields():
    with pytest.raises(ValidationError):
        LearningSequence(topic="Test")  # Missing concepts, cards, etc.
