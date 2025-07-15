from pydantic import BaseModel, Field
from typing import List, Optional

# Module for Pydantic data models


class Step(BaseModel):
    explanation: str
    output: str


class Subtopics(BaseModel):
    steps: List[Step]
    result: List[str]


class Topics(BaseModel):
    result: List[Subtopics]


class CardFront(BaseModel):
    question: Optional[str] = None


class CardBack(BaseModel):
    answer: Optional[str] = None
    explanation: str
    example: str


class Card(BaseModel):
    front: CardFront
    back: CardBack
    metadata: Optional[dict] = None
    card_type: str = "basic"  # Add card_type, default to basic


class CardList(BaseModel):
    topic: str
    cards: List[Card]


class ConceptBreakdown(BaseModel):
    main_concept: str
    prerequisites: List[str]
    learning_outcomes: List[str]
    difficulty_level: str  # "beginner", "intermediate", "advanced"


class CardGeneration(BaseModel):
    concept: str
    thought_process: str
    verification_steps: List[str]
    card: Card


class LearningSequence(BaseModel):
    topic: str
    concepts: List[ConceptBreakdown]
    cards: List[CardGeneration]
    suggested_study_order: List[str]
    review_recommendations: List[str]


class CrawledPage(BaseModel):
    url: str
    html_content: str
    text_content: str
    title: Optional[str] = None
    meta_description: Optional[str] = None
    meta_keywords: Optional[List[str]] = Field(default_factory=list)
    crawl_depth: int = 0
    parent_url: Optional[str] = None
