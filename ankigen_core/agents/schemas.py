"""
Pydantic schemas for structured outputs from agents.
These schemas ensure type safety and eliminate JSON parsing errors.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class DifficultyLevel(str, Enum):
    """Difficulty levels for flashcards"""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class CardType(str, Enum):
    """Types of flashcards"""

    BASIC = "basic"
    CLOZE = "cloze"


class CardFrontSchema(BaseModel):
    """Schema for the front of a flashcard"""

    question: str = Field(..., description="The question or prompt for the flashcard")


class CardBackSchema(BaseModel):
    """Schema for the back of a flashcard"""

    answer: str = Field(..., description="The main answer to the question")
    explanation: str = Field(..., description="Detailed explanation of the answer")
    example: str = Field(..., description="A concrete example illustrating the concept")


class CardMetadataSchema(BaseModel):
    """Schema for flashcard metadata"""

    topic: str = Field(..., description="The main topic of the card")
    subject: str = Field(..., description="The subject area (e.g., Biology, History)")
    difficulty: DifficultyLevel = Field(..., description="The difficulty level")
    tags: Optional[List[str]] = Field(
        None, description="Relevant tags for categorization"
    )
    learning_outcomes: Optional[List[str]] = Field(
        None, description="What the learner should achieve"
    )
    prerequisites: Optional[List[str]] = Field(
        None, description="Required prior knowledge"
    )
    related_concepts: Optional[List[str]] = Field(
        None, description="Related concepts to explore"
    )
    estimated_time: Optional[str] = Field(None, description="Estimated time to learn")
    common_mistakes: Optional[List[str]] = Field(
        None, description="Common mistakes to avoid"
    )
    memory_aids: Optional[List[str]] = Field(
        None, description="Memory aids or mnemonics"
    )
    real_world_applications: Optional[List[str]] = Field(
        None, description="Real-world applications"
    )


class CardSchema(BaseModel):
    """Complete schema for a flashcard"""

    card_type: CardType = Field(..., description="The type of flashcard")
    front: CardFrontSchema = Field(..., description="The front of the card")
    back: CardBackSchema = Field(..., description="The back of the card")
    metadata: CardMetadataSchema = Field(..., description="Metadata about the card")
    enhancement_notes: Optional[str] = Field(
        None, description="Notes about enhancements made"
    )


class CardsGenerationSchema(BaseModel):
    """Schema for multiple cards generation"""

    cards: List[CardSchema] = Field(..., description="List of generated flashcards")


class JudgeDecisionSchema(BaseModel):
    """Schema for judge decisions"""

    approved: bool = Field(..., description="Whether the card is approved")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Quality score between 0 and 1"
    )
    feedback: str = Field(..., description="Detailed feedback about the card")
    improvements: Optional[List[str]] = Field(
        None, description="Suggested improvements"
    )
    reasoning: str = Field(..., description="Detailed reasoning for the decision")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the decision"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class EnhancementSchema(BaseModel):
    """Schema for card enhancements"""

    enhanced_card: CardSchema = Field(..., description="The enhanced flashcard")
    enhancement_summary: str = Field(..., description="Summary of what was enhanced")
    enhancement_details: Optional[Dict[str, Any]] = Field(
        None, description="Detailed enhancement information"
    )


class GenerationRequestSchema(BaseModel):
    """Schema for generation requests"""

    topic: str = Field(..., description="The topic to generate cards for")
    subject: str = Field(..., description="The subject area")
    num_cards: int = Field(..., ge=1, le=20, description="Number of cards to generate")
    difficulty: DifficultyLevel = Field(..., description="Target difficulty level")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")


class TokenUsageSchema(BaseModel):
    """Schema for token usage tracking"""

    prompt_tokens: int = Field(..., ge=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(
        ..., ge=0, description="Number of tokens in the completion"
    )
    total_tokens: int = Field(..., ge=0, description="Total tokens used")
    estimated_cost: float = Field(..., ge=0.0, description="Estimated cost in USD")
    model: str = Field(..., description="Model used for the request")
