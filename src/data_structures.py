"""Core data structures for reasoning problems and solutions."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum

from .strategies import ReasoningStrategy


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    HINDI = "hi"
    BENGALI = "bn"
    TAMIL = "ta"
    TELUGU = "te"
    MARATHI = "mr"
    GUJARATI = "gu"
    KANNADA = "kn"
    MALAYALAM = "ml"
    PUNJABI = "pa"
    URDU = "ur"


class ProblemCategory(Enum):
    """Problem categories."""
    MATH = "math"
    LOGIC = "logic"
    COMMONSENSE = "commonsense"
    SCIENCE = "science"


@dataclass
class ReasoningProblem:
    """A reasoning problem to be solved."""
    id: str
    text: str
    language: Language
    category: ProblemCategory
    difficulty: int  # 1-5 scale
    target_answer: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "language": self.language.value,
            "category": self.category.value,
            "difficulty": self.difficulty,
            "target_answer": self.target_answer,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ReasoningProblem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            language=Language(data["language"]),
            category=ProblemCategory(data["category"]),
            difficulty=data["difficulty"],
            target_answer=data.get("target_answer"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    step_number: int
    strategy: ReasoningStrategy
    text: str
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "strategy": self.strategy.value,
            "text": self.text,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ReasoningStep":
        """Create from dictionary."""
        return cls(
            step_number=data["step_number"],
            strategy=ReasoningStrategy(data["strategy"]),
            text=data["text"],
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ReasoningSolution:
    """A complete solution to a reasoning problem."""
    problem_id: str
    selected_strategy: ReasoningStrategy
    steps: List[ReasoningStep]
    final_answer: str
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def get_reasoning_text(self) -> str:
        """Get the full reasoning chain as text."""
        return "\n".join([
            f"Step {step.step_number}: [{step.strategy.value}] {step.text}"
            for step in self.steps
        ])

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "problem_id": self.problem_id,
            "selected_strategy": self.selected_strategy.value,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ReasoningSolution":
        """Create from dictionary."""
        return cls(
            problem_id=data["problem_id"],
            selected_strategy=ReasoningStrategy(data["selected_strategy"]),
            steps=[ReasoningStep.from_dict(s) for s in data["steps"]],
            final_answer=data["final_answer"],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MetaCognitiveTrace:
    """A complete meta-cognitive reasoning trace."""
    problem: ReasoningProblem
    solution: ReasoningSolution
    is_correct: bool
    rewards: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "problem": self.problem.to_dict(),
            "solution": self.solution.to_dict(),
            "is_correct": self.is_correct,
            "rewards": self.rewards,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MetaCognitiveTrace":
        """Create from dictionary."""
        return cls(
            problem=ReasoningProblem.from_dict(data["problem"]),
            solution=ReasoningSolution.from_dict(data["solution"]),
            is_correct=data["is_correct"],
            rewards=data.get("rewards", {}),
            timestamp=data.get("timestamp"),
        )


@dataclass
class RewardComponents:
    """Multi-component reward breakdown."""
    answer: float  # R_answer: 0.0 or 1.0
    strategy: float  # R_strategy: 0.0 or 1.0
    process: float  # R_process: 0.0 to 1.0 (BERTScore)
    plan: float  # R_plan: 0.0 to 1.0 (consistency)
    
    # Weights for combining rewards
    WEIGHTS = {
        "answer": 0.4,
        "strategy": 0.2,
        "process": 0.3,
        "plan": 0.1,
    }

    def total(self) -> float:
        """Calculate total weighted reward."""
        return (
            self.WEIGHTS["answer"] * self.answer +
            self.WEIGHTS["strategy"] * self.strategy +
            self.WEIGHTS["process"] * self.process +
            self.WEIGHTS["plan"] * self.plan
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "strategy": self.strategy,
            "process": self.process,
            "plan": self.plan,
            "total": self.total(),
        }
