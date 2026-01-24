"""Reasoning strategy taxonomy and definitions."""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict


class ReasoningStrategy(Enum):
    """
    Taxonomy of 8 core reasoning strategies based on cognitive science.
    """
    DECOMPOSITION = "decomposition"
    DEDUCTIVE = "deductive_reasoning"
    INDUCTIVE = "inductive_reasoning"
    CAUSAL = "causal_inference"
    ANALOGICAL = "analogical_reasoning"
    BACKWARD_CHAINING = "backward_chaining"
    PROOF_BY_CONTRADICTION = "proof_by_contradiction"
    HYPOTHESIS_TESTING = "hypothesis_testing"


@dataclass
class StrategyDefinition:
    """Definition and description of a reasoning strategy."""
    strategy: ReasoningStrategy
    name: str
    description: str
    category: str
    examples: List[str]


# Strategy definitions
STRATEGY_DEFINITIONS: Dict[ReasoningStrategy, StrategyDefinition] = {
    ReasoningStrategy.DECOMPOSITION: StrategyDefinition(
        strategy=ReasoningStrategy.DECOMPOSITION,
        name="Decomposition",
        description="Break a complex problem into smaller, independent sub-problems that can be solved separately.",
        category="Decomposition",
        examples=[
            "Calculate the total cost by finding the cost of each item separately",
            "Solve a multi-step word problem by breaking it into individual steps"
        ]
    ),
    ReasoningStrategy.DEDUCTIVE: StrategyDefinition(
        strategy=ReasoningStrategy.DEDUCTIVE,
        name="Deductive Reasoning",
        description="Apply general rules or principles to specific cases to derive conclusions.",
        category="Logical",
        examples=[
            "All mammals have hearts. Dogs are mammals. Therefore, dogs have hearts.",
            "If x > 5 and x < 10, then x must be 6, 7, 8, or 9."
        ]
    ),
    ReasoningStrategy.INDUCTIVE: StrategyDefinition(
        strategy=ReasoningStrategy.INDUCTIVE,
        name="Inductive Reasoning",
        description="Generalize from specific examples or observations to form a general rule or pattern.",
        category="Logical",
        examples=[
            "Observe that 2, 4, 6, 8 are even numbers, conclude the pattern is adding 2",
            "Notice that all observed swans are white, hypothesize all swans are white"
        ]
    ),
    ReasoningStrategy.CAUSAL: StrategyDefinition(
        strategy=ReasoningStrategy.CAUSAL,
        name="Causal Inference",
        description="Identify and reason about cause-and-effect relationships between events or variables.",
        category="Causal",
        examples=[
            "If temperature increases, ice melts (cause → effect)",
            "Determine what caused the car accident by analyzing the sequence of events"
        ]
    ),
    ReasoningStrategy.ANALOGICAL: StrategyDefinition(
        strategy=ReasoningStrategy.ANALOGICAL,
        name="Analogical Reasoning",
        description="Map similarities from a known domain to a new problem domain to find solutions.",
        category="Analogical",
        examples=[
            "Solve a new problem by comparing it to a similar problem you've solved before",
            "Understand electricity flow by analogy to water flow in pipes"
        ]
    ),
    ReasoningStrategy.BACKWARD_CHAINING: StrategyDefinition(
        strategy=ReasoningStrategy.BACKWARD_CHAINING,
        name="Backward Chaining",
        description="Work backward from the goal to identify the required premises or steps.",
        category="Goal-Oriented",
        examples=[
            "To find x, I need to solve for y first, which requires knowing z",
            "To reach the destination, what route should I take?"
        ]
    ),
    ReasoningStrategy.PROOF_BY_CONTRADICTION: StrategyDefinition(
        strategy=ReasoningStrategy.PROOF_BY_CONTRADICTION,
        name="Proof by Contradiction",
        description="Assume the opposite of what you want to prove and show it leads to a logical contradiction.",
        category="Verification",
        examples=[
            "Prove √2 is irrational by assuming it's rational and deriving a contradiction",
            "Show there are infinitely many primes by assuming finitely many and finding a contradiction"
        ]
    ),
    ReasoningStrategy.HYPOTHESIS_TESTING: StrategyDefinition(
        strategy=ReasoningStrategy.HYPOTHESIS_TESTING,
        name="Hypothesis Testing",
        description="Formulate a hypothesis and systematically test it with evidence or experiments.",
        category="Verification",
        examples=[
            "Hypothesize that the answer is 42, then verify by substituting back",
            "Test different values to see which one satisfies the equation"
        ]
    ),
}


def get_strategy_description(strategy: ReasoningStrategy) -> str:
    """Get the description of a reasoning strategy."""
    return STRATEGY_DEFINITIONS[strategy].description


def get_strategy_name(strategy: ReasoningStrategy) -> str:
    """Get the human-readable name of a reasoning strategy."""
    return STRATEGY_DEFINITIONS[strategy].name


def get_all_strategies() -> List[ReasoningStrategy]:
    """Get all available reasoning strategies."""
    return list(ReasoningStrategy)


def get_strategies_by_category(category: str) -> List[ReasoningStrategy]:
    """Get all strategies in a specific category."""
    return [
        s for s, d in STRATEGY_DEFINITIONS.items()
        if d.category == category
    ]
