"""Prompt templates for the teacher model (Phase 1)."""

from typing import Dict
from ..strategies import ReasoningStrategy, get_all_strategies, get_strategy_description


def get_strategy_list_text() -> str:
    """Get formatted list of all strategies for prompts."""
    strategies = get_all_strategies()
    return "\n".join([
        f"{i+1}. {strategy.value}: {get_strategy_description(strategy)}"
        for i, strategy in enumerate(strategies)
    ])


# Phase 1: Problem Generation
PROBLEM_GENERATION_PROMPT = """You are an expert problem designer. Generate a challenging reasoning problem that requires multi-step reasoning to solve.

The problem should:
- Be clear and well-defined
- Require at least 3 steps to solve
- Be solvable using one of the reasoning strategies from the taxonomy
- Be at difficulty level {difficulty} (1=easy, 5=very hard)
- Be in the category: {category}

Generate only the problem text, without the solution.

Problem:"""

# Phase 2: Strategy Selection
STRATEGY_SELECTION_PROMPT = """Analyze the following problem and select the most appropriate reasoning strategy to solve it.

Problem: {problem_text}

Available Strategies:
{strategy_list}

Think carefully about which strategy would be most effective for this problem. Respond with only the strategy name (e.g., "backward_chaining").

Selected Strategy:"""

# Phase 3: Annotated Solution Generation
ANNOTATED_SOLUTION_PROMPT = """Solve the following problem using the selected reasoning strategy. Annotate each step with the cognitive primitive being applied.

Problem: {problem_text}

Selected Strategy: {strategy}

Generate a step-by-step solution where each step is formatted as:
Step N: [Cognitive Primitive] Description of what you're doing

Example format:
Step 1: [Identify Goal] We need to find the value of x.
Step 2: [Backward Chaining] To find x, we need to isolate it in the equation.
Step 3: [Apply Inverse Operation] Subtract 5 from both sides: x = 12 - 5
Step 4: [Calculate] x = 7
Step 5: [Verification] Check: 7 + 5 = 12 ✓

Final Answer: 7

Now solve the given problem:

Solution:"""

# Phase 4: Self-Evaluation
SELF_EVALUATION_PROMPT = """Evaluate the following solution to determine if it is correct and well-reasoned.

Problem: {problem_text}

Solution:
{solution_text}

Final Answer: {final_answer}

Evaluate the solution on these criteria:
1. Is the final answer correct?
2. Is the reasoning logically sound?
3. Does the solution follow the stated strategy?
4. Are all steps clearly explained?

Respond in this format:
Correct: [Yes/No]
Reasoning Quality: [1-5]
Strategy Adherence: [1-5]
Explanation: [Brief explanation of your evaluation]

Evaluation:"""

# Phase 5: Self-Correction
SELF_CORRECTION_PROMPT = """The previous solution was incorrect or incomplete. Identify the error and provide a corrected solution.

Problem: {problem_text}

Previous Solution:
{previous_solution}

Previous Answer: {previous_answer}

Error Identified: {error_description}

Provide a corrected solution following the same format:

Corrected Solution:"""


def format_problem_generation_prompt(difficulty: int, category: str) -> str:
    """Format the problem generation prompt."""
    return PROBLEM_GENERATION_PROMPT.format(
        difficulty=difficulty,
        category=category
    )


def format_strategy_selection_prompt(problem_text: str) -> str:
    """Format the strategy selection prompt."""
    return STRATEGY_SELECTION_PROMPT.format(
        problem_text=problem_text,
        strategy_list=get_strategy_list_text()
    )


def format_annotated_solution_prompt(problem_text: str, strategy: ReasoningStrategy) -> str:
    """Format the annotated solution generation prompt."""
    return ANNOTATED_SOLUTION_PROMPT.format(
        problem_text=problem_text,
        strategy=strategy.value
    )


def format_self_evaluation_prompt(problem_text: str, solution_text: str, final_answer: str) -> str:
    """Format the self-evaluation prompt."""
    return SELF_EVALUATION_PROMPT.format(
        problem_text=problem_text,
        solution_text=solution_text,
        final_answer=final_answer
    )


def format_self_correction_prompt(
    problem_text: str,
    previous_solution: str,
    previous_answer: str,
    error_description: str
) -> str:
    """Format the self-correction prompt."""
    return SELF_CORRECTION_PROMPT.format(
        problem_text=problem_text,
        previous_solution=previous_solution,
        previous_answer=previous_answer,
        error_description=error_description
    )
