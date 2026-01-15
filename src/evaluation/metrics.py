"""Metrics for evaluating reasoning model performance."""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a benchmark."""
    accuracy: float
    num_samples: int
    num_correct: int
    num_incorrect: int
    per_category_accuracy: Dict[str, float] = None
    per_difficulty_accuracy: Dict[int, float] = None


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    # Handle None or empty
    if answer is None:
        return ""
    if not isinstance(answer, str):
        answer = str(answer)
    
    # Remove whitespace
    answer = answer.strip()
    
    # Remove common prefixes
    prefixes = ["the answer is", "answer:", "final answer:"]
    for prefix in prefixes:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Remove punctuation at the end
    answer = answer.rstrip('.,;:!?')
    
    return answer


def extract_number(text: str) -> float:
    """Extract a number from text."""
    # Handle None
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    
    # Remove commas from numbers
    text = text.replace(',', '')
    
    # Try to find a number
    match = re.search(r'-?\d+\.?\d*', text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    
    return None


def extract_step_number(text: str) -> int:
    """
    Extract step number from text like 'Step 2', '2', 'step 3', etc.
    Used for MR-Ben meta-reasoning benchmark.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    
    text = text.strip().lower()
    
    # Handle N/A
    if 'n/a' in text or 'none' in text or 'no error' in text:
        return -1  # Special value for N/A
    
    # Try to find "step X" pattern
    match = re.search(r'step\s*(\d+)', text)
    if match:
        return int(match.group(1))
    
    # Try to find just a number at the start
    match = re.match(r'^\s*(\d+)', text)
    if match:
        return int(match.group(1))
    
    # Try to find any number in the text
    match = re.search(r'(\d+)', text)
    if match:
        return int(match.group(1))
    
    return None


def is_correct(predicted: str, target: str) -> bool:
    """
    Check if predicted answer matches target answer.
    
    Handles:
    - Exact string matching
    - Numerical equivalence
    - Case-insensitive matching
    - Step number extraction (for MR-Ben)
    """
    # Normalize both answers
    predicted = normalize_answer(predicted)
    target = normalize_answer(target)
    
    # Exact match (case-insensitive)
    if predicted.lower() == target.lower():
        return True
    
    # Try step number extraction (for MR-Ben style answers)
    pred_step = extract_step_number(predicted)
    target_step = extract_step_number(target)
    
    if pred_step is not None and target_step is not None:
        return pred_step == target_step
    
    # Try numerical comparison
    pred_num = extract_number(predicted)
    target_num = extract_number(target)
    
    if pred_num is not None and target_num is not None:
        # Allow small floating point errors
        return abs(pred_num - target_num) < 1e-6
    
    return False


def compute_accuracy(predictions: List[Dict]) -> EvaluationMetrics:
    """
    Compute accuracy metrics from predictions.
    
    Args:
        predictions: List of prediction dictionaries with keys:
            - 'predicted_answer': str
            - 'target_answer': str
            - 'category': str (optional)
            - 'difficulty': int (optional)
    
    Returns:
        EvaluationMetrics object
    """
    num_samples = len(predictions)
    num_correct = 0
    
    # Track per-category and per-difficulty accuracy
    category_correct = {}
    category_total = {}
    difficulty_correct = {}
    difficulty_total = {}
    
    for pred in predictions:
        predicted = pred['predicted_answer']
        target = pred['target_answer']
        category = pred.get('category')
        difficulty = pred.get('difficulty')
        
        # Check correctness
        correct = is_correct(predicted, target)
        if correct:
            num_correct += 1
        
        # Track by category
        if category:
            if category not in category_correct:
                category_correct[category] = 0
                category_total[category] = 0
            category_total[category] += 1
            if correct:
                category_correct[category] += 1
        
        # Track by difficulty
        if difficulty is not None:
            if difficulty not in difficulty_correct:
                difficulty_correct[difficulty] = 0
                difficulty_total[difficulty] = 0
            difficulty_total[difficulty] += 1
            if correct:
                difficulty_correct[difficulty] += 1
    
    # Compute overall accuracy
    accuracy = num_correct / num_samples if num_samples > 0 else 0.0
    
    # Compute per-category accuracy
    per_category_accuracy = None
    if category_total:
        per_category_accuracy = {
            cat: category_correct[cat] / category_total[cat]
            for cat in category_total
        }
    
    # Compute per-difficulty accuracy
    per_difficulty_accuracy = None
    if difficulty_total:
        per_difficulty_accuracy = {
            diff: difficulty_correct[diff] / difficulty_total[diff]
            for diff in difficulty_total
        }
    
    return EvaluationMetrics(
        accuracy=accuracy,
        num_samples=num_samples,
        num_correct=num_correct,
        num_incorrect=num_samples - num_correct,
        per_category_accuracy=per_category_accuracy,
        per_difficulty_accuracy=per_difficulty_accuracy,
    )


def format_metrics(metrics: EvaluationMetrics) -> str:
    """Format metrics as a readable string."""
    lines = [
        f"Accuracy: {metrics.accuracy:.1%} ({metrics.num_correct}/{metrics.num_samples})",
        f"Correct: {metrics.num_correct}",
        f"Incorrect: {metrics.num_incorrect}",
    ]
    
    if metrics.per_category_accuracy:
        lines.append("\nPer-Category Accuracy:")
        for category, acc in sorted(metrics.per_category_accuracy.items()):
            lines.append(f"  {category}: {acc:.1%}")
    
    if metrics.per_difficulty_accuracy:
        lines.append("\nPer-Difficulty Accuracy:")
        for difficulty, acc in sorted(metrics.per_difficulty_accuracy.items()):
            lines.append(f"  Level {difficulty}: {acc:.1%}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics computation
    predictions = [
        {'predicted_answer': '42', 'target_answer': '42', 'category': 'math'},
        {'predicted_answer': '43', 'target_answer': '42', 'category': 'math'},
        {'predicted_answer': 'The answer is 100', 'target_answer': '100', 'category': 'logic'},
        {'predicted_answer': '3.14', 'target_answer': '3.14', 'category': 'math', 'difficulty': 2},
    ]
    
    metrics = compute_accuracy(predictions)
    print(format_metrics(metrics))
