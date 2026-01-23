#!/usr/bin/env python3
"""
Score metacognitive traces on correctness and metacognitive quality.
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TraceScore:
    """Score for a single trace."""
    correctness: float  # 0.0 or 1.0
    monitoring: float  # 0.0 to 1.0
    control: float  # 0.0 to 1.0
    total: float  # Combined score
    
    def to_dict(self):
        return {
            'correctness': self.correctness,
            'monitoring': self.monitoring,
            'control': self.control,
            'total': self.total
        }


def extract_final_answer(trace: str) -> str:
    """Extract the final answer from a trace."""
    # Look for common answer patterns
    patterns = [
        r'(?:answer|Answer|ANSWER):\s*([^\n]+)',
        r'(?:final answer|Final Answer|FINAL ANSWER):\s*([^\n]+)',
        r'(?:therefore|Therefore|THEREFORE),?\s*(?:the answer is|answer is)?\s*([^\n]+)',
        r'(?:so|So|SO),?\s*(?:the answer is|answer is)?\s*([^\n]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, trace, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Clean up the answer
            answer = re.sub(r'[^\w\s\.\-]', '', answer).strip()
            return answer
    
    # If no pattern found, try to extract last number or word
    numbers = re.findall(r'\b\d+\.?\d*\b', trace)
    if numbers:
        return numbers[-1]
    
    # Return last line as fallback
    lines = [l.strip() for l in trace.split('\n') if l.strip()]
    if lines:
        return lines[-1][:50]  # First 50 chars of last line
    
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    # Remove whitespace
    answer = answer.strip().lower()
    
    # Remove common words
    for word in ['the', 'answer', 'is', 'equals', '=', ':']:
        answer = answer.replace(word, '')
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    return answer


def check_correctness(trace: str, ground_truth: str) -> float:
    """Check if trace produces correct answer."""
    predicted = extract_final_answer(trace)
    
    # Normalize both answers
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    # Check exact match
    if pred_norm == gt_norm:
        return 1.0
    
    # Check if ground truth is contained in prediction
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return 1.0
    
    # Check numerical equivalence
    try:
        pred_num = float(re.sub(r'[^\d\.\-]', '', predicted))
        gt_num = float(re.sub(r'[^\d\.\-]', '', ground_truth))
        if abs(pred_num - gt_num) < 0.01:  # Allow small floating point errors
            return 1.0
    except (ValueError, AttributeError):
        pass
    
    return 0.0


def assess_monitoring(trace: str) -> float:
    """Assess monitoring quality (confidence, uncertainty awareness)."""
    trace_lower = trace.lower()
    
    score = 0.0
    max_score = 4.0
    
    # 1. Explicit confidence statements (0.0-1.0)
    confidence_markers = [
        'confidence:', 'confident', 'certain', 'sure',
        'uncertain', 'unsure', 'not sure', 'maybe', 'might'
    ]
    has_confidence = any(marker in trace_lower for marker in confidence_markers)
    if has_confidence:
        score += 1.0
    
    # 2. Calibrated confidence (0.0-1.0)
    # Check if confidence aligns with correctness
    # (This is a simplified heuristic)
    if 'confident' in trace_lower or 'certain' in trace_lower:
        score += 0.5  # Bonus for expressing confidence
    if 'uncertain' in trace_lower or 'not sure' in trace_lower:
        score += 0.5  # Bonus for expressing uncertainty
    
    # 3. Self-questioning (0.0-1.0)
    question_markers = [
        'is this correct?', 'am i right?', 'does this make sense?',
        'wait,', 'hmm,', 'let me think', 'is this right?'
    ]
    has_questions = any(marker in trace_lower for marker in question_markers)
    if has_questions:
        score += 1.0
    
    # 4. Awareness of difficulty (0.0-1.0)
    difficulty_markers = [
        'this is tricky', 'this is complex', 'this is difficult',
        'this is easy', 'straightforward', 'simple'
    ]
    has_difficulty_awareness = any(marker in trace_lower for marker in difficulty_markers)
    if has_difficulty_awareness:
        score += 1.0
    
    return score / max_score


def assess_control(trace: str) -> float:
    """Assess control quality (verification, error correction, strategy adaptation)."""
    trace_lower = trace.lower()
    
    score = 0.0
    max_score = 5.0
    
    # 1. Explicit verification (0.0-1.5)
    verification_markers = [
        'verify:', 'check:', 'verification:', 'let me check',
        'let me verify', 'double-check', 'checking'
    ]
    has_verification = any(marker in trace_lower for marker in verification_markers)
    if has_verification:
        score += 1.5
    
    # 2. Error detection (0.0-1.5)
    error_markers = [
        'wait, that\'s wrong', 'that doesn\'t seem right',
        'i made a mistake', 'error', 'incorrect', 'that\'s not right'
    ]
    has_error_detection = any(marker in trace_lower for marker in error_markers)
    if has_error_detection:
        score += 1.5
    
    # 3. Self-correction (0.0-1.0)
    correction_markers = [
        'let me correct', 'actually,', 'correction:', 'revised:',
        'let me redo', 'let me recalculate'
    ]
    has_correction = any(marker in trace_lower for marker in correction_markers)
    if has_correction:
        score += 1.0
    
    # 4. Strategy adaptation (0.0-1.0)
    strategy_markers = [
        'different approach', 'another way', 'alternative method',
        'let me try', 'instead,', 'better way'
    ]
    has_strategy = any(marker in trace_lower for marker in strategy_markers)
    if has_strategy:
        score += 1.0
    
    return score / max_score


def score_trace(trace: str, ground_truth: str, problem_difficulty: str = "medium") -> TraceScore:
    """Score a single trace."""
    # 1. Correctness (most important)
    correctness = check_correctness(trace, ground_truth)
    
    # 2. Monitoring quality
    monitoring = assess_monitoring(trace)
    
    # 3. Control quality
    control = assess_control(trace)
    
    # 4. Calculate total score
    if problem_difficulty == "hard":
        # For hard problems, metacognition is more valuable
        total = correctness + 0.3 * monitoring + 0.3 * control
    else:
        # For easy/medium problems, correctness is enough
        total = correctness + 0.1 * monitoring + 0.1 * control
    
    return TraceScore(
        correctness=correctness,
        monitoring=monitoring,
        control=control,
        total=total
    )


def score_all_traces(traces_file: Path, output_file: Path):
    """Score all traces in a file."""
    print(f"Loading traces from: {traces_file}")
    
    with open(traces_file, 'r') as f:
        all_traces = json.load(f)
    
    print(f"Loaded {len(all_traces)} problems")
    
    # Score each trace
    for problem_data in all_traces:
        problem_id = problem_data['problem_id']
        ground_truth = problem_data['ground_truth_answer']
        difficulty = problem_data.get('difficulty', 'medium')
        
        print(f"\nScoring problem {problem_id} ({difficulty})...")
        
        for trace_data in problem_data['traces']:
            trace = trace_data['trace']
            
            # Score the trace
            score = score_trace(trace, ground_truth, difficulty)
            
            # Add scores to trace data
            trace_data['scores'] = score.to_dict()
            trace_data['predicted_answer'] = extract_final_answer(trace)
            
            print(f"  Trace {trace_data['trace_id']} ({trace_data['trace_type']}): "
                  f"correct={score.correctness:.1f}, "
                  f"monitoring={score.monitoring:.2f}, "
                  f"control={score.control:.2f}, "
                  f"total={score.total:.2f}")
    
    # Save scored traces
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_traces, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Scored traces saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print summary statistics
    print_summary_stats(all_traces)


def print_summary_stats(all_traces: List[Dict]):
    """Print summary statistics."""
    total_traces = sum(len(p['traces']) for p in all_traces)
    correct_traces = sum(
        1 for p in all_traces 
        for t in p['traces'] 
        if t['scores']['correctness'] == 1.0
    )
    
    avg_monitoring = sum(
        t['scores']['monitoring'] 
        for p in all_traces 
        for t in p['traces']
    ) / total_traces
    
    avg_control = sum(
        t['scores']['control'] 
        for p in all_traces 
        for t in p['traces']
    ) / total_traces
    
    print(f"\nSummary Statistics:")
    print(f"- Total traces: {total_traces}")
    print(f"- Correct traces: {correct_traces}/{total_traces} ({correct_traces/total_traces*100:.1f}%)")
    print(f"- Average monitoring score: {avg_monitoring:.2f}")
    print(f"- Average control score: {avg_control:.2f}")
    
    # Breakdown by trace type
    print(f"\nBreakdown by trace type:")
    trace_types = {}
    for p in all_traces:
        for t in p['traces']:
            ttype = t['trace_type']
            if ttype not in trace_types:
                trace_types[ttype] = {'correct': 0, 'total': 0, 'monitoring': [], 'control': []}
            
            trace_types[ttype]['total'] += 1
            if t['scores']['correctness'] == 1.0:
                trace_types[ttype]['correct'] += 1
            trace_types[ttype]['monitoring'].append(t['scores']['monitoring'])
            trace_types[ttype]['control'].append(t['scores']['control'])
    
    for ttype, stats in sorted(trace_types.items()):
        acc = stats['correct'] / stats['total'] * 100
        avg_mon = sum(stats['monitoring']) / len(stats['monitoring'])
        avg_ctrl = sum(stats['control']) / len(stats['control'])
        print(f"  {ttype}: accuracy={acc:.1f}%, monitoring={avg_mon:.2f}, control={avg_ctrl:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Score metacognitive traces")
    parser.add_argument("--traces_file", type=str, required=True, help="Path to traces JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output scored traces file")
    args = parser.parse_args()
    
    traces_file = Path(args.traces_file)
    output_file = Path(args.output_file)
    
    if not traces_file.exists():
        print(f"Error: Traces file not found: {traces_file}")
        return
    
    score_all_traces(traces_file, output_file)


if __name__ == "__main__":
    main()
