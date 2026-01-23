#!/usr/bin/env python3
"""
Pilot test for enhanced metacognitive prompting.
Tests on 10 samples from GSM8K to validate the new prompt doesn't degrade performance.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
from datetime import datetime
from src.evaluation.metacognitive_evaluator import MetacognitiveEvaluator
from src.evaluation.benchmarks import load_benchmark
from src.evaluation.metrics import compute_accuracy

def main():
    print("=" * 80)
    print("PILOT TEST: Enhanced Metacognitive Prompting")
    print("=" * 80)
    print()
    print("Testing Wang & Zhao (2024) enhanced with Nelson & Narens (1990) framework")
    print("Sample size: 10 from GSM8K")
    print()
    
    # Load evaluator
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    evaluator = MetacognitiveEvaluator(
        model_path=model_path,
        use_mlflow=False,  # Don't log pilot tests
    )
    
    # Load 10 samples from GSM8K
    print("Loading GSM8K samples...")
    samples = load_benchmark("gsm8k", split="test", max_samples=10)
    print(f"Loaded {len(samples)} samples")
    print()
    
    # Evaluate
    results = []
    correct = 0
    
    print("Evaluating samples...")
    print("-" * 80)
    
    for i, sample in enumerate(samples, 1):
        print(f"\n[{i}/10] Problem: {sample.question[:80]}...")
        
        # Generate answer
        full_response, predicted_answer = evaluator.generate_answer(
            sample.question,
            benchmark_name="gsm8k"
        )
        
        # Check correctness
        is_correct = predicted_answer.strip() == str(sample.answer).strip()
        if is_correct:
            correct += 1
        
        print(f"Predicted: {predicted_answer}")
        print(f"Expected: {sample.answer}")
        print(f"Correct: {'✓' if is_correct else '✗'}")
        
        # Save result
        results.append({
            "problem_id": i,
            "question": sample.question,
            "expected_answer": sample.answer,
            "predicted_answer": predicted_answer,
            "full_response": full_response,
            "correct": is_correct,
        })
    
    # Compute accuracy
    accuracy = correct / len(samples)
    
    print()
    print("=" * 80)
    print("PILOT TEST RESULTS")
    print("=" * 80)
    print(f"Samples evaluated: {len(samples)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1%}")
    print()
    
    # Compare with baseline
    print("COMPARISON WITH BASELINE:")
    print(f"Baseline GSM8K accuracy: 81.1%")
    print(f"Enhanced prompt accuracy: {accuracy:.1%}")
    
    if accuracy >= 0.70:  # At least 70% on 10 samples
        print()
        print("✅ PILOT TEST PASSED!")
        print("The enhanced prompt does not appear to degrade performance.")
        print("Recommend proceeding with full evaluation (1000 samples).")
    else:
        print()
        print("⚠️ PILOT TEST FAILED!")
        print("The enhanced prompt may be causing issues.")
        print("Review the full responses before proceeding.")
    
    # Save results
    output_dir = "results/pilot_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/enhanced_prompt_pilot_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "model": model_path,
                "benchmark": "gsm8k",
                "num_samples": len(samples),
                "timestamp": timestamp,
                "prompt_version": "enhanced_wang_zhao_nelson_narens",
            },
            "summary": {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(samples),
            },
            "results": results,
        }, f, indent=2)
    
    print()
    print(f"Results saved to: {output_file}")
    print()
    
    # Print a sample response for inspection
    print("=" * 80)
    print("SAMPLE RESPONSE (Problem 1)")
    print("=" * 80)
    print(f"Question: {results[0]['question']}")
    print()
    print("Full Response:")
    print(results[0]['full_response'])
    print()
    print(f"Extracted Answer: {results[0]['predicted_answer']}")
    print(f"Expected Answer: {results[0]['expected_answer']}")
    print(f"Correct: {'✓' if results[0]['correct'] else '✗'}")
    print()

if __name__ == "__main__":
    main()
