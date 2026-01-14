#!/usr/bin/env python3
"""Evaluate baseline Llama-3.1-8B model on benchmarks."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.evaluator import ModelEvaluator


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate baseline Llama-3.1-8B model")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model path or HuggingFace name"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="gsm8k,math,mmlu,hellaswag,bigbench,humaneval",
        help="Comma-separated benchmark names"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/results/baseline",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per benchmark (for testing)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("BASELINE MODEL EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Output: {args.output_dir}")
    print("="*60 + "\n")
    
    # Parse benchmarks
    benchmarks = [b.strip() for b in args.benchmarks.split(',')]
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        batch_size=args.batch_size,
    )
    
    # Run evaluation
    for benchmark in benchmarks:
        evaluator.evaluate_benchmark(
            benchmark_name=benchmark,
            split="test",
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )
    
    print("\n" + "="*60)
    print("BASELINE EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
