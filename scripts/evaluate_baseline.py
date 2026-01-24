#!/usr/bin/env python3
"""Evaluate baseline Llama-3.1-8B-Instruct model on benchmarks."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluation.evaluator import ModelEvaluator

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate baseline Llama-3.1-8B-Instruct model")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model path or HuggingFace name"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="gsm8k,mmlu,hellaswag,humaneval,mrben",
        help="Comma-separated benchmark names"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/baseline",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per benchmark (for testing)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)"
    )
    parser.add_argument(
        "--use_mlflow",
        action="store_true",
        default=True,
        help="Use MLflow for experiment tracking"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("BASELINE MODEL EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples} (testing mode)")
    print("="*60 + "\n")
    
    # Parse benchmarks
    benchmarks = [b.strip() for b in args.benchmarks.split(',')]
    
    # Setup MLflow if requested
    if args.use_mlflow and MLFLOW_AVAILABLE:
        mlflow_dir = os.path.expanduser("~/metacog-reasoning/mlruns")
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        
        # Create or get experiment
        experiment_name = "metacog-reasoning-baseline"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        print(f"✓ MLflow tracking enabled: {experiment_name}\n")
    elif args.use_mlflow and not MLFLOW_AVAILABLE:
        print("⚠ MLflow not available, disabling experiment tracking\n")
        args.use_mlflow = False
    
    # Create evaluator (it will load the model internally)
    evaluator = ModelEvaluator(
        model_path=args.model,
        device=args.device,
        use_mlflow=args.use_mlflow,
    )
    
    # Run evaluation on each benchmark
    all_results = {}
    for benchmark in benchmarks:
        print(f"\n{'='*60}")
        print(f"Starting evaluation on {benchmark.upper()}")
        print(f"{'='*60}\n")
        
        try:
            results = evaluator.evaluate_benchmark(
                benchmark_name=benchmark,
                split="test",
                output_dir=args.output_dir,
                max_samples=args.max_samples,
            )
            all_results[benchmark] = results
            
        except Exception as e:
            print(f"\n✗ Error evaluating {benchmark}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE EVALUATION COMPLETE")
    print("="*60)
    print("\nSummary:")
    for benchmark, results in all_results.items():
        accuracy = results['accuracy']
        num_samples = results['num_samples']
        num_correct = results['num_correct']
        print(f"  {benchmark:12s}: {accuracy:.1%} ({num_correct}/{num_samples})")
    print("="*60)


if __name__ == "__main__":
    main()
