#!/usr/bin/env python3
"""Compare evaluation results from multiple models."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.comparator import ModelComparator


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare model evaluation results")
    parser.add_argument(
        "--result_dirs",
        type=str,
        default="data/results/baseline,data/results/standard_selfplay,data/results/metacog_selfplay",
        help="Comma-separated list of result directories"
    )
    parser.add_argument(
        "--model_names",
        type=str,
        default="Baseline,Standard Self-Play,Meta-Cognitive Self-Play",
        help="Comma-separated list of model names"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="Baseline",
        help="Baseline model name for computing improvements"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/results/comparison",
        help="Output directory for comparison files"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Parse arguments
    result_dirs = [d.strip() for d in args.result_dirs.split(',')]
    model_names = [n.strip() for n in args.model_names.split(',')]
    
    print(f"Models: {', '.join(model_names)}")
    print(f"Baseline: {args.baseline}")
    print(f"Output: {args.output_dir}")
    print("="*60 + "\n")
    
    # Create comparator
    comparator = ModelComparator(result_dirs, model_names)
    
    # Save comparison
    comparator.save_comparison(args.output_dir, args.baseline)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
