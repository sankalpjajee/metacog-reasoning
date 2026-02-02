#!/usr/bin/env python3
"""
Inspect V4 training dataset structure, statistics, and samples.
"""

import json
import torch
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any


def load_metadata(path: str) -> Dict[str, Any]:
    """Load metadata JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_tensors(path: str) -> Dict[str, torch.Tensor]:
    """Load tensor data."""
    return torch.load(path)


def load_pca(path: str):
    """Load PCA model."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def inspect_benchmark(data_dir: str, benchmark: str):
    """Inspect dataset for a single benchmark."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {benchmark.upper()}")
    print(f"{'='*80}\n")
    
    # Load files
    metadata = load_metadata(f"{data_dir}/{benchmark}_metadata.json")
    tensors = load_tensors(f"{data_dir}/{benchmark}_tensors.pt")
    pca = load_pca(f"{data_dir}/{benchmark}_pca.pkl")
    
    # 1. Dataset Statistics
    print("📊 DATASET STATISTICS")
    print("-" * 80)
    print(f"Total samples: {metadata['num_samples']}")
    print(f"Baseline error rate: {metadata['baseline_error_rate']:.1%}")
    print(f"Utility positive rate: {metadata['utility_positive_rate']:.1%}")
    print(f"Mean utility: {metadata['mean_utility']:.3f}")
    print()
    
    # 2. Feature Dimensions
    print("🔢 FEATURE DIMENSIONS")
    print("-" * 80)
    print(f"Compressed hidden states: {tensors['compressed_hidden'].shape}")
    print(f"Dynamic features: {tensors['dynamic_features'].shape}")
    print(f"Early entropy: {tensors['early_entropy'].shape}")
    print(f"Total feature dims: {tensors['compressed_hidden'].shape[1] + tensors['dynamic_features'].shape[1] + tensors['early_entropy'].shape[1]}")
    print()
    
    # 3. PCA Information
    print("📉 PCA COMPRESSION")
    print("-" * 80)
    print(f"Original dimensions: {pca.n_features_in_}")
    print(f"Compressed dimensions: {pca.n_components_}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"Compression ratio: {pca.n_features_in_ / pca.n_components_:.1f}x")
    print()
    
    # 4. Label Distributions
    print("🎯 LABEL DISTRIBUTIONS")
    print("-" * 80)
    
    # Wrong labels
    wrong_labels = tensors['wrong_label'].numpy()
    print(f"Wrong label distribution:")
    print(f"  Correct (0): {(wrong_labels == 0).sum()} ({(wrong_labels == 0).mean():.1%})")
    print(f"  Wrong (1):   {(wrong_labels == 1).sum()} ({(wrong_labels == 1).mean():.1%})")
    print()
    
    # Conflict labels
    conflict_labels = tensors['conflict_label'].numpy()
    print(f"Conflict label distribution:")
    print(f"  No conflict (0):        {(conflict_labels == 0).sum()} ({(conflict_labels == 0).mean():.1%})")
    print(f"  Baseline wrong only (1): {(conflict_labels == 1).sum()} ({(conflict_labels == 1).mean():.1%})")
    print(f"  Metacog wrong only (2):  {(conflict_labels == 2).sum()} ({(conflict_labels == 2).mean():.1%})")
    print(f"  Both wrong (3):         {(conflict_labels == 3).sum()} ({(conflict_labels == 3).mean():.1%})")
    print()
    
    # Utility labels
    utility_labels = tensors['utility_label'].numpy()
    print(f"Utility label statistics:")
    print(f"  Min:    {utility_labels.min():.3f}")
    print(f"  Max:    {utility_labels.max():.3f}")
    print(f"  Mean:   {utility_labels.mean():.3f}")
    print(f"  Median: {np.median(utility_labels):.3f}")
    print(f"  Std:    {utility_labels.std():.3f}")
    print()
    
    # Utility distribution bins
    print(f"Utility distribution:")
    positive = (utility_labels > 0).sum()
    zero = (utility_labels == 0).sum()
    negative = (utility_labels < 0).sum()
    print(f"  Positive (>0):  {positive} ({positive/len(utility_labels):.1%})")
    print(f"  Zero (=0):      {zero} ({zero/len(utility_labels):.1%})")
    print(f"  Negative (<0):  {negative} ({negative/len(utility_labels):.1%})")
    print()
    
    # 5. Feature Statistics
    print("📈 FEATURE STATISTICS")
    print("-" * 80)
    
    # Compressed hidden states
    hidden = tensors['compressed_hidden'].numpy()
    print(f"Compressed hidden states (first 5 dims):")
    print(f"  Mean: {hidden[:, :5].mean(axis=0)}")
    print(f"  Std:  {hidden[:, :5].std(axis=0)}")
    print()
    
    # Dynamic features
    dynamic = tensors['dynamic_features'].numpy()
    print(f"Dynamic features (all 9 dims):")
    print(f"  Mean: {dynamic.mean(axis=0)}")
    print(f"  Std:  {dynamic.std(axis=0)}")
    print()
    
    # Early entropy
    entropy = tensors['early_entropy'].numpy()
    print(f"Early entropy features:")
    print(f"  Mean entropy: {entropy[:, 0].mean():.3f} ± {entropy[:, 0].std():.3f}")
    print(f"  Max entropy:  {entropy[:, 1].mean():.3f} ± {entropy[:, 1].std():.3f}")
    print()
    
    # 6. Sample Examples
    print("📝 SAMPLE EXAMPLES (first 3)")
    print("-" * 80)
    
    for i in range(min(3, len(metadata['samples']))):
        sample = metadata['samples'][i]
        print(f"\nSample {i+1}:")
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  Ground truth: {sample['ground_truth']}")
        print(f"  Baseline answer: {sample['baseline_answer']}")
        print(f"  Baseline correct: {sample['baseline_correct']}")
        print(f"  Metacog answer: {sample['metacog_answer']}")
        print(f"  Metacog correct: {sample['metacog_correct']}")
        print(f"  Utility: {sample['utility']:.3f}")
        print(f"  Wrong label: {sample['wrong_label']}")
        print(f"  Conflict label: {sample['conflict_label']}")
    
    print()


def main():
    """Main inspection function."""
    data_dir = "data/training/acc_v4"
    benchmarks = ["gsm8k", "mmlu", "hellaswag"]
    
    print("\n" + "="*80)
    print("V4 DATASET INSPECTION")
    print("="*80)
    print(f"\nData directory: {data_dir}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    
    for benchmark in benchmarks:
        try:
            inspect_benchmark(data_dir, benchmark)
        except Exception as e:
            print(f"\n❌ Error inspecting {benchmark}: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✅ Dataset inspection complete!")
    print("\nNext steps:")
    print("1. Train probe: python src/training/train_acc_probe_v4.py --data_dir data/training/acc_v4 --output_dir models/acc_v4")
    print("2. Run ablation: python src/training/ablation_study_v4.py --data_dir data/training/acc_v4 --output_dir results/ablation_v4")
    print("3. Evaluate: python src/evaluation/acc_adaptive_evaluator_v4.py --probe_path models/acc_v4/best_probe.pt")
    print()


if __name__ == "__main__":
    main()
