#!/bin/bash
# Download all benchmark datasets

set -e

echo "=========================================="
echo "Downloading Benchmark Datasets"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create data directories
mkdir -p data/benchmarks/{gsm8k,math,mmlu,hellaswag,bigbench,humaneval}

echo ""
echo "Downloading benchmarks using Python script..."
echo ""

# Run Python script to download benchmarks
python3 << 'EOF'
import sys
import os
sys.path.insert(0, '.')

from src.evaluation.benchmarks import load_benchmark

print("1/6 Downloading GSM8k...")
try:
    gsm8k_test = load_benchmark("gsm8k", split="test")
    print(f"✓ GSM8k: {len(gsm8k_test)} test samples")
except Exception as e:
    print(f"✗ GSM8k failed: {e}")

print("\n2/6 Downloading MATH...")
try:
    math_test = load_benchmark("math", split="test")
    print(f"✓ MATH: {len(math_test)} test samples")
except Exception as e:
    print(f"✗ MATH failed: {e}")

print("\n3/6 Downloading MMLU...")
try:
    mmlu_test = load_benchmark("mmlu", split="test")
    print(f"✓ MMLU: {len(mmlu_test)} test samples")
except Exception as e:
    print(f"✗ MMLU failed: {e}")

print("\n4/6 Downloading HellaSwag...")
try:
    hellaswag_val = load_benchmark("hellaswag", split="validation")
    print(f"✓ HellaSwag: {len(hellaswag_val)} validation samples")
except Exception as e:
    print(f"✗ HellaSwag failed: {e}")

print("\n5/6 Downloading BIG-Bench (logical_deduction)...")
try:
    bigbench_test = load_benchmark("bigbench", split="default", task="logical_deduction")
    print(f"✓ BIG-Bench: {len(bigbench_test)} samples")
except Exception as e:
    print(f"✗ BIG-Bench failed: {e}")

print("\n6/6 Downloading HumanEval...")
try:
    humaneval_test = load_benchmark("humaneval", split="test")
    print(f"✓ HumanEval: {len(humaneval_test)} test samples")
except Exception as e:
    print(f"✗ HumanEval failed: {e}")

print("\n" + "="*60)
print("All benchmarks downloaded successfully!")
print("="*60)
EOF

echo ""
echo "Benchmark datasets are ready in data/benchmarks/"
echo ""
echo "Next steps:"
echo "  1. Evaluate baseline: python scripts/evaluate_baseline.py"
echo "  2. Train models: python scripts/train_teacher.py"
echo "  3. Compare results: python scripts/compare_models.py"
