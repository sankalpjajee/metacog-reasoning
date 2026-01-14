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
mkdir -p data/benchmarks/{gsm8k,math,mmlu}

echo ""
echo "Downloading benchmarks using Python script..."
echo ""

# Run Python script to download benchmarks
python3 << 'EOF'
import sys
import os
sys.path.insert(0, '.')

from src.evaluation.benchmarks import load_benchmark

print("1/3 Downloading GSM8k...")
try:
    gsm8k_test = load_benchmark("gsm8k", split="test")
    print(f"✓ GSM8k: {len(gsm8k_test)} test samples")
except Exception as e:
    print(f"✗ GSM8k failed: {e}")

print("\n2/3 Downloading MATH...")
try:
    math_test = load_benchmark("math", split="test")
    print(f"✓ MATH: {len(math_test)} test samples")
except Exception as e:
    print(f"✗ MATH failed: {e}")

print("\n3/3 Downloading MMLU...")
try:
    mmlu_test = load_benchmark("mmlu", split="test")
    print(f"✓ MMLU: {len(mmlu_test)} test samples")
except Exception as e:
    print(f"✗ MMLU failed: {e}")

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
