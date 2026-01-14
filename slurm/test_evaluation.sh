#!/bin/bash
#SBATCH --job-name=test_eval
#SBATCH --partition=v100-32gb-hiprio
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/test_eval_%j.out
#SBATCH --error=logs/test_eval_%j.err

# Quick test evaluation with 10 samples per benchmark

echo "=========================================="
echo "Quick Test Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules
module load cuda/11.8
module load python/3.10

# Set environment variables
export HF_HOME=/work/sjajee/hf_cache
export TRANSFORMERS_CACHE=/work/sjajee/hf_cache
export HF_DATASETS_CACHE=/work/sjajee/hf_cache/datasets

# Navigate to project directory
cd /work/sjajee/metacog-reasoning
source venv/bin/activate

# Verify GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run quick test
echo "Running quick test (10 samples per benchmark)..."
echo ""

python scripts/evaluate_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k,math \
    --max_samples 10 \
    --output_dir data/results/baseline_test \
    --batch_size 4

echo ""
echo "=========================================="
echo "Test completed!"
echo "End Time: $(date)"
echo "=========================================="
