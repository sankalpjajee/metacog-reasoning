#!/bin/bash
#SBATCH --job-name=download_benchmarks
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/download_benchmarks_%j.out
#SBATCH --error=logs/download_benchmarks_%j.err

# Download all benchmark datasets
# This job doesn't need GPU, just CPU and internet access

echo "=========================================="
echo "Downloading Benchmark Datasets"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules
module load python/3.10

# Set environment variables
export HF_HOME=/work/sjajee/hf_cache
export HF_DATASETS_CACHE=/work/sjajee/hf_cache/datasets

# Navigate to project directory
cd /work/sjajee/metacog-reasoning
source venv/bin/activate

# Create directories
mkdir -p data/benchmarks/{gsm8k,math,mmlu,hellaswag,bigbench,humaneval}
mkdir -p logs

# Download benchmarks
echo ""
echo "Downloading benchmarks..."
bash scripts/download_benchmarks.sh

# Check results
echo ""
echo "=========================================="
echo "Download Summary:"
echo "=========================================="
du -sh data/benchmarks/*
echo ""
echo "Total size:"
du -sh data/benchmarks/
echo ""
echo "End Time: $(date)"
echo "=========================================="
