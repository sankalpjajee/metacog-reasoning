#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH --partition=v100-32gb-hiprio
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/eval_baseline_%j.out
#SBATCH --error=logs/eval_baseline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sjajee@email.sc.edu

# Job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules
module load cuda/11.8
module load python/3.10
module load gcc/9.3.0

# Set environment variables
export HF_HOME=/work/sjajee/hf_cache
export TRANSFORMERS_CACHE=/work/sjajee/hf_cache
export HF_DATASETS_CACHE=/work/sjajee/hf_cache/datasets
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory
cd /work/sjajee/metacog-reasoning

# Activate virtual environment
source venv/bin/activate

# Verify GPU
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Run evaluation
echo "Starting baseline evaluation..."
echo ""

python scripts/evaluate_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k,math,mmlu,hellaswag,bigbench,humaneval \
    --output_dir data/results/baseline \
    --batch_size 8

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed!"
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi
