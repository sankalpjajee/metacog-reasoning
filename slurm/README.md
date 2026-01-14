# SLURM Job Scripts

This directory contains SLURM batch scripts for running jobs on the HPC cluster.

## Quick Reference

### Initial Setup (One-Time)

```bash
# 1. Clone repository
cd /work/sjajee
git clone https://github.com/sankalpjajee/metacog-reasoning.git
cd metacog-reasoning

# 2. Run setup script
bash slurm/setup_cluster.sh
```

### Download Benchmarks

```bash
# Option 1: Submit as batch job (recommended)
sbatch slurm/download_benchmarks.sh

# Option 2: Run interactively (if you have allocation)
bash scripts/download_benchmarks.sh
```

### Run Evaluations

```bash
# Quick test (10 samples, ~10 minutes)
sbatch slurm/test_evaluation.sh

# Full baseline evaluation (all benchmarks, ~8 hours)
sbatch slurm/evaluate_baseline.sh
```

### Monitor Jobs

```bash
# Check job status
squeue -u sjajee

# Check specific job
squeue -j JOBID

# View job details
scontrol show job JOBID

# Cancel job
scancel JOBID

# View completed job info
sacct -j JOBID --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,MaxVMSize
```

### View Logs

```bash
# Logs are saved to logs/ directory
ls -lh logs/

# View output
cat logs/eval_baseline_JOBID.out

# View errors
cat logs/eval_baseline_JOBID.err

# Follow log in real-time
tail -f logs/eval_baseline_JOBID.out
```

## Available Scripts

### `setup_cluster.sh`
**Purpose:** One-time setup of environment  
**Usage:** `bash slurm/setup_cluster.sh`  
**Requirements:** None  
**Time:** ~10 minutes

Sets up:
- Virtual environment
- PyTorch with CUDA
- All dependencies
- Cache directories

### `download_benchmarks.sh`
**Purpose:** Download all benchmark datasets  
**Usage:** `sbatch slurm/download_benchmarks.sh`  
**Requirements:** Internet access  
**Time:** ~1 hour  
**Resources:** CPU only, 16GB RAM

Downloads:
- GSM8k (1,319 samples)
- MATH (5,000 samples)
- MMLU (14,042 samples)
- HellaSwag (10,042 samples)
- BIG-Bench (varies)
- HumanEval (164 samples)

### `test_evaluation.sh`
**Purpose:** Quick test with 10 samples  
**Usage:** `sbatch slurm/test_evaluation.sh`  
**Requirements:** 1× V100 GPU  
**Time:** ~10 minutes  
**Resources:** 1 GPU, 32GB RAM

Tests:
- Environment setup
- Model loading
- Inference pipeline
- Result saving

### `evaluate_baseline.sh`
**Purpose:** Full baseline evaluation  
**Usage:** `sbatch slurm/evaluate_baseline.sh`  
**Requirements:** 1× V100 GPU  
**Time:** ~8 hours  
**Resources:** 1 GPU, 64GB RAM

Evaluates:
- All 6 benchmarks
- ~30K test samples
- Saves detailed results

## Customizing Scripts

### Change GPU Partition

Edit the `#SBATCH --partition` line:
```bash
#SBATCH --partition=v100-32gb-hiprio  # V100 high priority
#SBATCH --partition=a100-80gb         # A100 (if available)
#SBATCH --partition=gpu               # Any GPU
```

### Change Time Limit

Edit the `#SBATCH --time` line:
```bash
#SBATCH --time=08:00:00  # 8 hours
#SBATCH --time=24:00:00  # 24 hours
#SBATCH --time=3-00:00:00  # 3 days
```

### Change Email Notifications

Edit the `#SBATCH --mail-user` line:
```bash
#SBATCH --mail-user=your.email@sc.edu
#SBATCH --mail-type=END,FAIL  # Notify on completion or failure
#SBATCH --mail-type=ALL       # Notify on all events
```

### Change Benchmarks

Edit the `--benchmarks` argument:
```bash
# Single benchmark
--benchmarks gsm8k

# Subset
--benchmarks gsm8k,math,hellaswag

# All (default)
--benchmarks gsm8k,math,mmlu,hellaswag,bigbench,humaneval
```

### Reduce Memory Usage

Enable 8-bit quantization in the evaluation script:
```python
# In src/evaluation/evaluator.py
self.model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,  # Add this line
)
```

## Troubleshooting

### Job Pending for Long Time

```bash
# Check queue
squeue -p v100-32gb-hiprio

# Check your priority
sprio -u sjajee

# Try different partition
#SBATCH --partition=gpu
```

### Out of Memory

```bash
# Reduce batch size
--batch_size 4  # or 2

# Or use 8-bit quantization (see above)
```

### Module Not Found

```bash
# Load modules in your script
module load cuda/11.8
module load python/3.10

# Activate venv
source venv/bin/activate
```

### Disk Quota Exceeded

```bash
# Check usage
du -sh /work/sjajee/*

# Clean cache
rm -rf /work/sjajee/hf_cache/hub/*.lock
```

## Best Practices

1. **Test first:** Always run `test_evaluation.sh` before full evaluation
2. **Use batch jobs:** Don't tie up interactive sessions for long runs
3. **Monitor resources:** Check `nvidia-smi` and memory usage
4. **Clean up:** Remove old logs and checkpoints regularly
5. **Use /work:** Never use /home for large files
6. **Set email:** Get notified when jobs complete

## Example Workflow

```bash
# 1. Setup (one-time)
bash slurm/setup_cluster.sh

# 2. Download data (one-time)
sbatch slurm/download_benchmarks.sh

# 3. Test
sbatch slurm/test_evaluation.sh
# Wait for completion, check logs/test_eval_*.out

# 4. Full evaluation
sbatch slurm/evaluate_baseline.sh
# Wait ~8 hours, check logs/eval_baseline_*.out

# 5. View results
cat data/results/baseline/summary.json
```

## Getting Help

- **Cluster docs:** https://rci.sc.edu/
- **SLURM manual:** `man sbatch`, `man squeue`
- **Project issues:** https://github.com/sankalpjajee/metacog-reasoning/issues
