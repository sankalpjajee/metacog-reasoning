# Cluster Setup Guide

This guide explains how to set up and run the metacog-reasoning project on an HPC cluster with SLURM job scheduler.

## Prerequisites

- Access to HPC cluster with SLURM
- GPU nodes available (V100, A100, or similar)
- Module system for loading CUDA, Python, etc.
- Internet access from compute nodes (or pre-downloaded models)

## Quick Start (Once GPU is Allocated)

```bash
# 1. Clone the repository
cd /work/sjajee  # or your work directory
git clone https://github.com/sankalpjajee/metacog-reasoning.git
cd metacog-reasoning

# 2. Load required modules
module load cuda/11.8  # or available CUDA version
module load python/3.10  # or available Python 3.10+

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Set cache directories (important for cluster)
export HF_HOME=/work/sjajee/hf_cache
export TRANSFORMERS_CACHE=/work/sjajee/hf_cache
export HF_DATASETS_CACHE=/work/sjajee/hf_cache/datasets

# 6. Download benchmarks (one-time)
bash scripts/download_benchmarks.sh

# 7. Run evaluation
python scripts/evaluate_baseline.py --max_samples 10  # test with 10 samples first
```

## Detailed Setup Instructions

### Step 1: Get GPU Allocation

You're already doing this! Once you see:
```
salloc: Granted job allocation 20531633
```

You'll have an interactive session with GPU access.

### Step 2: Navigate to Work Directory

```bash
cd /work/sjajee  # Use /work, not /home, for better performance and storage
```

### Step 3: Clone Repository

```bash
git clone https://github.com/sankalpjajee/metacog-reasoning.git
cd metacog-reasoning
```

### Step 4: Load Modules

Check available modules:
```bash
module avail cuda
module avail python
```

Load appropriate versions:
```bash
module load cuda/11.8  # or cuda/12.1, whatever is available
module load python/3.10  # or python/3.11
module load gcc/9.3.0  # may be needed for some packages
```

Verify GPU access:
```bash
nvidia-smi
```

### Step 5: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Step 6: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (check your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Step 7: Configure Cache Directories

**IMPORTANT:** HuggingFace will download large models. Set cache to /work, not /home:

```bash
# Add to ~/.bashrc for persistence
echo 'export HF_HOME=/work/sjajee/hf_cache' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/work/sjajee/hf_cache' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/work/sjajee/hf_cache/datasets' >> ~/.bashrc

# Apply immediately
export HF_HOME=/work/sjajee/hf_cache
export TRANSFORMERS_CACHE=/work/sjajee/hf_cache
export HF_DATASETS_CACHE=/work/sjajee/hf_cache/datasets

# Create directories
mkdir -p /work/sjajee/hf_cache/datasets
```

### Step 8: Download Benchmarks

```bash
bash scripts/download_benchmarks.sh
```

This will download ~5GB of benchmark data to `data/benchmarks/`.

### Step 9: Test Installation

Quick test with 10 samples:
```bash
python scripts/evaluate_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k \
    --max_samples 10 \
    --output_dir data/results/baseline_test
```

## Running Evaluations

### Interactive Session (Current Setup)

You're in an interactive session, so you can run directly:

```bash
# Full evaluation on one benchmark
python scripts/evaluate_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k \
    --output_dir data/results/baseline

# Multiple benchmarks (will take 2-3 hours)
python scripts/evaluate_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k,math,mmlu \
    --output_dir data/results/baseline
```

### Batch Job (Recommended for Long Runs)

For longer evaluations, submit a batch job instead:

```bash
sbatch slurm/evaluate_baseline.sh
```

See the SLURM scripts section below.

## SLURM Job Scripts

### Check Job Status

```bash
squeue -u sjajee  # Check your jobs
sacct -j JOBID    # Check specific job
scancel JOBID     # Cancel a job
```

### Monitor GPU Usage

```bash
watch -n 1 nvidia-smi  # Real-time GPU monitoring
```

## Common Issues and Solutions

### Issue 1: Out of Memory (OOM)

**Solution:** Reduce batch size or use 8-bit quantization:

```python
# In evaluator.py, change:
load_in_8bit=True  # Enable 8-bit quantization
```

### Issue 2: CUDA Out of Memory

**Solution:** Clear cache and use smaller models:

```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Or request more GPU memory
salloc --partition=v100-32gb-hiprio --gres=gpu:1 --mem=64G
```

### Issue 3: Module Not Found

**Solution:** Ensure virtual environment is activated:

```bash
source venv/bin/activate
which python  # Should show venv path
```

### Issue 4: HuggingFace Download Timeout

**Solution:** Increase timeout or download manually:

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=300  # 5 minutes

# Or pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')"
```

### Issue 5: Disk Quota Exceeded

**Solution:** Use /work instead of /home:

```bash
# Check disk usage
du -sh /work/sjajee/*
quota -s  # Check your quota

# Clean up if needed
rm -rf /work/sjajee/hf_cache/hub/*.lock
```

## Directory Structure on Cluster

```
/work/sjajee/
├── metacog-reasoning/          # Git repository
│   ├── data/
│   │   ├── benchmarks/         # Downloaded datasets (~5GB)
│   │   └── results/            # Evaluation results
│   ├── checkpoints/            # Model checkpoints (will be large)
│   ├── venv/                   # Virtual environment
│   └── ...
├── hf_cache/                   # HuggingFace cache (~50GB)
│   ├── hub/                    # Downloaded models
│   └── datasets/               # Downloaded datasets
└── logs/                       # Job logs (optional)
```

## Resource Requirements

### For Evaluation (Baseline)

- **GPU:** 1× V100 (32GB) or A100 (40GB)
- **Memory:** 32-64GB RAM
- **Time:** 2-3 hours per benchmark
- **Storage:** ~60GB (models + data + results)

### For Training (Phase 1)

- **GPU:** 4-8× A100 (80GB) recommended
- **Memory:** 256-512GB RAM
- **Time:** 2-3 weeks
- **Storage:** ~200GB (checkpoints + data)

## Best Practices

1. **Use /work, not /home:** Better performance and larger quota
2. **Set cache directories:** Avoid filling up /home
3. **Test with small samples first:** Use `--max_samples 10`
4. **Use batch jobs for long runs:** Don't tie up interactive sessions
5. **Monitor GPU usage:** Use `nvidia-smi` to check utilization
6. **Clean up regularly:** Remove old checkpoints and logs
7. **Use version control:** Commit changes before long runs

## Next Steps

After successful setup:

1. **Test evaluation:** Run baseline on 10 samples
2. **Full evaluation:** Run baseline on all benchmarks
3. **Implement training:** Add training scripts for Phase 1
4. **Submit batch jobs:** Use SLURM for long-running jobs
5. **Monitor results:** Check `data/results/` for outputs

## Getting Help

- **Cluster documentation:** Check your university's HPC docs
- **SLURM commands:** `man sbatch`, `man squeue`
- **Project issues:** https://github.com/sankalpjajee/metacog-reasoning/issues

---

**Ready to start?** Once your GPU allocation is granted, follow the Quick Start section above!
