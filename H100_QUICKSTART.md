# H100 Server Quick Start Guide

## You're on the H100 server! Here's what to do:

### Step 1: Fix the Installation

The error you encountered is because `indicnlp-corpus` doesn't exist on PyPI. It's not needed for baseline evaluation (only for Phase 2 Indic language training).

Run this to fix and complete setup:

```bash
cd ~/metacog-reasoning

# Pull the fixed version
git pull

# Run the H100-specific setup script
bash setup_h100.sh
```

This will:
- Detect your CUDA version automatically
- Install PyTorch with correct CUDA support
- Install all dependencies (without problematic packages)
- Set up cache directories
- Verify GPU access

**Time:** ~5-10 minutes

### Step 2: Download Benchmarks

```bash
# Make sure you're in the project directory
cd ~/metacog-reasoning

# Activate environment
source venv/bin/activate

# Download all 6 benchmarks
bash scripts/download_benchmarks.sh
```

**Time:** ~20-30 minutes  
**Size:** ~5GB

### Step 3: Quick Test

Test with just 10 samples to verify everything works:

```bash
python scripts/evaluate_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k \
    --max_samples 10 \
    --output_dir data/results/baseline_test
```

**Time:** ~2 minutes on H100 (vs ~10 minutes on V100!)

### Step 4: Full Evaluation

Once the test works, run full evaluation:

```bash
python scripts/evaluate_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k,math,mmlu,hellaswag,bigbench,humaneval \
    --output_dir data/results/baseline
```

**Time:** ~1.5 hours on H100 (vs ~8 hours on V100!)

## Using tmux for Long Runs

For jobs that take hours, use `tmux` so they keep running if you disconnect:

```bash
# Start a new tmux session
tmux new -s eval

# Run your command
python scripts/evaluate_baseline.py ...

# Detach from session: Press Ctrl+B, then D
# Your job keeps running!

# Later, reattach to see progress:
tmux attach -s eval

# List all sessions:
tmux ls

# Kill a session:
tmux kill-session -t eval
```

## Using Both H100 GPUs

You have **2× H100 GPUs**! Use them:

```bash
# Check which GPUs are available
nvidia-smi

# Use GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_baseline.py ...

# Use GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_baseline.py ...

# Use both GPUs (for training in Phase 1)
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_teacher.py ...
```

## Monitoring GPU Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Check your processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Kill a stuck process
kill -9 PID
```

## What Each Script Does

### `setup_h100.sh`
- One-time setup
- Installs everything
- Detects CUDA version
- Verifies GPU access

### `scripts/download_benchmarks.sh`
- Downloads all 6 benchmarks
- Caches them locally
- ~5GB total

### `scripts/evaluate_baseline.py`
- Evaluates model on benchmarks
- Saves detailed results
- Can test with `--max_samples 10`

### `scripts/compare_models.py`
- Compares multiple models
- Generates comparison tables
- Use after training models

## Expected Performance on H100

| Benchmark | Samples | Time on H100 |
|:----------|:--------|:-------------|
| GSM8k | 1,319 | ~5 minutes |
| MATH | 5,000 | ~15 minutes |
| MMLU | 14,042 | ~30 minutes |
| HellaSwag | 10,042 | ~20 minutes |
| BIG-Bench | varies | ~10 minutes |
| HumanEval | 164 | ~2 minutes |
| **Total** | **~30K** | **~1.5 hours** |

Compare to V100: **~8 hours** → **5× faster!**

## Troubleshooting

### "No module named 'src'"

```bash
# Make sure you're in the project directory
cd ~/metacog-reasoning

# Activate venv
source venv/bin/activate

# Install in development mode
pip install -e .
```

### "CUDA out of memory"

```bash
# Reduce batch size
python scripts/evaluate_baseline.py --batch_size 4

# Or use specific GPU
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_baseline.py
```

### "venv/bin/activate: No such file or directory"

```bash
# Create venv first
python3 -m venv venv

# Then activate
source venv/bin/activate
```

## File Locations

```
~/metacog-reasoning/
├── data/
│   ├── benchmarks/          # Downloaded datasets (~5GB)
│   └── results/             # Evaluation results
├── venv/                    # Virtual environment (~2GB)
├── logs/                    # Log files
└── scripts/                 # All scripts

~/hf_cache/                  # HuggingFace cache (~50GB)
├── hub/                     # Downloaded models
└── datasets/                # Cached datasets
```

## Next Steps

1. ✅ **Setup complete** → `bash setup_h100.sh`
2. ✅ **Download benchmarks** → `bash scripts/download_benchmarks.sh`
3. ✅ **Test evaluation** → `python scripts/evaluate_baseline.py --max_samples 10`
4. ✅ **Full evaluation** → `python scripts/evaluate_baseline.py`
5. 🔜 **Implement training** → Phase 1 (teacher model)
6. 🔜 **Train models** → 3-5 days on H100 (vs 2-3 weeks on V100!)
7. 🔜 **Compare results** → `python scripts/compare_models.py`

## Getting Help

- **Server admin:** Guiu Puigcercos i Vilar (guiupuigc@musc.edu)
- **Team wiki:** http://bmic-nlp-wks.mdc.musc.edu:8282
- **GitHub repo:** https://github.com/sankalpjajee/metacog-reasoning
- **Issues:** https://github.com/sankalpjajee/metacog-reasoning/issues

---

**You have 2× H100 GPUs - one of the most powerful AI training setups available!** 🚀
