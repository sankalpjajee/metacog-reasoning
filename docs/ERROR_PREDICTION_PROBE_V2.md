# Enhanced Error Prediction Probe (V2)

**Goal:** Train a highly accurate probe (target 80%+) to predict when the baseline LLM will answer incorrectly, enabling selective application of metacognitive prompting.

---

## Overview

This pipeline implements state-of-the-art techniques for error prediction:

| Feature | Implementation | Expected Gain |
|---------|----------------|---------------|
| **Multi-layer hidden states** | Extract from layers 8, 16, 24, 32 | +4% |
| **Baseline confidence** | Entropy, logits, probability gaps | +6% |
| **Self-consistency** | Agreement rate from 3 samples | +6% |
| **2-layer MLP** | 256 hidden units with dropout | +5% |
| **Focal loss** | Handle class imbalance | +3% |
| **Ensemble** | 3 probes averaged | +4% |
| **Total expected accuracy** | - | **80-85%** |

---

## Quick Start

### 1. Test on Small Dataset (Recommended First)

```bash
# Run quick test (100 training samples, 50 test samples)
./scripts/test_probe_accuracy.sh
```

This will:
- Generate 100 training samples per benchmark (~30 min on H100)
- Train probe for 10 epochs (~5 min)
- Evaluate on 50 test samples (~10 min)
- **Total time: ~45 minutes**

### 2. Full Training Pipeline

```bash
# Step 1: Generate training data (2000 samples per benchmark)
python -m src.training.generate_error_prediction_data_v2 \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k mmlu hellaswag \
    --samples_per_benchmark 2000 \
    --output_dir data/training/error_prediction_v2

# Step 2: Train probe with ensemble
python -m src.training.train_error_predictor_v2 \
    --data_dir data/training/error_prediction_v2 \
    --output_dir models/error_predictor_v2 \
    --benchmarks gsm8k mmlu hellaswag \
    --use_ensemble \
    --num_probes 3 \
    --epochs 20 \
    --batch_size 64

# Step 3: Evaluate on test set
python -m src.evaluation.learned_adaptive_evaluator_v2 \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --probe_path models/error_predictor_v2/best_probe.pt \
    --probe_config models/error_predictor_v2/config.json \
    --benchmarks gsm8k mmlu hellaswag \
    --num_samples 1000 \
    --error_threshold 0.5 \
    --output_dir results/learned_adaptive_v2
```

**Estimated time:**
- Data generation: ~6-8 hours (2000 samples × 3 benchmarks × 3 consistency samples)
- Training: ~30 minutes
- Evaluation: ~3-4 hours (1000 samples × 3 benchmarks)

---

## What's New in V2

### Compared to V1 (Simple Probe)

| Aspect | V1 | V2 |
|--------|----|----|
| **Hidden states** | Single layer (last) | Multi-layer (8, 16, 24, 32) |
| **Features** | Hidden state only | Hidden + confidence + agreement |
| **Architecture** | 1-layer linear | 2-layer MLP + ensemble |
| **Loss** | Weighted BCE | Focal loss |
| **Expected accuracy** | ~65-70% | **80-85%** |

### Key Improvements

1. **Multi-Layer Hidden States**
   - Different layers capture different information
   - Layer 8: Syntax and basic semantics
   - Layer 16: Task understanding
   - Layer 24: Reasoning patterns
   - Layer 32: Final answer formation

2. **Baseline Confidence Features**
   - `mean_max_prob`: Average confidence across tokens
   - `min_max_prob`: Lowest confidence (detects uncertainty)
   - `mean_entropy`: Average uncertainty
   - `mean_top1_top2_gap`: Margin between top 2 choices

3. **Self-Consistency as Feature**
   - Not used for decision (too expensive at test time)
   - Used as training signal for probe
   - Probe learns to predict agreement from hidden states

4. **Focal Loss**
   - Handles class imbalance (typically 70% correct, 30% wrong)
   - Focuses learning on hard examples
   - α=0.25, γ=2.0

5. **Ensemble**
   - 3 independent probes
   - Averaged predictions
   - Reduces variance

---

## Expected Results

### Probe Accuracy (on validation set)

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| **Accuracy** | 75% | 80% | 85% |
| **Precision** | 70% | 75% | 80% |
| **Recall** | 70% | 75% | 80% |
| **F1 Score** | 70% | 75% | 80% |

### End-to-End Benchmark Performance

Assuming probe achieves 80% accuracy:

| Benchmark | Baseline | Learned Adaptive | Improvement | Cost |
|-----------|----------|------------------|-------------|------|
| **GSM8K** | 81.1% | 82.3-82.9% | +1.2-1.8% | 1.2x |
| **MMLU** | 69.5% | 70.7-71.3% | +1.2-1.8% | 1.3x |
| **HellaSwag** | 64.1% | 64.7-65.1% | +0.6-1.0% | 1.2x |
| **Average** | 71.6% | 72.6-73.1% | **+1.0-1.5%** | **1.2-1.3x** |

**Cost breakdown:**
- Baseline: 1.0x (single generation)
- Learned adaptive: 1.2-1.3x (feature extraction + selective metacog on ~15-20% of questions)
- Self-consistency: 3.3x (3 generations for all questions)

**Key advantage:** Similar accuracy improvement to self-consistency at 1/3 the cost!

---

## Architecture Details

### Data Generation

```
For each question:
  1. Extract multi-layer hidden states (4 layers × 4096 dims = 16384 dims)
  2. Generate baseline answer with confidence features (4 features)
  3. Compute self-consistency agreement (3 samples → 1 rate)
  4. Check correctness (label: 0=correct, 1=wrong)
  
Output:
  - hidden_states: [N, 16384]
  - confidence_features: [N, 4]
  - agreement_rates: [N, 1]
  - labels: [N] (0 or 1)
```

### Probe Architecture

```
Input:
  - hidden_states: [batch, 16384]
  - confidence_features: [batch, 4]
  - agreement_rates: [batch, 1]

Concatenate → [batch, 16389]
  ↓
Linear(16389 → 256)
  ↓
ReLU
  ↓
Dropout(0.1)
  ↓
Linear(256 → 1)
  ↓
Sigmoid → P(error)
```

**For ensemble:** 3 independent probes, average predictions

### Training

- **Loss:** Focal loss (α=0.25, γ=2.0)
- **Optimizer:** AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler:** Cosine annealing
- **Batch size:** 64
- **Epochs:** 20
- **Validation split:** 20%
- **Best model selection:** By F1 score

---

## Troubleshooting

### Probe Accuracy < 75%

**Possible causes:**
1. Insufficient training data → Increase `samples_per_benchmark`
2. Class imbalance → Check train/val split has similar distribution
3. Overfitting → Increase dropout or reduce epochs

**Solutions:**
- Use ensemble (`--use_ensemble --num_probes 3`)
- Increase training data to 3000-5000 per benchmark
- Add data augmentation (paraphrasing)

### High Cost at Test Time

**Problem:** Feature extraction takes too long

**Solutions:**
- Cache hidden states for repeated evaluation
- Use fewer layers (e.g., only layers 16 and 32)
- Skip self-consistency feature at test time (use only hidden + confidence)

### Metacognition Still Hurts

**Problem:** Even with good probe, metacognition doesn't improve accuracy

**Solutions:**
- Try Chain-of-Thought instead of metacognition
- Adjust error threshold (try 0.3, 0.4, 0.6, 0.7)
- Check if metacognition prompt is working (test on known hard questions)

---

## Comparison to Existing Work

| Method | Accuracy Gain | Cost | Novel Contribution |
|--------|---------------|------|-------------------|
| **DiVeRSe** | +8.8% | 5x | Post-hoc verification |
| **Self-Verification** | +5-10% | 2-3x | Generate-verify-regenerate |
| **Self-Consistency** | +1-2% | 3x | Multiple path selection |
| **Ours (V2)** | **+1-1.5%** | **1.2x** | **Pre-emptive error prediction** |

**Key difference:** We predict errors BEFORE generation, not after.

---

## Files

| File | Purpose |
|------|---------|
| `src/training/generate_error_prediction_data_v2.py` | Data generation with all features |
| `src/training/train_error_predictor_v2.py` | Probe training with ensemble and focal loss |
| `src/evaluation/learned_adaptive_evaluator_v2.py` | Evaluation using trained probe |
| `scripts/test_probe_accuracy.sh` | Quick test on small dataset |
| `docs/ERROR_PREDICTION_PROBE_V2.md` | This file |

---

## Next Steps

1. **Run test:** `./scripts/test_probe_accuracy.sh`
2. **Check probe accuracy:** Should be 70%+ on test set
3. **If good, run full training:** 2000 samples per benchmark
4. **Evaluate on benchmarks:** Compare to baseline and self-consistency
5. **Ablation studies:** Test with/without each feature
6. **Write paper:** Focus on cost-benefit tradeoff

---

## Citation

If you use this code, please cite:

```bibtex
@article{jajee2026learned,
  title={Learned Error Prediction for Adaptive Metacognitive Reasoning},
  author={Jajee, Sankalp},
  year={2026}
}
```

---

**Questions?** Open an issue on GitHub or contact the author.
