# Self-Consistency Based Selective Metacognition

## Overview

This evaluator implements selective metacognition using **self-consistency** as an uncertainty signal. It addresses the core challenge of determining when to apply metacognitive reasoning by using an objective, external measure of model uncertainty.

## How It Works

### Algorithm

1. **Generate Multiple Baseline Answers** (N=3 by default)
   - Use baseline prompt (no metacognition)
   - Sample with temperature=0.7 for diversity
   - Extract answers from each response

2. **Compute Agreement Rate**
   - Count how many answers agree
   - Agreement = (most common answer count) / N
   - Identify majority answer

3. **Selective Application**
   - If agreement ≥ threshold (default 0.67):
     - **High confidence** → Use majority answer (baseline)
   - If agreement < threshold:
     - **Low confidence** → Apply full 6-step metacognitive reasoning

### Key Insight

**Self-consistency provides an objective uncertainty signal** that doesn't require the model to self-assess. High agreement indicates the model is confident (single dominant mode in the probability distribution), while low agreement indicates uncertainty (multiple competing modes).

## Usage

### Basic Usage

```bash
python -m src.evaluation.selfconsistency_evaluator \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k,mmlu \
    --num_samples 100 \
    --output_dir results/selfconsistency_100
```

### Full Parameters

```bash
python -m src.evaluation.selfconsistency_evaluator \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k,mmlu,hellaswag,mrben \
    --num_samples 1000 \
    --n_samples 3 \
    --agreement_threshold 0.67 \
    --temperature 0.7 \
    --output_dir results/selfconsistency_1000
```

### Parameters

- `--model_name`: Model to evaluate (default: meta-llama/Llama-3.1-8B-Instruct)
- `--benchmarks`: Comma-separated list of benchmarks (gsm8k, mmlu, hellaswag, mrben)
- `--num_samples`: Number of samples per benchmark (None = all)
- `--n_samples`: Number of samples for self-consistency check (default: 3)
- `--agreement_threshold`: Agreement threshold 0-1 (default: 0.67)
- `--temperature`: Sampling temperature for diversity (default: 0.7)
- `--output_dir`: Directory to save results

## Expected Results

### Performance

Based on literature (Wang et al., 2022) and our oracle experiments:

| Metric | Expected Value |
|--------|----------------|
| **Accuracy improvement** | +5-6% over baseline |
| **% of oracle improvement** | ~70% |
| **Baseline usage** | ~70-80% of questions |
| **Metacog usage** | ~20-30% of questions |

### Cost

- **Inference cost:** 3-4x baseline
  - 3x for self-consistency check
  - +0.2-0.3x for metacognition on disagreements
- **Latency:** 3-4x baseline (can be parallelized)

## Comparison to Other Approaches

| Approach | Signal | Accuracy | Cost | Training Required |
|----------|--------|----------|------|-------------------|
| **Baseline** | None | 81.1% (GSM8K) | 1x | No |
| **Always Metacog** | None | 76.9% (-4.2%) | 2x | No |
| **Adaptive (SIMPLE/COMPLEX)** | Explicit classification | 80.3% (-0.8%) to 37.2% (-32%) | 1-2x | No |
| **Threshold (Confidence)** | Self-reported confidence | 55.7% (-25%) | 1-2x | No |
| **Self-Consistency** | Answer variance | **~86% (+5%)** | 3-4x | **No** ✅ |
| **Oracle** | Know answer | 89.5% (+8.4%) | 1-2x | No (impractical) |

## Why Self-Consistency Works

### Theoretical Foundation

From Wang et al. (2022):

> "A complex reasoning problem typically admits multiple different ways of thinking leading to its unique correct answer. [...] If the model-generated reasoning paths are diverse and lead to a consistent answer, we have greater confidence that the final answer is correct."

### Probabilistic Interpretation

Self-consistency approximates:
```
P(answer | question) ≈ ∑ P(answer | reasoning_path) × P(reasoning_path | question)
```

By sampling multiple reasoning paths and checking agreement, we estimate the marginal probability distribution over answers. High agreement = low epistemic uncertainty.

### Why It's Better Than Explicit Prompting

1. **Objective signal:** Doesn't rely on model's self-assessment
2. **Proven to work:** 3850+ citations, widely validated
3. **No training required:** Zero-shot approach
4. **Generalizes across tasks:** Works for math, knowledge, reasoning

## Output Format

### Results File

```json
{
  "benchmark": "gsm8k",
  "n_samples": 3,
  "agreement_threshold": 0.67,
  "temperature": 0.7,
  "accuracy": 0.86,
  "correct": 860,
  "total": 1000,
  "baseline_count": 750,
  "metacog_count": 250,
  "baseline_accuracy": 0.88,
  "metacog_accuracy": 0.80,
  "avg_agreement": 0.75,
  "results": [...]
}
```

### Per-Sample Result

```json
{
  "question": "Janet's ducks lay 16 eggs per day...",
  "ground_truth": "18",
  "baseline_answers": ["18", "18", "18"],
  "agreement_rate": 1.0,
  "majority_answer": "18",
  "method_used": "baseline",
  "final_answer": "18",
  "correct": true,
  "full_response": "..."
}
```

## Next Steps

### If Self-Consistency Works (+5-6%)

1. **Run full evaluation** (1000 samples)
2. **Write up findings** for publication
3. **Move to Phase 2:** Train difficulty classifier using self-consistency labels

### If Self-Consistency Doesn't Work

1. **Analyze failure modes:** Why doesn't agreement correlate with correctness?
2. **Try variants:** Different N, thresholds, temperatures
3. **Move to training-based approaches:** RL, confidence calibration, MoE

## References

1. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *arXiv preprint arXiv:2203.11171*.

2. Our internal experiments showing +8% oracle improvement (Jan 2026).

3. Literature review: `literature_review/selective_metacognition_review.md`

## Contact

For questions or issues, see the main project README or open a GitHub issue.
