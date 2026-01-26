# Metacognitive Prompting Evaluation

This directory contains documentation and scripts for evaluating different metacognitive prompting strategies.

---

## Quick Start

### 1. Threshold-Based Evaluation

Apply metacognition based on confidence thresholds:

```bash
python -m src.evaluation.threshold_evaluator \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k mmlu \
    --num_samples 1000 \
    --output_dir results/threshold_1000
```

### 2. Targeted Evaluation (Errors Only)

Apply metacognition only to baseline errors:

```bash
python -m src.evaluation.targeted_evaluator \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --baseline_file results/baseline_1000/gsm8k_results.json \
    --benchmark gsm8k \
    --output_dir results/targeted_gsm8k
```

---

## Evaluation Strategies

### Baseline Evaluation
- **Script:** `scripts/evaluate_baseline.py`
- **Description:** Standard evaluation without metacognition
- **Use:** Establish baseline performance

### Adaptive Metacognitive Evaluation
- **Script:** `src/evaluation/metacognitive_evaluator.py`
- **Description:** Binary SIMPLE/COMPLEX classification
- **Issue:** Misclassifies 97.9% of MMLU as COMPLEX
- **Performance:** GSM8K -0.8%, MMLU -32.3%

### Threshold-Based Evaluation (NEW)
- **Script:** `src/evaluation/threshold_evaluator.py`
- **Description:** Confidence-based metacognition
- **Thresholds:**
  - ≥90%: Direct answer
  - 70-89%: Light verification
  - <70%: Full metacognition
- **Status:** Ready to test

### Targeted Evaluation (NEW)
- **Script:** `src/evaluation/targeted_evaluator.py`
- **Description:** Apply metacognition only to baseline errors
- **Purpose:** Test if metacognition helps on hard questions
- **Status:** Ready to test

---

## Workflow

### Step 1: Run Baseline
```bash
python scripts/evaluate_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k,mmlu \
    --max_samples 1000 \
    --output_dir results/baseline_1000
```

### Step 2: Test Threshold Approach
```bash
python -m src.evaluation.threshold_evaluator \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k mmlu \
    --num_samples 1000 \
    --output_dir results/threshold_1000
```

### Step 3: Targeted Error Analysis
```bash
# For GSM8K
python -m src.evaluation.targeted_evaluator \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --baseline_file results/baseline_1000/gsm8k_results.json \
    --benchmark gsm8k \
    --output_dir results/targeted_gsm8k

# For MMLU
python -m src.evaluation.targeted_evaluator \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --baseline_file results/baseline_1000/mmlu_results.json \
    --benchmark mmlu \
    --output_dir results/targeted_mmlu
```

---

## Expected Outcomes

### Threshold-Based Evaluation

**If successful:**
- MMLU accuracy near baseline (69.5%)
- GSM8K accuracy improved (>81.1%)
- High-confidence questions answered directly
- Low-confidence questions get metacognition

**If unsuccessful:**
- Model reports high confidence even when wrong
- Or model too conservative (low confidence on everything)
- Indicates self-reported confidence isn't reliable

### Targeted Evaluation

**Key metrics:**
- **Fix rate:** % of baseline errors fixed by metacognition
- **Projected accuracy:** Baseline + fixed errors
- **Improvement:** Difference from baseline

**If fix rate > 50%:**
- ✅ Metacognition helps on hard questions
- ✅ Problem is applying it to easy questions

**If fix rate < 20%:**
- ❌ Metacognition doesn't help even on hard questions
- ❌ Need different approach (training-based)

---

## Files

- `PROMPT_HISTORY.md` - Complete history of all prompt versions and results
- `README.md` - This file
- `../src/evaluation/threshold_evaluator.py` - Threshold-based evaluation
- `../src/evaluation/targeted_evaluator.py` - Error-focused evaluation
- `../src/evaluation/metacognitive_evaluator.py` - Original adaptive evaluation

---

## Research Questions

1. **Can models self-assess confidence accurately?**
   - Test: Threshold evaluation
   - Metric: Correlation between confidence and correctness

2. **Does metacognition help on hard questions?**
   - Test: Targeted evaluation
   - Metric: Fix rate on baseline errors

3. **Is the problem classification or metacognition itself?**
   - Compare: Adaptive (97.9% misclassified) vs Threshold (continuous)
   - If threshold works: Problem was classification
   - If threshold fails: Problem is metacognition

4. **Does model scale matter?**
   - Test: 8B vs 70B on same prompts
   - Question: Do larger models classify/reason better?

---

## Next Steps

1. Run threshold evaluation on 8B (fast test)
2. If promising, run on 70B
3. Run targeted evaluation to measure fix rate
4. Compare all approaches in final analysis

---

**Last Updated:** January 26, 2026
