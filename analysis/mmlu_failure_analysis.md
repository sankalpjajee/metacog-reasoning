# MMLU Catastrophic Failure Analysis

**Date:** January 26, 2026  
**Model:** Llama 3.1 8B Instruct  
**Issue:** Adaptive metacognitive prompting causes -32.3% degradation on MMLU

---

## Summary

**Adaptive metacognitive prompting catastrophically fails on MMLU:**
- Baseline: 69.5%
- Adaptive Metacog: 37.2%
- **Degradation: -32.3%**

**Root cause:** Model misclassifies 97.9% of MMLU questions as COMPLEX, applying full metacognition to simple factual questions.

---

## The Numbers

### Overall Performance

| Benchmark | Baseline | Adaptive | Difference |
|-----------|----------|----------|------------|
| GSM8K | 81.1% | 80.3% | -0.8% ✓ |
| MMLU | 69.5% | 37.2% | **-32.3%** ✗✗✗ |

### MMLU Classification Breakdown

| Classification | Count | % | Accuracy |
|----------------|-------|---|----------|
| **COMPLEX** | 979 | 97.9% | **36.8%** ❌ |
| **SIMPLE** | 13 | 1.3% | **76.9%** ✓ |

### Error Patterns

| Error Type | Count | % of Errors |
|------------|-------|-------------|
| Misclassification-induced | ~600 | 95.5% |
| Repetitive loops | 4 | 0.6% |
| Truncated responses | 5 | 0.8% |
| Other | 19 | 3.0% |

---

## Why This Happens

### 1. Misclassification Problem

**The adaptive prompt asks:**
> "Is this question SIMPLE or COMPLEX?"

**Model's interpretation:**
- MMLU questions are academic/technical → sounds complex
- Multiple choice options → needs analysis
- Model defaults to "complex" when uncertain

**Result:** 97.9% misclassified as COMPLEX

### 2. Metacognition Hurts Simple Questions

**Example: Factual question**
```
Question: "What is the capital of France?"
Correct answer: Paris

Baseline approach:
- Direct answer: "Paris" ✓

Metacognitive approach:
1. "This is COMPLEX because it requires geographical knowledge"
2. "Let me clarify: France is a country in Europe..."
3. "Preliminary solution: Paris is commonly known as the capital"
4. "Verification: Let me check if Paris is indeed the capital..."
5. "Wait, could it be a trick question?"
6. Gets confused → Wrong answer ❌
```

### 3. When Classified Correctly, It Works

**The 13 SIMPLE questions:**
- Accuracy: 76.9% (near baseline 69.5%)
- Model answers directly
- No overthinking

**But only 1.3% are classified correctly!**

---

## Comparison to GSM8K

### Why GSM8K Works Better

**GSM8K (math problems):**
- Most problems ARE actually complex
- Multi-step reasoning is appropriate
- Metacognition helps (slightly)
- Result: -0.8% (acceptable)

**MMLU (knowledge):**
- Most questions are simple factual recall
- Metacognition causes overthinking
- Model misclassifies 97.9% as complex
- Result: -32.3% (catastrophic)

---

## Key Insights

### 1. Binary Classification Doesn't Work

**Problem:**
- SIMPLE or COMPLEX is too coarse
- Model defaults to COMPLEX when uncertain
- No middle ground

**Solution:**
- Use confidence thresholds (continuous scale)
- Let model self-assess uncertainty
- Apply appropriate level of metacognition

### 2. Explicit Prompting Has Limits

**What we learned:**
- 8B models can't accurately classify complexity
- Even with careful prompt engineering
- Misclassification rate: 97.9%

**Implications:**
- Need training-based approach
- Or test larger models (70B)
- Or use task-specific prompts

### 3. Metacognition Isn't Always Bad

**When it works:**
- Correctly classified SIMPLE: 76.9% accuracy
- GSM8K (appropriate complexity): -0.8% only

**When it fails:**
- Misclassified as COMPLEX: 36.8% accuracy
- Applied to simple factual questions

---

## Next Steps

### 1. Test Threshold-Based Approach

**Replace binary classification with confidence:**
```
Confidence ≥ 90%: Direct answer
Confidence 70-89%: Light verification
Confidence < 70%: Full metacognition
```

**Expected outcome:**
- MMLU: Should stay near baseline (69.5%)
- GSM8K: Might improve (>81.1%)

**Test command:**
```bash
python -m src.evaluation.threshold_evaluator \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k mmlu \
    --num_samples 1000 \
    --output_dir results/threshold_1000
```

### 2. Test Targeted Approach

**Apply metacognition only to baseline errors:**

**Test command:**
```bash
python -m src.evaluation.targeted_evaluator \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --baseline_file results/baseline_1000/mmlu_results.json \
    --benchmark mmlu \
    --output_dir results/targeted_mmlu
```

**Key metric:** Fix rate (% of errors corrected by metacognition)

### 3. Test 70B Model

**Question:** Does 70B classify better than 8B?

**Test:**
- 70B baseline
- 70B adaptive (with classification)
- 70B threshold (without classification)

**If 70B also misclassifies:**
- Confirms explicit prompting doesn't work
- Move to training-based approach

---

## Research Implications

### Finding #1: Misclassification is the Bottleneck

**Not the metacognitive process itself:**
- When correctly classified as SIMPLE: 76.9% (works!)
- When misclassified as COMPLEX: 36.8% (fails!)

**The problem is classification accuracy: 1.3% correct**

### Finding #2: Binary Classification is Insufficient

**Need continuous scale:**
- Not SIMPLE vs COMPLEX
- But confidence: 0-100%
- With thresholds for different levels of metacognition

### Finding #3: Task-Specific Prompting May Be Necessary

**One prompt doesn't fit all:**
- GSM8K: Metacognition appropriate (-0.8%)
- MMLU: Metacognition harmful (-32.3%)

**Options:**
1. Threshold-based (adaptive depth)
2. Task-specific (different prompts)
3. Training-based (implicit metacognition)

---

## Conclusion

**Adaptive metacognitive prompting fails on MMLU because:**

1. ❌ Model misclassifies 97.9% of questions as COMPLEX
2. ❌ Metacognition on simple questions causes overthinking
3. ❌ Binary classification is too coarse
4. ❌ 8B models lack capacity for accurate complexity assessment

**Next steps:**
1. Test threshold-based approach (confidence > classification)
2. Test targeted approach (errors only)
3. Test 70B to see if scale helps
4. Consider training-based implicit metacognition

**The misclassification rate (97.9%) is the smoking gun.**

---

**Files:**
- Full results: `results/metacognitive_adaptive_1000samples/metacog_mmlu_test_results.json`
- Analysis script: `analyze_mmlu.py`
- Prompt history: `prompts/PROMPT_HISTORY.md`

**Last Updated:** January 26, 2026
