# Metacognitive Prompting: Version History

This document tracks all versions of metacognitive prompts used in experiments, along with their performance results.

---

## Version 1: Original Metacognitive Prompt (7-step explicit)

**Date:** January 2026  
**File:** `src/evaluation/metacognitive_evaluator.py` (original version)

**Structure:**
- 7 explicit metacognitive steps
- Applied to all questions uniformly
- No complexity classification

**Prompt:**
```
Follow these metacognitive steps:

1. Clarify your understanding of what the question is asking
2. Make a preliminary analysis of each option
3. Monitor your confidence in the solution
4. Decide whether you need additional verification
5. Provide your final answer with clear reasoning
6. Rate your overall confidence (0-100%)
7. [Additional verification step if needed]
```

**Performance (Llama 3.1 8B, 1000 samples):**
- GSM8K: 76.0% (baseline: 81.1%) → **-5.1%**
- MMLU: 64.3% (baseline: 69.5%) → **-5.2%**

**Issues:**
- Overthinking simple questions
- Verification introduces errors
- Consistent degradation across benchmarks

---

## Version 2: Adaptive Metacognitive Prompt (SIMPLE/COMPLEX classification)

**Date:** January 24, 2026  
**File:** `src/evaluation/metacognitive_evaluator.py` (current version, lines 126-213)

**Structure:**
- First assess if question is SIMPLE or COMPLEX
- SIMPLE: Answer directly
- COMPLEX: Apply 6-step metacognitive process

**Prompt (MMLU/HellaSwag/MR-Ben):**
```
First, assess if this question is SIMPLE or COMPLEX:
- SIMPLE: The answer is immediately clear from basic knowledge or intuition
- COMPLEX: Requires careful analysis, comparison of options, or multi-step reasoning

If SIMPLE: Choose the answer directly and provide your final answer.

If COMPLEX: Follow these steps:

1. Clarify your understanding of what the question is asking.
2. Make a preliminary analysis of each option.
3. Monitor your confidence in the solution. If you think it's wrong or has potential errors, pause and verify your work.
4. Once you have selected an option, decide whether you need additional verification or if you're confident enough to finalize.
5. Provide your final answer with a clear explanation of your reasoning.
6. Rate your overall confidence (0-100%) in this answer and explain why.
```

**Performance (Llama 3.1 8B, 1000 samples):**
- GSM8K: 80.3% (baseline: 81.1%) → **-0.8%** ✓
- MMLU: 37.2% (baseline: 69.5%) → **-32.3%** ✗✗✗

**Classification Analysis (MMLU):**
- Classified as COMPLEX: 979/1000 (97.9%)
- Classified as SIMPLE: 13/1000 (1.3%)
- COMPLEX accuracy: 36.8%
- SIMPLE accuracy: 76.9%

**Issues:**
- Model misclassifies 97.9% of MMLU questions as COMPLEX
- Binary classification doesn't work
- Catastrophic failure on knowledge-based questions
- Works reasonably well on math (GSM8K)

**Key Finding:**
> The adaptive prompt fails because 8B models cannot accurately classify problem complexity. Even when correctly classified as SIMPLE (13 cases), accuracy is near baseline (76.9%). The problem is misclassification, not the metacognitive process itself.

---

## Version 3: Threshold-Based Metacognitive Prompt (PLANNED)

**Date:** January 26, 2026  
**Status:** In development

**Structure:**
- Generate initial answer with confidence rating
- Apply metacognition based on confidence threshold:
  - Confidence ≥ 90%: Direct answer (no metacognition)
  - Confidence 70-89%: Light verification
  - Confidence < 70%: Full metacognitive analysis

**Rationale:**
- Replaces binary SIMPLE/COMPLEX with continuous confidence scale
- Allows model to self-assess uncertainty
- Adaptive depth of reasoning based on confidence
- Avoids misclassification problem

**Expected Performance:**
- MMLU: Should stay near baseline (69.5%) for high-confidence questions
- GSM8K: Might improve over baseline (>81.1%) with targeted verification

**Status:** To be implemented and tested

---

## Summary Table

| Version | Type | GSM8K | MMLU | Key Issue |
|---------|------|-------|------|-----------|
| Baseline | None | 81.1% | 69.5% | - |
| V1: Original | Always apply | 76.0% (-5.1%) | 64.3% (-5.2%) | Overthinking |
| V2: Adaptive | Binary classification | 80.3% (-0.8%) | 37.2% (-32.3%) | Misclassification |
| V3: Threshold | Confidence-based | TBD | TBD | TBD |

---

## Research Insights

### Why Explicit Metacognitive Prompting Fails (8B Models)

1. **Verification introduces errors** (73.6% of GSM8K errors are "verification errors")
2. **Models misinterpret problems** when asked to "clarify understanding"
3. **Binary classification doesn't work** (97.9% misclassified as COMPLEX)
4. **Models lack self-monitoring capacity** (high confidence in wrong answers)

### Next Steps

1. Test threshold-based approach (Version 3)
2. Test if 70B models classify better than 8B
3. Consider task-specific prompting (different prompts for different benchmarks)
4. Move to training-based implicit metacognition if prompting continues to fail

---

**Last Updated:** January 26, 2026
