# Self-Consistency + Metacognition: Failure Analysis

## Summary: The Method is NOT Working

| Benchmark | Baseline | Self-Cons + Metacog | Change |
|-----------|----------|---------------------|--------|
| **MMLU** | 69.5% | 67.3% | **-2.2%** ❌ |
| **HellaSwag** | 64.1% | 64.9% | +0.8% |
| **Average** | **66.8%** | **66.1%** | **-0.7%** ❌ |

**Conclusion:** Self-consistency + selective metacognition is **hurting performance** overall.

---

## Root Cause: Metacognition Fails on Uncertain Questions

### MMLU Breakdown

| Method | Count | Accuracy | vs Baseline |
|--------|-------|----------|-------------|
| **Baseline (high agreement)** | 681 (68.1%) | **77.7%** | +8.2% ✅ |
| **Metacognition (low agreement)** | 319 (31.9%) | **45.1%** | **-24.4%** ❌ |

### HellaSwag Breakdown

| Method | Count | Accuracy | vs Baseline |
|--------|-------|----------|-------------|
| **Baseline (high agreement)** | 706 (70.6%) | **73.8%** | +9.7% ✅ |
| **Metacognition (low agreement)** | 294 (29.4%) | **43.5%** | **-20.6%** ❌ |

**Key Finding:** Self-consistency correctly identifies "easy" questions (high agreement → high accuracy), but **metacognition makes uncertain questions WORSE** (-20-24% accuracy drop).

---

## Why Metacognition Fails

### 1. **Metacognition Overrides Correct Baseline Answers**

**MMLU Analysis:**
- Metacog failures: 175 / 319 (54.9%)
- Of these 175 failures, **44 cases (25.1%)** had the correct answer in the baseline majority
- **Metacognition changed a correct answer to a wrong answer**

**Example:**
```
Question: "Statement 1 | A factor group of a non-Abelian group is non-Abelian..."
Ground truth: B
Baseline answers: ['D', 'D', 'C'] (majority: D, wrong but 67% agreement)
Metacog answer: A (also wrong, but different from baseline)
```

### 2. **Low Agreement Doesn't Mean Metacognition Will Help**

**The assumption:** Low agreement → model is uncertain → metacognition will help

**The reality:** Low agreement → question is hard → **metacognition also struggles**

| Scenario | Baseline Agreement | Baseline Accuracy | Metacog Accuracy |
|----------|-------------------|-------------------|------------------|
| Easy questions | High (>67%) | **77.7%** (MMLU) | N/A |
| Hard questions | Low (<67%) | ~50-55% (estimated) | **45.1%** (MMLU) ❌ |

**Metacognition is WORSE than baseline on hard questions!**

### 3. **Metacognitive Prompt May Be Too Complex**

The 6-step metacognitive prompt asks the model to:
1. Clarify understanding
2. Analyze each option
3. Monitor confidence
4. Decide on verification
5. Provide final answer with reasoning
6. Rate confidence (0-100%)

**Problem:** This adds cognitive load and may:
- Introduce reasoning errors
- Cause the model to overthink
- Lead to different (often wrong) conclusions

---

## Detailed Failure Patterns

### Pattern 1: Baseline Disagrees, Metacog Makes It Worse

**Example 1 (MMLU):**
```
Q: "Let p = (1, 2, 5, 4)(2, 3) in S_5. Find the index of <p> in S_5."
GT: C (24)
Baseline: ['...formula...', '...order is 8...', 'A'] (33% agreement, all wrong)
Metacog: A (wrong)
```

**Analysis:** Baseline is confused (33% agreement), metacog picks one of the wrong answers.

### Pattern 2: Baseline Has Correct Answer, Metacog Overrides

**Example 2 (MMLU):**
```
Q: "Statement 1 | A factor group of a non-Abelian group is non-Abelian..."
GT: B
Baseline: ['D', 'D', 'C'] (67% agreement on D, wrong)
Metacog: A (wrong, different from baseline)
```

**Analysis:** Even though baseline majority is wrong, at least one baseline sample might have had useful reasoning. Metacog ignores all of them and produces a different wrong answer.

### Pattern 3: High Confidence, Wrong Answer

**Agreement rates:**
- MMLU: 0.87 (very high)
- HellaSwag: 0.89 (very high)

**Problem:** The model is "confident" (agrees with itself) even when wrong. Self-consistency can't detect these cases.

---

## Why Self-Consistency Alone Isn't Enough

### The Fundamental Flaw

**Self-consistency assumption:** If the model disagrees with itself, it's uncertain and needs help.

**Reality:** 
1. **High agreement ≠ correct** (model can be confidently wrong)
2. **Low agreement ≠ metacognition will help** (hard questions are hard for metacognition too)

### The Math

**MMLU Example:**
- Baseline (all questions): 69.5%
- Baseline (high agreement): 77.7% ✅
- Metacog (low agreement): 45.1% ❌

**Net effect:**
```
0.681 × 77.7% + 0.319 × 45.1% = 67.3%
```

**Loss:** -2.2 percentage points

---

## What Went Wrong with Our Hypothesis

### Original Hypothesis
> "When the model disagrees with itself (low self-consistency), it's uncertain. Applying metacognitive prompting will help it reason more carefully and improve accuracy."

### Why It Failed

1. **Low agreement indicates difficulty, not just uncertainty**
   - Hard questions are hard for both baseline and metacognition
   - Metacognition doesn't magically solve hard problems

2. **Metacognitive prompt introduces new errors**
   - More complex prompt → more room for reasoning mistakes
   - The 6-step process may not be appropriate for all question types

3. **Baseline disagreement contains useful information**
   - When baseline produces ['A', 'B', 'C'], one of them might be correct
   - Metacognition ignores this and produces a single answer (often wrong)

---

## Comparison to Other Methods

| Method | MMLU | HellaSwag | Average | Cost |
|--------|------|-----------|---------|------|
| **Baseline** | **69.5%** | 64.1% | **66.8%** | 1.0x |
| Meta Prompt (always) | 64.3% | 53.0% | 58.7% | 1.2x |
| **Self-Cons + Metacog** | 67.3% | **64.9%** | 66.1% | 3.3x |

**Self-consistency + metacog is:**
- Worse than baseline on MMLU (-2.2%)
- Slightly better than baseline on HellaSwag (+0.8%)
- **3.3x more expensive** than baseline
- **Not worth the cost**

---

## Why This Happened

### 1. **Wrong Metacognitive Prompt**

The 6-step prompt was designed for math problems (GSM8K), not knowledge QA (MMLU) or commonsense (HellaSwag).

**Evidence:**
- Meta Prompt on GSM8K: 76.9% (only -4.2% vs baseline)
- Meta Prompt on MMLU: 64.3% (-5.2% vs baseline)
- Meta Prompt on HellaSwag: 53.0% (**-11.1%** vs baseline)

**The prompt gets worse as the task moves away from math reasoning.**

### 2. **Self-Consistency Threshold Too Low**

Agreement threshold: 0.67 (67%)

**Problem:** This triggers metacognition on 30-32% of questions, many of which are just hard (not uncertain).

**Better approach:** Use a higher threshold (e.g., 0.9) to only trigger metacognition on truly uncertain cases.

### 3. **No Task-Specific Adaptation**

The same method is applied to all benchmarks, but:
- Math (GSM8K): Benefits from step-by-step verification
- Knowledge (MMLU): Benefits from confidence awareness, not complex reasoning
- Commonsense (HellaSwag): Doesn't benefit from metacognition at all

---

## Recommendations

### 1. **DO NOT Proceed with Current Approach**

Self-consistency + selective metacognition is **not working**:
- MMLU: -2.2%
- HellaSwag: +0.8%
- Average: -0.7%
- Cost: 3.3x

**This is not publishable.**

### 2. **Pivot to Learned Confidence Prediction**

Instead of using self-consistency at test time, train a probe to predict:
- **Not just "will the model be wrong?"**
- But **"will metacognition help?"**

**Key insight:** Low agreement doesn't mean metacognition will help. We need to learn when metacognition helps vs hurts.

### 3. **Design Task-Specific Metacognitive Prompts**

Current prompt is one-size-fits-all. We need:
- **Math (GSM8K):** Step-by-step verification prompt
- **Knowledge (MMLU):** Confidence and alternative answers prompt
- **Commonsense (HellaSwag):** Skip metacognition entirely, or use a minimal prompt

### 4. **Alternative Approaches to Explore**

| Approach | Description | Expected Improvement |
|----------|-------------|---------------------|
| **Self-Refine** | Iterative refinement | +1-2% |
| **Chain-of-Thought** | Explicit reasoning | +2-3% |
| **Ensemble (3x baseline)** | Vote among 3 baseline samples | +1-2% |
| **Learned confidence → CoT** | Use probe to decide when to use CoT | +1-2% at 1.5x cost |

---

## Next Steps

### Immediate (This Week)
1. ❌ **Stop self-consistency experiments** - not working
2. ✅ **Run GSM8K self-consistency** (to complete the picture)
3. ✅ **Analyze why metacognition fails** (done)

### Short-term (Next Week)
4. **Pivot to learned confidence prediction**
   - Train probe to predict: "Will metacognition help on this question?"
   - Use features: question, baseline answer, task type
   - Target: Predict when metacog improves over baseline

5. **Try alternative prompts**
   - Chain-of-Thought (simpler than metacognition)
   - Self-Refine (iterative improvement)
   - Task-specific prompts

### Long-term (2-3 Weeks)
6. **Implement learned-adaptive with CoT**
   - Use probe to decide: baseline vs CoT
   - CoT is simpler and more effective than metacognition
   - Target: +1-2% at 1.5x cost

---

## Conclusion

**Self-consistency + selective metacognition does NOT work because:**

1. ✅ Self-consistency correctly identifies easy questions (high agreement → high accuracy)
2. ❌ But metacognition **makes hard questions worse** (45% accuracy on low-agreement questions)
3. ❌ Net effect: **-0.7% average accuracy** at **3.3x cost**

**The fundamental flaw:** Low agreement indicates difficulty, not that metacognition will help.

**The path forward:** Train a learned confidence predictor to identify when simpler methods (like CoT) will help, not when the model is uncertain.

**For NeurIPS:** We need to show a method that improves accuracy at reasonable cost. Current approach doesn't meet this bar.
