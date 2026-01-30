# Self-Consistency Improves Chain of Thought Reasoning in Language Models

**Paper:** arXiv:2203.11171 (March 2022, ICLR 2023)  
**Authors:** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou  
**URL:** https://arxiv.org/abs/2203.11171  
**Citations:** 3850+

## Key Contribution

**Problem:** Chain-of-thought prompting uses greedy decoding, which only takes the single most likely reasoning path.

**Insight:** A complex reasoning problem typically admits multiple different ways of thinking leading to its unique correct answer.

**Solution:** Self-consistency - sample multiple reasoning paths and select the most consistent answer.

## Methodology

### Algorithm:

```
1. Sample diverse reasoning paths (temperature > 0)
   - Generate N different reasoning chains
   - Each chain may use different approaches

2. Extract final answers from each path
   - Parse the answer from each reasoning chain

3. Marginalize out reasoning paths
   - Take majority vote over final answers
   - The most consistent answer is selected
```

### Why This Works:

**Intuition:** If multiple independent reasoning paths lead to the same answer, that answer is likely correct.

**Mathematical interpretation:**
- Each reasoning path samples from P(reasoning | question)
- Final answer aggregation approximates P(answer | question)
- Marginalizing out reasoning paths reduces variance

## Results

**Massive improvements on reasoning benchmarks:**

| Benchmark | Improvement |
|-----------|-------------|
| GSM8K | +17.9% |
| SVAMP | +11.0% |
| AQuA | +12.2% |
| StrategyQA | +6.4% |
| ARC-challenge | +3.9% |

**Key finding:** Self-consistency dramatically improves reasoning performance without any model training.

## Relevance to Our Work

### Connection to Selective Metacognition:

**Self-consistency provides an uncertainty signal:**
- High agreement (e.g., 5/5 same answer) → High confidence → Skip metacognition
- Low agreement (e.g., 2/3/0 split) → Low confidence → Apply metacognition

**This is exactly what we need!**

### Advantages:

1. ✅ **No training required** (zero-shot approach)
2. ✅ **Objective uncertainty signal** (answer variance)
3. ✅ **Proven to work** (3850+ citations, ICLR 2023)
4. ✅ **Model-agnostic** (works with any LLM)

### Disadvantages:

1. ❌ **Computational cost** (N forward passes for agreement check)
2. ❌ **Only works for tasks with discrete answers** (not open-ended generation)
3. ❌ **May not generalize to all question types**

## How to Use for Selective Metacognition

### Proposed Algorithm:

```python
def selective_metacognition_via_self_consistency(question, n_samples=3):
    # Step 1: Generate multiple baseline answers
    answers = []
    for i in range(n_samples):
        answer = model.generate(question, temperature=0.7)
        answers.append(answer)
    
    # Step 2: Check agreement
    unique_answers = set(answers)
    agreement_rate = max(answers.count(a) for a in unique_answers) / n_samples
    
    # Step 3: Decide whether to apply metacognition
    if agreement_rate >= 0.67:  # 2/3 or better agreement
        # High confidence → use majority answer
        return most_common(answers)
    else:
        # Low confidence → apply metacognition
        return model.generate_with_metacognition(question)
```

### Expected Performance:

**If 80% of questions have high agreement:**
- 80% use baseline (fast)
- 20% use metacognition (slow but accurate)
- Average cost: 3.0 + 0.2×2 = 3.4x baseline
- Accuracy: Baseline on easy + Metacog on hard

**Compared to our oracle results:**
- Oracle: +8% improvement (knows correct answer)
- Self-consistency: +5-6% improvement (estimates from agreement)
- **70-75% of oracle performance**

## Relationship to Probability Distributions

### Question: How does self-consistency relate to LLM probability distributions?

**Answer:** Self-consistency is sampling from the implicit distribution over reasoning paths.

### Mathematical Framework:

**LLM defines a distribution:**
```
P(answer | question) = ∫ P(answer | reasoning, question) × P(reasoning | question) d(reasoning)
```

**Greedy decoding:**
```
answer = argmax P(token_t | token_1...token_{t-1}, question)
```
- Takes single most likely path
- Ignores alternative reasoning approaches

**Self-consistency:**
```
1. Sample reasoning paths: r_1, r_2, ..., r_N ~ P(reasoning | question)
2. Extract answers: a_1, a_2, ..., a_N
3. Aggregate: answer = majority_vote(a_1, ..., a_N)
```
- Approximates marginal distribution P(answer | question)
- Reduces variance by averaging over multiple paths

### Why This Provides Uncertainty:

**High agreement:**
- P(answer_A | question) ≈ 1.0
- Model is confident
- Multiple reasoning paths converge

**Low agreement:**
- P(answer_A | question) ≈ 0.4, P(answer_B | question) ≈ 0.3, ...
- Model is uncertain
- Reasoning paths diverge

**This is a natural uncertainty signal!**

## Connection to Metacognition

### Human Metacognition:

**When humans are uncertain:**
- Try multiple approaches
- Check if they lead to same answer
- If not, think more carefully

**Self-consistency mimics this:**
- Generate multiple reasoning paths
- Check agreement
- If low, apply more careful reasoning (metacognition)

### Why This Is Better Than Confidence Prompting:

| Approach | Signal | Reliability |
|----------|--------|-------------|
| **Confidence prompting** | Model self-reports confidence | ❌ Unreliable (our threshold approach failed) |
| **Self-consistency** | Answer variance across samples | ✅ Objective, proven to work |

**Key insight:** Don't ask the model "Are you confident?" Instead, check if it gives consistent answers.

## Implementation Considerations

### 1. Number of Samples (N):

**Trade-off:**
- N=3: Fast, less reliable agreement signal
- N=5: Balanced
- N=10: Slow, very reliable agreement signal

**Recommendation:** Start with N=3 for speed

### 2. Agreement Threshold:

**Options:**
- Strict (100% agreement): Apply metacog to 40-50% of questions
- Moderate (67% agreement): Apply metacog to 20-30% of questions
- Lenient (50% agreement): Apply metacog to 10-15% of questions

**Recommendation:** Start with 67% (2/3 agreement)

### 3. Temperature:

**For sampling:**
- Too low (0.1): Not enough diversity
- Too high (1.0): Too random
- **Recommended:** 0.5-0.7

### 4. Answer Extraction:

**Challenge:** Need to reliably extract final answer from reasoning chains

**Solution:**
- Use consistent format (e.g., "The answer is: X")
- Or train answer extractor
- Or use regex patterns

## Comparison to Training-Based Approaches

| Approach | Method | Cost | Accuracy | Efficiency |
|----------|--------|------|----------|------------|
| **Thinkless** | RL training with control tokens | Training required | +0% (maintains baseline) | 50-90% reduction |
| **TON** | Thought dropout + RL | Training required | +0% (maintains baseline) | Up to 90% reduction |
| **Self-consistency** | Zero-shot sampling + agreement | No training | +5-6% (estimated) | 3-4x baseline |

**Trade-offs:**
- Training approaches: More efficient at inference, but require training
- Self-consistency: No training, but higher inference cost

## Next Steps for Our Research

### Phase 1: Implement Self-Consistency Evaluator

```python
class SelfConsistencyEvaluator:
    def __init__(self, model, n_samples=3, agreement_threshold=0.67):
        self.model = model
        self.n_samples = n_samples
        self.agreement_threshold = agreement_threshold
    
    def evaluate(self, question):
        # Generate multiple answers
        answers = [self.model.generate(question, temperature=0.7) 
                   for _ in range(self.n_samples)]
        
        # Check agreement
        agreement = self.compute_agreement(answers)
        
        # Decide
        if agreement >= self.agreement_threshold:
            return self.majority_vote(answers), "baseline"
        else:
            return self.model.generate_with_metacognition(question), "metacog"
```

### Phase 2: Test on 100 Samples

**Expected results:**
- GSM8K: 81.1% → 85-86% (+4-5%)
- MMLU: 69.5% → 73-74% (+3.5-4.5%)

**If successful:** Major contribution (zero-shot selective metacognition)

### Phase 3: Compare to Training Approaches

**If self-consistency works:**
- Provides baseline for training approaches
- Shows what's achievable without training
- Validates selective metacognition concept

**If it doesn't work:**
- Confirms training is necessary
- But provides training data (agreement labels)
- Can use for confidence calibration training

## Key Takeaways

1. ✅ **Self-consistency is a proven approach** (3850+ citations)
2. ✅ **Provides objective uncertainty signal** (answer variance)
3. ✅ **No training required** (zero-shot)
4. ✅ **Can be used for selective metacognition** (apply metacog when agreement is low)
5. ⚠️ **Higher inference cost** (3-4x baseline)
6. ⚠️ **Only works for discrete answers** (not open-ended generation)

**This is the most promising zero-shot approach for selective metacognition!**
