# Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations

**Paper:** arXiv:2505.13763 (May 2025)  
**Authors:** Li Ji-An, Hua-Dong Xiong, Robert C. Wilson, Marcelo G. Mattar, Marcus K. Benna  
**URL:** https://arxiv.org/abs/2505.13763

## Key Contribution

**Question:** Do LLMs have metacognitive abilities to monitor and control their own internal processes?

**Finding:** LLMs can sometimes report the strategies they use, but at other times cannot recognize those strategies.

**Implication:** Limited degree of metacognition - capacity to monitor one's own cognitive processes.

## Methodology

### Neuroscience-Inspired Neurofeedback Paradigm:

**Uses in-context learning to quantify metacognitive abilities:**

1. **Monitoring task:** Can LLM report its internal activation patterns?
2. **Control task:** Can LLM manipulate its internal activation patterns?

**Key insight:** Test whether LLMs can access and report their own neural activations.

## Key Findings

### 1. Metacognitive Abilities Depend on Several Factors:

| Factor | Impact on Metacognition |
|--------|-------------------------|
| **Number of in-context examples** | More examples → Better monitoring |
| **Semantic interpretability** | Interpretable directions → Easier to report |
| **Variance explained** | High-variance directions → More accessible |

### 2. "Metacognitive Space" is Low-Dimensional:

**Critical finding:** LLMs can monitor only a small subset of their neural activations.

**Interpretation:**
- Neural space: High-dimensional (thousands of dimensions)
- Metacognitive space: Low-dimensional (small subset)
- **LLMs have limited self-awareness**

### 3. Implications for AI Safety:

**Concerns:**
- Models may obfuscate their internal processes
- Can evade neural-activation-based oversight
- Safety detectors may be circumvented

**Opportunities:**
- Quantify metacognitive abilities
- Design better oversight mechanisms
- Understand limits of self-monitoring

## Relevance to Our Work

### Connection to Selective Metacognition:

**Our problem:** LLMs can't reliably identify when they need metacognition

**This paper explains why:**
- LLMs have LIMITED metacognitive abilities
- Can only monitor small subset of activations
- Semantic interpretability matters

### Key Insights:

1. ✅ **LLMs have some metacognitive capacity** (not zero)
2. ❌ **But it's limited to low-dimensional subspace** (not full self-awareness)
3. ⚠️ **Depends on interpretability** (some things easier to monitor than others)

### Implications for Our Approaches:

| Approach | Requires | Feasibility |
|----------|----------|-------------|
| **Confidence prompting** | Self-report confidence | ❌ May be outside metacognitive space |
| **SIMPLE/COMPLEX classification** | Self-assess difficulty | ❌ May not be semantically interpretable |
| **Self-consistency** | Generate multiple answers | ✅ Doesn't require self-monitoring |
| **Training-based** | Learn to monitor | ✅ Can expand metacognitive space |

## Understanding Probability Distributions

### How LLMs Work:

**LLMs model a probability distribution over tokens:**
```
P(token_t | token_1, ..., token_{t-1}, context)
```

**At each step:**
1. Compute logits for all possible next tokens
2. Apply softmax to get probability distribution
3. Sample or select token based on distribution

### Uncertainty in Probability Distributions:

**Two types of uncertainty:**

1. **Aleatoric (data) uncertainty:**
   - Inherent randomness in the data
   - Multiple valid answers exist
   - Example: "What's a good restaurant?" (subjective)

2. **Epistemic (model) uncertainty:**
   - Model doesn't know the answer
   - Lack of knowledge or training data
   - Example: "What's the capital of Tuvalu?" (model unsure)

### How to Measure Uncertainty:

**Option 1: Token-level probability**
```python
# High probability → High confidence
max_prob = max(P(token_t | context))

if max_prob > 0.9:
    # Model is confident
else:
    # Model is uncertain
```

**Problem:** Models are often overconfident (high prob but wrong answer)

**Option 2: Entropy**
```python
# Low entropy → High confidence (peaked distribution)
# High entropy → Low confidence (flat distribution)
entropy = -sum(p * log(p) for p in distribution)
```

**Problem:** Doesn't distinguish aleatoric vs epistemic uncertainty

**Option 3: Self-consistency (our approach)**
```python
# Sample multiple times
answers = [sample(P(answer | context)) for _ in range(N)]

# Check agreement
if all_agree(answers):
    # High confidence (low epistemic uncertainty)
else:
    # Low confidence (high epistemic uncertainty)
```

**Advantage:** Directly measures epistemic uncertainty

### Why Self-Consistency Works for Metacognition:

**Probability distribution perspective:**

**When model is uncertain (epistemic):**
- Multiple modes in P(answer | question)
- Sampling produces different answers
- Self-consistency detects this

**When model is confident:**
- Single dominant mode in P(answer | question)
- Sampling produces same answer
- Self-consistency confirms this

**Key insight:** Self-consistency approximates the marginal distribution P(answer | question) by sampling from P(reasoning | question).

## Metacognition and Probability Distributions

### Question: How does metacognition relate to probability distributions?

**Answer:** Metacognition is about monitoring the SHAPE of the probability distribution.

### Ideal Metacognitive Model:

```python
# Metacognitive monitoring:
def should_think_carefully(question):
    # Examine P(answer | question)
    distribution = model.get_distribution(question)
    
    # Check shape
    entropy = compute_entropy(distribution)
    max_prob = max(distribution)
    
    # Decision
    if entropy > threshold or max_prob < threshold:
        return True  # Uncertain → Think carefully
    else:
        return False  # Confident → Answer directly
```

**Problem:** LLMs can't directly access P(answer | question) before generating!

**Why:** 
- LLMs generate token-by-token
- P(answer | question) requires generating full answer
- Can't assess uncertainty before solving

### This Explains Our Failures:

| Approach | What it asks | Why it fails |
|----------|--------------|--------------|
| **Confidence prompting** | "Rate your confidence before solving" | Can't access P(answer) before generating |
| **SIMPLE/COMPLEX** | "Is this question hard?" | Difficulty ≠ model's uncertainty |
| **Threshold** | "Generate initial answer + confidence" | Model generates, THEN reports confidence (not before) |

### Why Self-Consistency Works:

**It approximates P(answer | question) by sampling:**

```python
# Self-consistency approximates the distribution
samples = [model.generate(question) for _ in range(N)]

# Empirical distribution
P_empirical(answer) = count(answer in samples) / N

# Uncertainty measure
uncertainty = 1 - max(P_empirical(answer))
```

**Key advantage:** Doesn't require model to self-report uncertainty!

## Training-Based Approaches

### How Training Can Expand Metacognitive Space:

**Current state (from paper):**
- Metacognitive space is low-dimensional
- LLMs can only monitor small subset of activations

**With training:**
- Teach model to recognize patterns associated with uncertainty
- Expand the semantically interpretable directions
- Learn to monitor task-relevant activations

### Example: Confidence Calibration Training

```python
# Training objective:
1. Generate answer
2. Predict P(correct | answer, question)
3. Compare with actual correctness
4. Update model to improve calibration

# After training:
- Model learns to recognize when it's uncertain
- Expands metacognitive space to include "correctness" dimension
- Can self-assess before committing to answer
```

### Why This Works:

**Training creates new semantically interpretable directions:**
- "Correctness likelihood" becomes a monitored dimension
- Model learns internal signals that correlate with being wrong
- Metacognitive space expands to include these signals

## Synthesis: Why Our Approaches Failed/Succeeded

### Failed Approaches:

1. **Adaptive (SIMPLE/COMPLEX):**
   - Requires monitoring "problem difficulty"
   - May not be in metacognitive space
   - Model can't reliably access this dimension

2. **Threshold (confidence):**
   - Requires monitoring "my confidence"
   - Model generates answer first, then reports
   - Not true metacognitive monitoring (post-hoc rationalization)

### Successful Approaches:

1. **Self-consistency:**
   - Doesn't require metacognitive monitoring
   - Samples from distribution directly
   - Objective measure of uncertainty

2. **Training-based (Thinkless, TON):**
   - Expands metacognitive space through learning
   - Creates new monitored dimensions
   - Model learns when to think

### Oracle (targeted):
   - Proves metacognition helps (+8%)
   - But requires knowing answer (impractical)
   - Shows upper bound of what's achievable

## Key Takeaways

1. ✅ **LLMs have limited metacognitive abilities** (low-dimensional space)
2. ✅ **Can't monitor all internal states** (only semantically interpretable directions)
3. ✅ **Self-consistency bypasses this limitation** (samples distribution directly)
4. ✅ **Training can expand metacognitive space** (learn new monitored dimensions)
5. ⚠️ **Explicit prompting hits fundamental limits** (can't access what's not in metacognitive space)

## Implications for Our Research

### Short-term (Zero-shot):

**Implement self-consistency:**
- Doesn't require metacognitive monitoring
- Objective uncertainty signal
- Can test immediately

### Long-term (Training):

**Train metacognitive monitoring:**
- Expand metacognitive space
- Learn to recognize uncertainty patterns
- Create semantically interpretable "need to think" dimension

### Paper Contribution:

**We can cite this paper to explain:**
- Why explicit prompting failed (limited metacognitive space)
- Why self-consistency works (bypasses monitoring)
- Why training is necessary (expands metacognitive space)

**Our contribution:**
- Empirically demonstrate the limits
- Show self-consistency as zero-shot solution
- Propose training to expand metacognitive abilities
