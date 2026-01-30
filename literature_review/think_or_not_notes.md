# Think or Not? Selective Reasoning via Reinforcement Learning for Vision-Language Models

**Paper:** arXiv:2505.16854 (May 2025)  
**Authors:** Jiaqi Wang, Kevin Qinghong Lin, James Cheng, Mike Zheng Shou  
**URL:** https://arxiv.org/abs/2505.16854

## Key Contribution

**Inspiration:** Human-like thinking process
- People skip reasoning for easy questions
- Think carefully when needed

**Goal:** Enable VLMs to first decide when reasoning is necessary

**Solution:** TON (Think or Not) - a two-stage training strategy

## Methodology

### Stage 1: Supervised Fine-Tuning (SFT) with "Thought Dropout"

**Key innovation:** Thought dropout operation
- Reasoning traces are randomly replaced with empty thoughts
- Introduces "think-or-not" format
- Serves as cold start for selective reasoning

**Why this works:**
- Model learns that sometimes reasoning is not needed
- Creates format for selective reasoning
- Prepares model for RL stage

### Stage 2: GRPO (Group Relative Policy Optimization)

**Training objective:**
- Model freely explores when to think or not
- Maximizes task-aware outcome rewards
- Reward = accuracy - computational cost

**Result:**
- Model learns optimal policy
- Progressively bypasses unnecessary reasoning
- Maintains or improves performance

## Results

**Efficiency gains:**
- Reduces completion length by up to **90%** compared to vanilla GRPO
- No performance sacrifice (or even improvement)

**Benchmarks tested:**
- LLM: GSM8K
- VLM: CLEVR, Super-CLEVR, GeoQA
- Agentic: AITZ

**Model sizes:** 3B and 7B

**Key finding:** Model progressively learns to bypass unnecessary reasoning steps as training advances

## Relevance to Our Work

**This directly solves our problem:**
- We found explicit prompting fails (adaptive, threshold)
- TON shows training-based approach works

**Key differences:**

| Approach | Method | Result |
|----------|--------|--------|
| Our adaptive | Explicit SIMPLE/COMPLEX classification | Failed (-32%) |
| Our threshold | Confidence-based prompting | Failed (-25%) |
| TON | Thought dropout + RL training | Success (-90% tokens, same accuracy) |

## Technical Insights

### 1. Thought Dropout (SFT Stage)

**Purpose:** Create format for selective reasoning

**Implementation:**
```
Training data:
- 50% of examples: Full reasoning traces
- 50% of examples: Empty thoughts (direct answer)

Model learns:
- Both formats are valid
- Can choose either based on context
```

**Why this is important:**
- Without this, model always generates reasoning (default behavior)
- Thought dropout teaches model that skipping is an option
- Cold start for RL exploration

### 2. RL Training (GRPO Stage)

**Reward function:**
```
Reward = Accuracy - λ * (Token_count)
```

**Where:**
- Accuracy: Task performance
- Token_count: Computational cost
- λ: Trade-off parameter

**Model learns:**
- When reasoning improves accuracy (worth the cost)
- When reasoning doesn't help (skip it)
- Optimal policy for selective reasoning

### 3. Progressive Learning

**Observation:** Model progressively learns to bypass unnecessary reasoning as training advances

**Interpretation:**
- Early training: Model explores both options
- Mid training: Model starts identifying easy questions
- Late training: Model confidently skips easy questions

**This is exactly what we need!**

## Comparison to Thinkless

| Aspect | Thinkless | TON |
|--------|-----------|-----|
| **Approach** | Control tokens (`<short>`, `<think>`) | Thought dropout + RL |
| **Training** | DeGRPO (decoupled objectives) | SFT + GRPO |
| **Cold start** | Not specified | Thought dropout in SFT |
| **Efficiency** | 50-90% reduction | Up to 90% reduction |
| **Domain** | Language models | Vision-language models |

**Both approaches:**
- Use RL training
- Learn when to think
- Achieve 50-90% efficiency gains
- Maintain or improve accuracy

## Implications for Our Research

### 1. Training is necessary

**Evidence:**
- Our prompting approaches failed
- TON and Thinkless succeed with training
- Conclusion: Selective reasoning requires learning

### 2. Thought dropout is key

**Insight:**
- Model needs to learn that skipping is valid
- Can't discover this through prompting alone
- SFT with thought dropout creates the format

### 3. RL enables optimal policy

**Why RL:**
- Balances accuracy vs. cost
- Learns from exploration
- Discovers optimal policy

### 4. Progressive learning is natural

**Observation:**
- Model doesn't immediately know when to skip
- Learns gradually through training
- Eventually develops reliable policy

## Implementation Plan for Our Work

### Phase 1: Prepare Training Data

```python
# For each question in dataset:
1. Baseline answer (no reasoning)
2. Metacognitive answer (full reasoning)
3. Label: baseline_correct, metacog_correct

# Create training data:
- If baseline_correct: 70% baseline, 30% metacog
- If !baseline_correct && metacog_correct: 100% metacog
```

### Phase 2: SFT with Thought Dropout

```python
# Training format:
<question>
<thought> [empty or full metacognitive reasoning] </thought>
<answer> [final answer] </answer>

# Thought dropout:
- 50% of examples: Full reasoning
- 50% of examples: Empty (direct answer)
```

### Phase 3: RL Training

```python
# Reward function:
Reward = Accuracy - λ * (Reasoning_tokens / Total_tokens)

# Where:
- Accuracy: 1 if correct, 0 if wrong
- λ: Cost parameter (tune based on efficiency goals)
```

## Code

Available at: https://github.com/xxx (mentioned in paper)
