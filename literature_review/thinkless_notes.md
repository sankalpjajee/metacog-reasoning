# Thinkless: LLM Learns When to Think

**Paper:** arXiv:2505.13379 (May 2025)  
**Authors:** Gongfan Fang, Xinyin Ma, Xinchao Wang  
**URL:** https://arxiv.org/abs/2505.13379

## Key Contribution

**Problem:** Reasoning LLMs apply elaborate reasoning to ALL queries, causing computational inefficiency when many problems have straightforward solutions.

**Question:** Can LLMs learn when to think?

**Solution:** Thinkless - a learnable framework that empowers LLMs to adaptively select between:
- `<short>` token: Concise responses (no reasoning)
- `<think>` token: Detailed reasoning (chain-of-thought)

## Methodology

### Training Approach: Reinforcement Learning

**Algorithm:** Decoupled Group Relative Policy Optimization (DeGRPO)

**Key Innovation:** Decomposes the learning objective into two components:
1. **Control token loss:** Governs selection of reasoning mode
2. **Response loss:** Improves accuracy of generated answers

**Why decoupled?**
- Enables fine-grained control over each objective
- Stabilizes training
- Prevents collapse observed in vanilla GRPO

### How It Works

1. Model learns to assess task complexity
2. Model learns its own ability on that task
3. Based on both, selects `<short>` or `<think>`
4. RL reward: Accuracy - computational cost

## Results

**Benchmarks:** Minerva Algebra, MATH-500, GSM8K

**Efficiency gains:**
- Reduces long-chain thinking usage by **50-90%**
- Significantly improves efficiency of Reasoning LLMs
- Maintains accuracy while reducing computation

## Relevance to Our Work

**This directly addresses our problem:**
- We found metacognition helps on hard questions (+8% oracle)
- But hurts on easy questions (-32% when misapplied)
- Thinkless shows LLMs CAN learn when to think via RL training

**Key insight:** The decision of "when to think" is LEARNABLE, not prompt-able.

**Approach:**
- Use control tokens (`<short>`, `<think>`)
- Train with RL (reward = accuracy - cost)
- Model learns optimal policy

**Difference from our approaches:**
- Our adaptive: Explicit classification (SIMPLE/COMPLEX) → Failed
- Our threshold: Confidence-based → Failed
- Thinkless: Learned policy via RL → Success

## Implications

1. **Explicit prompting doesn't work** (confirmed by our results)
2. **Training is necessary** (RL with control tokens)
3. **Decoupled objectives are important** (prevents training collapse)
4. **50-90% efficiency gain is possible** (while maintaining accuracy)

## Code

Available at: https://github.com/gongfan99/Thinkless (mentioned in paper)
