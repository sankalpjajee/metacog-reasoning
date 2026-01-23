# Metacognitive Self-Play Training: Design Document

**Paper Title**: *Metacognitive Self-Play: Training Language Models to Monitor and Control Their Own Reasoning*

**Target Venue**: NeurIPS 2026

**Date**: January 23, 2026

---

## 1. Core Concept

**Metacognitive Self-Play** is a novel training method that explicitly teaches language models to **monitor** and **control** their own reasoning processes, grounded in Nelson & Narens' (1990) metacognition framework.

### Key Differentiation

| Approach | Metacognition Type | Training Signal |
|----------|-------------------|-----------------|
| **DeepSeek-R1** | Emergent (implicit) | RL on final outcomes only |
| **SPIN/SPPO** | None | Self-play on solution quality |
| **Our Method** | Explicit (trained) | Self-play on metacognitive quality |

### Novel Contribution

We train models to:
1. **Monitor**: Assess confidence, detect errors, identify uncertainty
2. **Control**: Adjust reasoning strategies, backtrack, verify solutions
3. **Self-improve**: Generate preference pairs based on metacognitive quality

---

## 2. Training Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Problem (e.g., GSM8K)           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Generate Multiple Metacognitive Traces            │
│  - Sample N solutions with explicit metacognitive markers   │
│  - Each trace includes:                                     │
│    • Reasoning steps                                        │
│    • Confidence assessments                                 │
│    • Error detection attempts                               │
│    • Strategy adjustments                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Metacognitive Scoring                             │
│  - Correctness: Is final answer correct?                    │
│  - Monitoring Quality: Are confidence assessments accurate? │
│  - Control Quality: Are strategy adjustments effective?     │
│  - Overall Score: Weighted combination                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: Preference Pair Creation                          │
│  - Rank traces by metacognitive score                       │
│  - Create (chosen, rejected) pairs                          │
│  - Chosen: Higher metacognitive quality                     │
│  - Rejected: Lower metacognitive quality                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: DPO Training                                       │
│  - Train on preference pairs                                │
│  - Loss: DPO loss on (chosen, rejected) pairs               │
│  - Update model to prefer high-quality metacognition        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5: Iterate                                            │
│  - Use updated model to generate new traces                 │
│  - Repeat cycle for multiple iterations                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Metacognitive Prompt Design

### 3.1 Training Prompt Template

```
You are a careful problem solver. For each problem, think step-by-step and explicitly:
1. State your confidence level after each reasoning step
2. Check for potential errors in your reasoning
3. Adjust your strategy if needed
4. Verify your final answer

Problem: {problem}

Think carefully and show your metacognitive reasoning:
```

### 3.2 Metacognitive Markers

The model should learn to produce traces with explicit markers:

- **Confidence**: "I'm [confident/uncertain/very unsure] about this step because..."
- **Error Detection**: "Wait, let me check if this is correct... [verification]"
- **Strategy Adjustment**: "This approach isn't working, let me try [alternative]"
- **Verification**: "Let me verify: [checking work]"

### 3.3 Example Metacognitive Trace

```
Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins 
with four. She sells the remainder at the farmers' market for $2 per egg. How much does 
she make every day?

Metacognitive Reasoning:
Step 1: Calculate total eggs consumed
- Janet eats 3 eggs for breakfast
- She uses 4 eggs for muffins
- Total consumed: 3 + 4 = 7 eggs
[Confidence: High - this is straightforward arithmetic]

Step 2: Calculate eggs sold
- Total laid: 16 eggs
- Total consumed: 7 eggs
- Eggs sold: 16 - 7 = 9 eggs
[Confidence: High - simple subtraction]

Step 3: Calculate revenue
- Price per egg: $2
- Eggs sold: 9
- Revenue: 9 × $2 = $18
[Confidence: High - direct multiplication]

[Verification: Let me double-check: 3 + 4 = 7 consumed, 16 - 7 = 9 sold, 9 × 2 = 18. ✓]

Answer: $18
```

---

## 4. Metacognitive Scoring Function

### 4.1 Score Components

**Correctness Score (C)**: 0 or 1
- 1 if final answer is correct
- 0 if final answer is incorrect

**Monitoring Score (M)**: 0 to 1
- Measures quality of confidence assessments
- High score if:
  - High confidence → correct steps
  - Low confidence → incorrect steps or uncertain situations
  - Includes explicit error checking

**Control Score (K)**: 0 to 1
- Measures quality of strategy adjustments
- High score if:
  - Detects errors and corrects them
  - Adjusts strategy when stuck
  - Verifies final answer

### 4.2 Overall Metacognitive Score

```
MetacogScore = α·C + β·M + γ·K

Where:
- α = 0.5 (correctness weight)
- β = 0.25 (monitoring weight)
- γ = 0.25 (control weight)
```

### 4.3 Monitoring Score Calculation

```python
def calculate_monitoring_score(trace, is_correct):
    """
    Analyze confidence markers and their alignment with correctness
    """
    confidence_markers = extract_confidence_markers(trace)
    
    # Check if confidence aligns with correctness
    if is_correct and has_high_confidence(confidence_markers):
        return 1.0
    elif not is_correct and has_low_confidence(confidence_markers):
        return 0.8  # Good monitoring, but wrong answer
    elif is_correct and has_low_confidence(confidence_markers):
        return 0.6  # Correct but underconfident
    else:  # not correct and high confidence
        return 0.0  # Poor monitoring
```

### 4.4 Control Score Calculation

```python
def calculate_control_score(trace, is_correct):
    """
    Analyze strategy adjustments and error corrections
    """
    has_verification = "verify" in trace.lower() or "check" in trace.lower()
    has_error_detection = "wait" in trace.lower() or "error" in trace.lower()
    has_strategy_adjustment = "try" in trace.lower() or "instead" in trace.lower()
    
    control_score = 0.0
    
    if has_verification:
        control_score += 0.4
    if has_error_detection:
        control_score += 0.3
    if has_strategy_adjustment:
        control_score += 0.3
    
    # Bonus if correct
    if is_correct:
        control_score *= 1.2
    
    return min(control_score, 1.0)
```

---

## 5. Preference Pair Creation

### 5.1 Pairing Strategy

For each problem, generate N traces (e.g., N=4):

1. Score all N traces using metacognitive scoring
2. Rank traces by score
3. Create preference pairs:
   - **Chosen**: Top 50% traces
   - **Rejected**: Bottom 50% traces
4. Create all possible (chosen, rejected) pairs

### 5.2 Example

```
Problem: [GSM8K problem]

Generated Traces:
- Trace A: MetacogScore = 0.85 (correct, high monitoring, good control)
- Trace B: MetacogScore = 0.75 (correct, medium monitoring, some control)
- Trace C: MetacogScore = 0.40 (correct, no monitoring, no control)
- Trace D: MetacogScore = 0.20 (incorrect, overconfident, no control)

Preference Pairs:
1. (Trace A, Trace C): Prefer explicit metacognition over implicit
2. (Trace A, Trace D): Prefer correct + metacognitive over incorrect
3. (Trace B, Trace C): Prefer monitoring over no monitoring
4. (Trace B, Trace D): Prefer correct + some metacog over incorrect
```

---

## 6. DPO Training Details

### 6.1 Training Objective

Use Direct Preference Optimization (DPO) loss:

```
L_DPO = -log σ(β · [log π_θ(y_w|x) - log π_ref(y_w|x) 
                    - log π_θ(y_l|x) + log π_ref(y_l|x)])

Where:
- y_w: chosen (winner) trace
- y_l: rejected (loser) trace
- π_θ: policy being trained
- π_ref: reference policy (frozen)
- β: temperature parameter
```

### 6.2 Training Hyperparameters

```python
training_config = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "num_traces_per_problem": 4,
    "num_iterations": 3,
    "problems_per_iteration": 5000,  # From GSM8K training set
    
    # DPO parameters
    "beta": 0.1,
    "learning_rate": 5e-7,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_length": 2048,
    "num_epochs": 1,
    
    # Metacognitive scoring weights
    "alpha": 0.5,  # correctness
    "beta_score": 0.25,  # monitoring
    "gamma": 0.25,  # control
}
```

### 6.3 Training Data

**Iteration 1**: Use GSM8K training set (7,473 problems)
- Sample 5,000 problems
- Generate 4 traces per problem → 20,000 traces
- Create ~10,000 preference pairs

**Iteration 2**: Use updated model
- Generate new traces on same 5,000 problems
- Model should produce better metacognitive traces

**Iteration 3**: Final iteration
- Further refinement

---

## 7. Evaluation Plan

### 7.1 Primary Metrics

**Task Performance**:
- GSM8K accuracy (baseline: 81.1%)
- MMLU accuracy (baseline: 69.5%)
- HellaSwag accuracy (baseline: 64.1%)
- MR-Ben accuracy (baseline: 28.8%)

**Metacognitive Quality**:
- Monitoring score (average across test set)
- Control score (average across test set)
- Calibration error (confidence vs. correctness alignment)

### 7.2 Ablation Studies

1. **No Monitoring**: Train without monitoring markers
2. **No Control**: Train without control markers
3. **Correctness Only**: Train on correctness alone (like SPIN)
4. **Emergent Baseline**: Train with RL on outcomes only (like DeepSeek-R1)

### 7.3 Human Evaluation

Sample 100 problems and have human annotators rate:
- Reasoning quality (1-5 scale)
- Metacognitive clarity (1-5 scale)
- Trustworthiness (1-5 scale)

---

## 8. Implementation Roadmap

### Phase 1: Data Generation (Week 1)
- [ ] Implement metacognitive prompt template
- [ ] Implement trace generation with sampling
- [ ] Generate 4 traces per problem for 5,000 GSM8K problems
- [ ] Save traces to disk

### Phase 2: Scoring (Week 1-2)
- [ ] Implement correctness scoring
- [ ] Implement monitoring score calculation
- [ ] Implement control score calculation
- [ ] Score all generated traces

### Phase 3: Preference Pairs (Week 2)
- [ ] Implement preference pair creation
- [ ] Create training dataset in DPO format
- [ ] Validate pair quality

### Phase 4: Training (Week 2-3)
- [ ] Set up DPO training with TRL library
- [ ] Train iteration 1
- [ ] Evaluate on dev set
- [ ] Generate new traces with updated model
- [ ] Train iterations 2-3

### Phase 5: Evaluation (Week 3-4)
- [ ] Run full benchmark evaluation
- [ ] Calculate metacognitive quality metrics
- [ ] Run ablation studies
- [ ] Conduct human evaluation

### Phase 6: Paper Writing (Week 4-6)
- [ ] Write methodology section
- [ ] Create figures and tables
- [ ] Write results section
- [ ] Write discussion and related work
- [ ] Prepare submission

---

## 9. Expected Outcomes

### 9.1 Hypotheses

**H1**: Explicit metacognitive training improves task performance
- Expected: +3-5% on GSM8K, +2-3% on MMLU

**H2**: Metacognitive quality is measurable and improves with training
- Expected: Monitoring score increases from ~0.3 → 0.7
- Expected: Control score increases from ~0.2 → 0.6

**H3**: Metacognitive training improves calibration
- Expected: Calibration error decreases by 20-30%

**H4**: Explicit training outperforms emergent approaches
- Expected: Better metacognitive quality than DeepSeek-R1 style training

### 9.2 Potential Challenges

1. **Trace Generation Cost**: 20,000 traces × 2048 tokens = expensive
   - Mitigation: Use smaller subset initially, scale up if promising

2. **Scoring Reliability**: Heuristic-based scoring may be noisy
   - Mitigation: Validate scoring with human annotations on subset

3. **Training Instability**: DPO can be unstable
   - Mitigation: Careful hyperparameter tuning, use smaller learning rate

4. **Overfitting to Markers**: Model may learn to produce markers without genuine metacognition
   - Mitigation: Evaluate on diverse benchmarks, include human evaluation

---

## 10. Next Steps

**Immediate**:
1. Implement trace generation script
2. Test on 10 problems to validate prompt design
3. Implement scoring functions
4. Generate small dataset (100 problems) for validation

**This Week**:
1. Generate full training dataset (5,000 problems)
2. Implement DPO training pipeline
3. Run iteration 1 training

**Next Week**:
1. Evaluate iteration 1 model
2. Run iterations 2-3
3. Begin full evaluation

---

## References

- Nelson, T. O., & Narens, L. (1990). Metamemory: A theoretical framework and new findings.
- Rafailov, R., et al. (2023). Direct Preference Optimization.
- Chen, Z., et al. (2024). Self-Play Fine-Tuning (SPIN).
- DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs.
- Griot, M., et al. (2025). Large language models lack metacognition. *Nature*.
