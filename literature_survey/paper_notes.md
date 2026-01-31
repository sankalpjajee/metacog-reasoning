# Literature Survey: Training Methods for Confidence/Uncertainty in LLMs

## Paper 1: "Large Language Models Must Be Taught to Know What They Don't Know" (NeurIPS 2024)
**Authors:** Kapoor et al. (NYU, Abacus AI, Cambridge)
**Cited by:** 80

### Key Findings:
1. **Prompting alone is insufficient** for good calibration
2. **Fine-tuning on a small dataset** of correct and incorrect answers can create uncertainty estimates with good generalization
3. **~1000 graded examples are sufficient** to outperform baseline methods
4. **Training through model features + LoRA** is necessary for good performance on large open-source models
5. Models can be used as **general-purpose uncertainty estimators** (not just for their own uncertainties but also other models)

### Method Overview:
- Create a "Graded Dataset" with questions, model answers, and correctness labels
- Fine-tune the LLM to predict confidence scores
- Use LoRA for efficient fine-tuning
- Achieves better ECE (Expected Calibration Error) and AUROC

### Key Insight for Our Work:
- We can use self-consistency agreement as a **proxy for correctness** to create training labels
- Fine-tuning with LoRA is tractable and effective
- ~1000 examples should be sufficient


### Evaluation Metrics (from paper):
- **ECE (Expected Calibration Error)**: Lower is better, measures calibration
- **AUROC**: Measures ability to distinguish correct from incorrect answers
- Use greedy decoding for answer generation
- Label answers as correct/incorrect using auxiliary LLM or ground truth

### Section 4: Black-box methods are insufficient
- For multiple choice: token probabilities can work
- For open-ended generation: black-box methods struggle
- Need fine-tuning for reliable uncertainty estimates

---


## Paper 2: "Training Language Models to Self-Correct via Reinforcement Learning" (SCoRe)
**Authors:** Kumar et al. (Google DeepMind)
**Cited by:** 323

### Key Findings:
1. **SFT alone is insufficient** for self-correction - falls prey to distribution mismatch or behavior collapse
2. **Multi-turn online RL** is needed for effective self-correction
3. Uses **entirely self-generated data** - no external supervision needed
4. Achieves 15.6% improvement on MATH, 9.1% on HumanEval

### Why SFT Fails:
- Distribution mismatch: training on mistakes from data-collection policy, but model makes different mistakes
- Behavior collapse: model learns only one mode of correction that doesn't generalize

### SCoRe Method:
1. Initial phase: multi-turn RL on base model to generate policy initialization
2. Regularization to steer learning toward effective self-correction
3. Reward bonus to amplify self-correction behavior

### Relevance for Our Work:
- Confirms that simple SFT may not be enough
- But RL is complex and expensive
- Our approach (confidence prediction) is simpler than full self-correction

---


## Paper 3: "Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification" (2025)
**Authors:** Zhang et al. (NYU)
**URL:** https://arxiv.org/abs/2504.05419

### Key Findings:
1. **Simple linear probes on hidden states** can predict answer correctness with high accuracy
2. **Highly calibrated** - ECE (Expected Calibration Error) below 0.1
3. **Look-ahead information** - correctness can be predicted BEFORE the answer is fully generated
4. **24% reduction in inference tokens** using probe as early-exit verifier

### Method:
1. Extract last-layer hidden states at last token position
2. Train 2-layer MLP probe on (hidden_state, correctness_label) pairs
3. Use weighted binary cross-entropy loss (for imbalanced data)
4. Probe outputs confidence score for early-exit decisions

### Key Insight for Our Work:
- We can train a simple probe on hidden states to predict when metacognition is needed
- Don't need complex RL - just a classifier on embeddings
- Can potentially predict BEFORE generating the full answer

---


## Paper 4: "CREST: Consistency-driven Rationale Evaluation for Self-Training" (NAACL 2025)
**Authors:** Lee et al.
**URL:** https://arxiv.org/abs/2411.06387

### Key Idea:
Use consistency as a training signal for self-training LLMs on reasoning tasks.

### Method:
1. Generate rationales and evaluate via follow-up questions
2. Filter out rationales that frequently result in incorrect answers
3. Preference learning based on consistency evaluation

### Relevance for Our Work:
- Confirms that consistency can be used as a training signal
- We can use self-consistency agreement as a similar signal
- Filter training data based on agreement rate

---

## Paper 5: "Self-Rewarded Training (SRT)" 
**URL:** https://self-rewarding-llm-training.github.io/

### Key Idea:
Use consistency across multiple model-generated solutions to estimate correctness during RL training.

### Method:
- No labeled data needed - uses self-supervision
- Consistency = proxy for correctness
- RL training with consistency-based rewards

### Relevance for Our Work:
- Direct precedent for using self-consistency as training signal
- Can adapt this for our confidence predictor training

---

