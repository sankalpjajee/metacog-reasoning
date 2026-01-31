# How to Maximize Error Prediction Probe Accuracy

## The Challenge

**Goal:** Train a probe that predicts `P(baseline will be wrong | question)`

**Why it's hard:**
- Baseline accuracy is ~70-80% (class imbalance)
- Errors are diverse (math mistakes, knowledge gaps, reasoning failures)
- Hidden states are high-dimensional (4096 dims)
- Limited training data (2000-6000 samples)

**Target accuracy:** 75-85% (to be useful)

---

## Strategy 1: Architecture Choices

### Option A: Simple Linear Probe (Baseline)

```python
class LinearProbe(nn.Module):
    def __init__(self, hidden_dim=4096):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden_state):
        return torch.sigmoid(self.linear(hidden_state))
```

**Pros:** Fast, interpretable, less prone to overfitting
**Cons:** May not capture complex patterns
**Expected accuracy:** 65-70%

### Option B: 2-Layer MLP (Recommended)

```python
class MLPProbe(nn.Module):
    def __init__(self, hidden_dim=4096, intermediate_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(intermediate_dim, 1)
    
    def forward(self, hidden_state):
        x = self.fc1(hidden_state)
        x = self.relu(x)
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))
```

**Pros:** More expressive, can learn non-linear patterns
**Cons:** More parameters, needs regularization
**Expected accuracy:** 70-75%

### Option C: Attention-Based Probe (Advanced)

```python
class AttentionProbe(nn.Module):
    def __init__(self, hidden_dim=4096, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden_state):
        # Self-attention over hidden state
        attn_output, _ = self.attention(hidden_state, hidden_state, hidden_state)
        return torch.sigmoid(self.fc(attn_output.mean(dim=0)))
```

**Pros:** Can focus on relevant parts of hidden state
**Cons:** More complex, needs more data
**Expected accuracy:** 75-80%

### Option D: Ensemble of Probes (Best Performance)

```python
class EnsembleProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_probe = LinearProbe()
        self.mlp_probe = MLPProbe()
        self.attention_probe = AttentionProbe()
        self.combiner = nn.Linear(3, 1)
    
    def forward(self, hidden_state):
        p1 = self.linear_probe(hidden_state)
        p2 = self.mlp_probe(hidden_state)
        p3 = self.attention_probe(hidden_state)
        combined = torch.cat([p1, p2, p3], dim=-1)
        return torch.sigmoid(self.combiner(combined))
```

**Pros:** Best accuracy, robust
**Cons:** Slower, more complex
**Expected accuracy:** 75-85%

---

## Strategy 2: Feature Engineering

### Use Multiple Hidden States

Instead of just the last token's hidden state, use:

```python
# Extract hidden states from multiple positions
hidden_states = []
for layer in [8, 16, 24, 32]:  # Different layers
    hidden_states.append(model.get_hidden_state(question, layer=layer))

# Concatenate or pool
combined_hidden = torch.cat(hidden_states, dim=-1)  # Or mean/max pool
```

**Why:** Different layers capture different information
- Early layers: Syntax, basic semantics
- Middle layers: Reasoning, world knowledge
- Late layers: Task-specific features

**Expected improvement:** +3-5% accuracy

### Add Explicit Features

Augment hidden states with hand-crafted features:

```python
features = {
    'question_length': len(question.split()),
    'has_numbers': int(bool(re.search(r'\d', question))),
    'num_options': 4,  # For multiple choice
    'question_type': encode_type(question),  # Math, knowledge, commonsense
    'baseline_confidence': get_logit_confidence(baseline_output),
    'baseline_entropy': calculate_entropy(baseline_logits),
}

# Concatenate with hidden state
combined_input = torch.cat([hidden_state, feature_vector], dim=-1)
```

**Expected improvement:** +2-4% accuracy

### Use Baseline Output Features

The baseline's output contains useful signals:

```python
# Get baseline generation with logits
baseline_output, logits = model.generate(question, return_logits=True)

features = {
    'max_logit': logits.max().item(),
    'entropy': -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum().item(),
    'top1_prob': logits.softmax(dim=-1).max().item(),
    'top1_top2_gap': logits.softmax(dim=-1).topk(2)[0].diff().item(),
}
```

**Why:** Low confidence / high entropy often correlates with errors

**Expected improvement:** +5-8% accuracy

---

## Strategy 3: Training Techniques

### 1. Weighted Loss for Class Imbalance

```python
# Calculate class weights
pos_weight = (num_correct / num_wrong)  # e.g., 3.0 if 75% correct, 25% wrong

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
```

**Why:** Prevents probe from just predicting "correct" for everything

### 2. Focal Loss for Hard Examples

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

**Why:** Focuses training on hard-to-classify examples

**Expected improvement:** +2-3% accuracy

### 3. Data Augmentation

```python
# Generate multiple baseline samples per question
for question in training_set:
    for seed in range(5):  # 5 different random seeds
        baseline_answer = model.generate(question, seed=seed)
        is_correct = (baseline_answer == ground_truth)
        hidden_state = model.get_hidden_state(question, seed=seed)
        training_data.append((hidden_state, is_correct))
```

**Why:** Increases training data 5x, captures variance in baseline behavior

**Expected improvement:** +3-5% accuracy

### 4. Curriculum Learning

```python
# Start with easy examples, gradually add harder ones
easy_examples = [d for d in training_data if d['is_obvious_error']]
hard_examples = [d for d in training_data if not d['is_obvious_error']]

# Train in stages
train_on(easy_examples, epochs=5)
train_on(easy_examples + hard_examples[:len(hard_examples)//2], epochs=5)
train_on(all_examples, epochs=10)
```

**Why:** Helps probe learn basic patterns before tackling edge cases

**Expected improvement:** +2-3% accuracy

### 5. Ensemble Training with Bagging

```python
# Train multiple probes on bootstrap samples
probes = []
for i in range(5):
    bootstrap_sample = random.sample(training_data, len(training_data))
    probe = train_probe(bootstrap_sample)
    probes.append(probe)

# At test time, average predictions
def predict(question):
    predictions = [probe(question) for probe in probes]
    return np.mean(predictions)
```

**Why:** Reduces overfitting, improves generalization

**Expected improvement:** +3-5% accuracy

---

## Strategy 4: Multi-Task Learning

### Train on Multiple Objectives

```python
class MultiTaskProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4096, 256)
        self.error_head = nn.Linear(256, 1)  # Predict if wrong
        self.confidence_head = nn.Linear(256, 1)  # Predict confidence
        self.difficulty_head = nn.Linear(256, 1)  # Predict difficulty
    
    def forward(self, hidden_state):
        shared_repr = self.shared(hidden_state)
        error_pred = self.error_head(shared_repr)
        confidence_pred = self.confidence_head(shared_repr)
        difficulty_pred = self.difficulty_head(shared_repr)
        return error_pred, confidence_pred, difficulty_pred
```

**Training:**
```python
loss = (
    bce_loss(error_pred, is_wrong) +
    mse_loss(confidence_pred, baseline_confidence) +
    mse_loss(difficulty_pred, human_difficulty_rating)
)
```

**Why:** Auxiliary tasks provide additional supervision signal

**Expected improvement:** +4-6% accuracy

---

## Strategy 5: Use Self-Consistency as a Feature

Don't throw away self-consistency—use it as an input feature!

```python
# Generate 3 baseline samples
baseline_samples = [model.generate(question, temp=0.7) for _ in range(3)]
agreement_rate = calculate_agreement(baseline_samples)

# Use as feature
features = {
    'hidden_state': hidden_state,
    'agreement_rate': agreement_rate,
    'majority_answer': get_majority(baseline_samples),
}
```

**Why:** Self-consistency is informative, just not sufficient alone

**Expected improvement:** +5-7% accuracy

---

## Strategy 6: Task-Specific Probes

Train separate probes for each task type:

```python
probes = {
    'gsm8k': train_probe(gsm8k_data),
    'mmlu': train_probe(mmlu_data),
    'hellaswag': train_probe(hellaswag_data),
}

# At test time
def predict(question, task_type):
    probe = probes[task_type]
    return probe(question)
```

**Why:** Error patterns differ across tasks

**Expected improvement:** +3-5% accuracy per task

---

## Strategy 7: Active Learning

Iteratively improve the probe:

```python
# Round 1: Train on initial data
probe_v1 = train_probe(initial_data)

# Round 2: Find uncertain predictions
uncertain_questions = [q for q in unlabeled_data 
                       if 0.4 < probe_v1(q) < 0.6]

# Label these questions (run baseline and check)
new_labels = label_questions(uncertain_questions)

# Round 3: Retrain on expanded data
probe_v2 = train_probe(initial_data + new_labels)
```

**Why:** Focuses labeling effort on informative examples

**Expected improvement:** +2-4% accuracy

---

## Strategy 8: Use Contrastive Learning

Learn representations that separate correct from incorrect:

```python
class ContrastiveProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4096, 128)
        self.classifier = nn.Linear(128, 1)
    
    def forward(self, hidden_state):
        embedding = self.encoder(hidden_state)
        return self.classifier(embedding), embedding

# Contrastive loss
def contrastive_loss(emb1, emb2, label):
    distance = F.pairwise_distance(emb1, emb2)
    loss = label * distance.pow(2) + (1 - label) * F.relu(margin - distance).pow(2)
    return loss.mean()

# Train
for (h1, l1), (h2, l2) in pairs:
    pred1, emb1 = model(h1)
    pred2, emb2 = model(h2)
    loss = bce_loss(pred1, l1) + bce_loss(pred2, l2) + contrastive_loss(emb1, emb2, l1 == l2)
```

**Why:** Learns better representations by contrasting correct vs incorrect

**Expected improvement:** +3-5% accuracy

---

## Recommended Combination

**For maximum accuracy, combine multiple strategies:**

### Architecture
- 2-layer MLP probe (Strategy 1B)
- Ensemble of 3-5 probes (Strategy 1D)

### Features
- Multi-layer hidden states (Strategy 2)
- Baseline confidence/entropy (Strategy 2)
- Self-consistency agreement (Strategy 5)

### Training
- Weighted focal loss (Strategy 3)
- Data augmentation with multiple seeds (Strategy 3)
- Multi-task learning (Strategy 4)

### Expected Combined Accuracy: 80-85%

---

## Evaluation Metrics

Don't just look at accuracy! Track:

```python
metrics = {
    'accuracy': (TP + TN) / (TP + TN + FP + FN),
    'precision': TP / (TP + FP),  # When probe says "wrong", how often is it right?
    'recall': TP / (TP + FN),  # Of all errors, how many does probe catch?
    'f1_score': 2 * (precision * recall) / (precision + recall),
    'auc_roc': roc_auc_score(labels, predictions),
    'calibration_error': expected_calibration_error(predictions, labels),
}
```

**Key metrics for our use case:**
- **Precision:** High precision means we don't waste metacog on correct answers
- **Recall:** High recall means we catch most errors
- **F1:** Balance between precision and recall

**Target:** F1 > 0.75, AUC > 0.85

---

## Debugging Low Accuracy

If probe accuracy is < 70%, check:

1. **Class imbalance:** Are you using weighted loss?
2. **Overfitting:** Try more regularization (dropout, weight decay)
3. **Underfitting:** Try deeper model or more features
4. **Data quality:** Are labels correct? Check a few manually
5. **Feature quality:** Are hidden states from the right layer/position?
6. **Training instability:** Try lower learning rate, gradient clipping

---

## Summary: Path to 80%+ Accuracy

| Strategy | Difficulty | Expected Gain | Priority |
|----------|------------|---------------|----------|
| 2-layer MLP | Easy | +5% | High |
| Multi-layer hidden states | Easy | +4% | High |
| Baseline confidence features | Medium | +6% | High |
| Self-consistency feature | Easy | +6% | High |
| Weighted focal loss | Easy | +3% | Medium |
| Data augmentation | Medium | +4% | Medium |
| Ensemble (3-5 probes) | Medium | +4% | Medium |
| Multi-task learning | Hard | +5% | Low |

**Cumulative (not additive):** Combining top 5 strategies should get you to 80-85% accuracy.

**Start with:** MLP + multi-layer hidden states + baseline confidence + self-consistency + weighted loss
**Expected:** 75-80% accuracy
**Then add:** Ensemble + data augmentation
**Expected:** 80-85% accuracy
