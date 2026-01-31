# ACC-Inspired Probe Architecture (V4)

**Inspired by:** Anterior Cingulate Cortex (ACC) - the brain's conflict monitoring and cognitive control system

**Key Principle:** Aggressively compress high-dimensional neural activity into a small number of control-relevant signals that predict **value** (not just difficulty).

---

## Problems with V2/V3

| Problem | V2/V3 Approach | Why It Fails |
|---------|----------------|--------------|
| **Dimensionality** | 16,384 dims (raw hidden states) | Overfitting, poor generalization, high cost |
| **Static features** | Snapshots of hidden states | Misses instability and representational drift |
| **Weak confidence** | Entropy, logits (domain-dependent) | Poorly calibrated, secondary signal |
| **Expensive self-consistency** | 3-5 samples (3-5x cost) | Noisy, contradicts pre-emptive goal |
| **No value awareness** | Predicts difficulty, not utility | Over-triggers on hard but unrecoverable cases |
| **Fixed features** | All layers, all features | No empirical justification, likely redundant |

---

## ACC-Inspired Design Principles

### **1. Aggressive Compression**

**Brain analogy:** ACC compresses vast cortical activity into compact control signals

**Implementation:**
```
Raw hidden states (16,384 dims)
  ↓
Learned projection / PCA (200-300 dims)
  ↓
Control-relevant representation
```

**Benefits:**
- Reduces overfitting
- Improves generalization
- Forces focus on stable, task-relevant structure
- Lower inference cost

---

### **2. Dynamic Features (Not Snapshots)**

**Brain analogy:** ACC monitors **changes** in neural activity, not static states

**Implementation:** Measure how representations evolve across layers

| Dynamic Feature | Formula | What It Captures |
|-----------------|---------|------------------|
| **Cosine drift** | `1 - cos(h[L], h[L-1])` | Direction change between layers |
| **Norm ratio** | `‖h[L]‖ / ‖h[L-1]‖` | Magnitude change (growth/shrinkage) |
| **Residual change** | `‖h[L] - h[L-1]‖ / ‖h[L-1]‖` | Relative update size |
| **Attention entropy** | `-Σ p log p` (per layer) | Uncertainty in attention patterns |

**Why it works:** Conflict and error likelihood manifest as **instability** across layers, not in any single layer.

---

### **3. Value-Aware Prediction**

**Brain analogy:** ACC estimates whether exerting more effort will improve outcomes enough to justify the metabolic cost

**Implementation:** Predict **expected utility gain**

```python
utility_gain = (
    P(metacog correct | baseline wrong) * accuracy_gain
    - compute_cost * cost_weight
)
```

**Not just:**
- ❌ "Is this hard?" (difficulty)
- ❌ "Will baseline be wrong?" (error prediction)

**But:**
- ✅ "Will extra compute improve the outcome enough to justify the cost?"

---

### **4. Reduced Reliance on Self-Consistency**

**Problem:** 3-5 samples is expensive (3-5x cost) and noisy

**Solution:** Replace with **early branching uncertainty**

```python
# Generate only 1-2 tokens
partial_generation = model.generate(question, max_new_tokens=2)

# Measure branching entropy at early decode
early_entropy = compute_entropy(logits[0:2])

# High early entropy → high uncertainty
```

**Benefits:**
- Much cheaper (<1.1x cost vs 3-5x)
- Often more informative (early uncertainty predicts final uncertainty)
- Aligns with pre-emptive goal

---

### **5. Empirical Feature Selection**

**Problem:** Not all layers and features contribute equally

**Solution:** Systematic ablation

```python
# Test each feature's contribution
for feature in all_features:
    probe_without_feature = train_probe(features - {feature})
    importance[feature] = baseline_f1 - probe_without_feature.f1

# Keep only top-K features
selected_features = top_k(importance, k=20)
```

**Expected result:** ~200-300 dims (not 16,384)

---

## V4 Architecture

### **Feature Extraction Pipeline**

```
Input: Question
  ↓
[1] Extract hidden states from key layers (e.g., 8, 16, 24, 32)
  ↓
[2] Compute dynamic features:
    - Cosine drift between consecutive layers
    - Norm ratios
    - Residual changes
    - Attention entropy per layer
  ↓
[3] Compress via learned projection:
    - 4 layers × 4096 dims = 16,384 dims
    - → Projection matrix (16,384 × 256)
    - → Compressed representation (256 dims)
  ↓
[4] Add lightweight static features:
    - Early branching entropy (1-2 tokens)
    - Mean/min logit confidence (2 features)
    - Question length, token count (2 features)
  ↓
[5] Final feature vector: ~260-270 dims
```

### **Probe Architecture**

```
Input: Compressed features (260 dims)
  ↓
2-layer MLP (260 → 128 → 64)
  ↓
Three prediction heads:
  ├─→ P(error): Will baseline be wrong?
  ├─→ Conflict score: Instability/uncertainty level
  └─→ Utility gain: Expected improvement - cost
```

### **Routing Decision**

```python
# Value-aware routing
if utility_gain > threshold:
    use_metacognition()
else:
    use_baseline()
```

**Not:**
```python
# Old approach (difficulty-based)
if P(error) > threshold:
    use_metacognition()  # Might not help!
```

---

## Implementation Plan

### **Phase 1: Dimensionality Reduction**

**Option A: Learned Projection (Recommended)**
```python
class CompressionLayer(nn.Module):
    def __init__(self, input_dim=16384, output_dim=256):
        self.projection = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, hidden_states):
        compressed = self.projection(hidden_states)
        compressed = self.layer_norm(compressed)
        return compressed
```

**Option B: PCA (Faster, No Training)**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=256)
compressed = pca.fit_transform(hidden_states)
```

**Recommendation:** Start with PCA for speed, add learned projection if needed.

---

### **Phase 2: Dynamic Features**

```python
def extract_dynamic_features(hidden_states_per_layer):
    """
    Args:
        hidden_states_per_layer: List of [batch, hidden_dim] tensors
    
    Returns:
        dynamic_features: [batch, num_features] tensor
    """
    features = []
    
    for i in range(1, len(hidden_states_per_layer)):
        h_prev = hidden_states_per_layer[i-1]
        h_curr = hidden_states_per_layer[i]
        
        # Cosine drift
        cosine_sim = F.cosine_similarity(h_prev, h_curr, dim=-1)
        cosine_drift = 1.0 - cosine_sim
        features.append(cosine_drift)
        
        # Norm ratio
        norm_prev = torch.norm(h_prev, dim=-1)
        norm_curr = torch.norm(h_curr, dim=-1)
        norm_ratio = norm_curr / (norm_prev + 1e-8)
        features.append(norm_ratio)
        
        # Residual change
        residual = h_curr - h_prev
        residual_norm = torch.norm(residual, dim=-1)
        residual_change = residual_norm / (norm_prev + 1e-8)
        features.append(residual_change)
    
    return torch.stack(features, dim=-1)
```

**Result:** 3 layers × 3 features = 9 dynamic features

---

### **Phase 3: Early Branching Uncertainty**

```python
def compute_early_branching_entropy(model, question, max_tokens=2):
    """
    Generate 1-2 tokens and measure early uncertainty.
    Much cheaper than full self-consistency (3-5 samples).
    """
    inputs = tokenizer(question, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    # Compute entropy at each early token
    entropies = []
    for logits in outputs.scores[:max_tokens]:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        entropies.append(entropy.item())
    
    return {
        'mean_early_entropy': np.mean(entropies),
        'max_early_entropy': np.max(entropies),
    }
```

**Cost:** <1.1x (vs 3-5x for self-consistency)

---

### **Phase 4: Value-Aware Training**

```python
def compute_utility_gain_label(
    baseline_correct: bool,
    metacog_correct: bool,
    compute_cost: float = 0.5,  # Metacog is 1.5x baseline cost
):
    """
    Compute utility gain for training.
    
    Utility = Accuracy gain - Cost
    """
    if baseline_correct and metacog_correct:
        accuracy_gain = 0.0  # Both correct
    elif not baseline_correct and metacog_correct:
        accuracy_gain = 1.0  # Metacog fixes error
    elif baseline_correct and not metacog_correct:
        accuracy_gain = -1.0  # Metacog breaks correct answer
    else:
        accuracy_gain = 0.0  # Both wrong
    
    utility_gain = accuracy_gain - compute_cost
    
    return utility_gain
```

**Training target:** Predict this utility gain directly

---

## Expected Improvements

| Metric | V2/V3 | V4 (ACC-Inspired) | Improvement |
|--------|-------|-------------------|-------------|
| **Feature dims** | 16,389 | 260-270 | **60x reduction** |
| **Inference cost** | 1.2x | 1.1x | **Lower** |
| **Generalization** | Poor (overfitting) | Good (compressed) | **Better** |
| **Routing accuracy** | 75% | 85% | **+10%** |
| **Value awareness** | No | Yes | **New** |
| **End-to-end gain** | +1.0% | +1.5-2.5% | **+0.5-1.5%** |

---

## Ablation Study Plan

### **Feature Groups to Test**

1. **Compressed hidden states** (256 dims)
2. **Dynamic features** (9 dims)
3. **Early branching entropy** (2 dims)
4. **Lightweight confidence** (4 dims)

### **Ablation Experiments**

| Experiment | Features Used | Expected F1 |
|------------|---------------|-------------|
| **Baseline** | All features | 0.85 |
| **- Compressed hidden** | Dynamic + entropy + confidence | 0.65 |
| **- Dynamic features** | Compressed + entropy + confidence | 0.78 |
| **- Early entropy** | Compressed + dynamic + confidence | 0.82 |
| **- Confidence** | Compressed + dynamic + entropy | 0.83 |

**Expected finding:** Compressed hidden states + dynamic features are most important

---

## Summary: V2/V3 → V4

| Aspect | V2/V3 | V4 (ACC-Inspired) |
|--------|-------|-------------------|
| **Philosophy** | Feature engineering | Control system |
| **Dimensionality** | 16,389 | 260-270 |
| **Features** | Static snapshots | Dynamic + compressed |
| **Self-consistency** | 3-5 samples (expensive) | Early branching (cheap) |
| **Prediction** | Difficulty | **Value (utility gain)** |
| **Feature selection** | Fixed | Empirical ablation |
| **Inspiration** | ML best practices | **Neuroscience (ACC)** |

---

## Next Steps

1. **Implement compression layer** (PCA or learned projection)
2. **Extract dynamic features** (cosine drift, norm ratios, residual changes)
3. **Replace self-consistency** with early branching entropy
4. **Train value-aware probe** with utility gain target
5. **Run ablation study** to validate feature importance
6. **Evaluate end-to-end** on benchmarks

**Expected outcome:** A compact, robust, value-aware control system that routes computation efficiently.
