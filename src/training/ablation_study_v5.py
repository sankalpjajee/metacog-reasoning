"""
V5 FOK Probe Ablation Study

Tests which feature groups contribute to probe performance by:
1. Training with ALL features (full model)
2. Training with each feature group REMOVED (leave-one-out)
3. Training with ONLY each feature group (isolation test)
4. Computing feature importance via gradient-based attribution

Feature Groups:
  Group 1: Baseline Dynamic (9 dims) [0:9]
  Group 2: Baseline Early Entropy (2 dims) [9:11]
  Group 3: Metacog Dynamic (9 dims) [11:20]
  Group 4: Metacog Early Entropy (2 dims) [20:22]
  Group 5: Answer-Position Gap (2 dims) [22:24]
  Group 6: Metacog Divergence Speed (1 dim) [24]
  Group 7: Baseline Layer Entropy (4 dims) [25:29]
  Group 8: Metacog Layer Entropy (4 dims) [29:33]
  Group 9: Cross-Comparison (2 dims) [33:35]
  Group 10: Early Answer Agreement (3 dims) [35:38]
  Group 11: Baseline Temporal Dynamics (2 dims) [38:40]
  Group 12: Metacog Temporal Dynamics (2 dims) [40:42]
  Group 13: Confidence Calibration (3 dims) [42:45]

Usage:
  python src/training/ablation_study_v5.py \
      --data_dir data/training/acc_v5 \
      --output_dir results/ablation_v5
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import the sanitize_and_normalize function from the training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_acc_probe_v5 import sanitize_and_normalize


# ============================================================
# Feature Group Definitions
# ============================================================

FEATURE_GROUPS = {
    "baseline_dynamic":       {"start": 0,  "end": 9,  "dims": 9,  "desc": "Baseline cosine drift, norm ratio, residual change"},
    "baseline_early_entropy": {"start": 9,  "end": 11, "dims": 2,  "desc": "Baseline early entropy (mean, max)"},
    "metacog_dynamic":        {"start": 11, "end": 20, "dims": 9,  "desc": "Metacog cosine drift, norm ratio, residual change"},
    "metacog_early_entropy":  {"start": 20, "end": 22, "dims": 2,  "desc": "Metacog early entropy (mean, max)"},
    "answer_position_gap":    {"start": 22, "end": 24, "dims": 2,  "desc": "Top-1 vs Top-2 gap at answer position"},
    "metacog_div_speed":      {"start": 24, "end": 25, "dims": 1,  "desc": "Metacog divergence speed"},
    "baseline_layer_entropy": {"start": 25, "end": 29, "dims": 4,  "desc": "Baseline logit lens entropy at layers 8,16,24,32"},
    "metacog_layer_entropy":  {"start": 29, "end": 33, "dims": 4,  "desc": "Metacog logit lens entropy at layers 8,16,24,32"},
    "cross_comparison":       {"start": 33, "end": 35, "dims": 2,  "desc": "Token divergence + answer gap difference"},
    "answer_agreement":       {"start": 35, "end": 38, "dims": 3,  "desc": "Token agreement rate, KL div, top-k overlap"},
    "baseline_temporal":      {"start": 38, "end": 40, "dims": 2,  "desc": "Baseline entropy slope (early-mid, mid-late)"},
    "metacog_temporal":       {"start": 40, "end": 42, "dims": 2,  "desc": "Metacog entropy slope (early-mid, mid-late)"},
    "calibration":            {"start": 42, "end": 45, "dims": 3,  "desc": "KNN calibration (baseline acc, metacog acc, utility)"},
}

# Higher-level groupings for summary
CATEGORY_GROUPS = {
    "V4_baseline_only":   ["baseline_dynamic", "baseline_early_entropy"],
    "metacog_mirror":     ["metacog_dynamic", "metacog_early_entropy"],
    "FOK_features":       ["answer_position_gap", "metacog_div_speed", "baseline_layer_entropy",
                           "metacog_layer_entropy", "cross_comparison", "answer_agreement"],
    "temporal_dynamics":  ["baseline_temporal", "metacog_temporal"],
    "calibration":        ["calibration"],
}


# ============================================================
# Probe Architecture (same as training)
# ============================================================

class FOKProbe(nn.Module):
    def __init__(self, input_dim: int = 45, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(dropout)
        self.wrong_head = nn.Linear(32, 1)
        self.conflict_head = nn.Linear(32, 1)
        self.utility_head = nn.Linear(32, 1)

    def forward(self, features):
        x = self.dropout1(self.relu(self.bn1(self.fc1(features))))
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu(self.fc3(x)))
        wrong = self.wrong_head(x).squeeze(-1)
        conflict = torch.sigmoid(self.conflict_head(x)).squeeze(-1)
        utility = self.utility_head(x).squeeze(-1)
        return wrong, conflict, utility


# ============================================================
# Training Function
# ============================================================

def train_probe(
    features: torch.Tensor,
    wrong_labels: torch.Tensor,
    utility_labels: torch.Tensor,
    input_dim: int,
    num_epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 64,
    weight_positive: float = 20.0,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict:
    """Train a probe and return validation metrics."""

    # Train/val split (80/20)
    n = len(features)
    indices = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    train_features = features[train_idx].to(device)
    train_wrong = wrong_labels[train_idx].to(device)
    train_utility = utility_labels[train_idx].to(device)

    val_features = features[val_idx].to(device)
    val_wrong = wrong_labels[val_idx].to(device)
    val_utility = utility_labels[val_idx].to(device)

    # Create data loader
    train_dataset = TensorDataset(train_features, train_wrong, train_utility)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = FOKProbe(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Compute sample weights for utility loss
    utility_weights = torch.ones_like(train_utility)
    utility_weights[train_utility > 0] = weight_positive

    best_val_loss = float('inf')
    best_metrics = {}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_features, batch_wrong, batch_utility in train_loader:
            optimizer.zero_grad()

            wrong_pred, conflict_pred, utility_pred = model(batch_features)

            # Wrong prediction loss (BCE)
            wrong_loss = F.binary_cross_entropy_with_logits(wrong_pred, batch_wrong)

            # Utility loss (weighted MSE)
            batch_weights = torch.ones_like(batch_utility)
            batch_weights[batch_utility > 0] = weight_positive
            utility_loss = (batch_weights * (utility_pred - batch_utility) ** 2).mean()

            loss = wrong_loss + utility_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_wrong_pred, val_conflict_pred, val_utility_pred = model(val_features)

            val_wrong_loss = F.binary_cross_entropy_with_logits(val_wrong_pred, val_wrong).item()

            val_utility_weights = torch.ones_like(val_utility)
            val_utility_weights[val_utility > 0] = weight_positive
            val_utility_loss = (val_utility_weights * (val_utility_pred - val_utility) ** 2).mean().item()

            val_loss = val_wrong_loss + val_utility_loss

            # Compute metrics
            wrong_preds_binary = (torch.sigmoid(val_wrong_pred) > 0.5).float()
            wrong_acc = (wrong_preds_binary == val_wrong).float().mean().item()

            utility_corr = np.corrcoef(
                val_utility_pred.cpu().numpy(),
                val_utility.cpu().numpy()
            )[0, 1] if len(val_utility) > 1 else 0.0

            # Routing metrics
            routing_decisions = (val_utility_pred > 0.0).float()
            routing_rate = routing_decisions.mean().item()

            # Utility of routing decisions
            routed_mask = routing_decisions.bool()
            if routed_mask.sum() > 0:
                routed_utility = val_utility[routed_mask].mean().item()
            else:
                routed_utility = 0.0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                'val_loss': val_loss,
                'val_wrong_loss': val_wrong_loss,
                'val_utility_loss': val_utility_loss,
                'wrong_accuracy': wrong_acc,
                'utility_correlation': utility_corr if not np.isnan(utility_corr) else 0.0,
                'routing_rate': routing_rate,
                'routed_utility': routed_utility,
            }

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={epoch_loss/len(train_loader):.4f}, "
                  f"val_loss={val_loss:.4f}, wrong_acc={wrong_acc:.3f}, "
                  f"util_corr={utility_corr:.3f}, route={routing_rate:.3f}")

    return best_metrics


# ============================================================
# Feature Masking Functions
# ============================================================

def mask_feature_group(features: torch.Tensor, group_name: str) -> torch.Tensor:
    """Zero out a specific feature group (leave-one-out)."""
    masked = features.clone()
    group = FEATURE_GROUPS[group_name]
    masked[:, group["start"]:group["end"]] = 0.0
    return masked


def isolate_feature_group(features: torch.Tensor, group_name: str) -> torch.Tensor:
    """Keep only a specific feature group, zero out everything else."""
    isolated = torch.zeros_like(features)
    group = FEATURE_GROUPS[group_name]
    isolated[:, group["start"]:group["end"]] = features[:, group["start"]:group["end"]]
    return isolated


def select_feature_groups(features: torch.Tensor, group_names: List[str]) -> torch.Tensor:
    """Select and concatenate specific feature groups into a new tensor."""
    selected = []
    for name in group_names:
        group = FEATURE_GROUPS[name]
        selected.append(features[:, group["start"]:group["end"]])
    return torch.cat(selected, dim=1)


# ============================================================
# Gradient-Based Feature Importance
# ============================================================

def compute_gradient_importance(
    features: torch.Tensor,
    utility_labels: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute feature importance via integrated gradients.
    Returns importance score for each feature group.
    """
    features = features.to(device).requires_grad_(True)
    utility_labels = utility_labels.to(device)

    model = FOKProbe(input_dim=features.shape[1]).to(device)

    # Quick training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for _ in range(30):
        optimizer.zero_grad()
        _, _, utility_pred = model(features)
        loss = F.mse_loss(utility_pred, utility_labels)
        loss.backward()
        optimizer.step()

    # Compute gradients
    model.eval()
    features_grad = features.detach().clone().requires_grad_(True)
    _, _, utility_pred = model(features_grad)
    loss = F.mse_loss(utility_pred, utility_labels)
    loss.backward()

    gradients = features_grad.grad.abs().mean(dim=0).cpu().numpy()

    # Aggregate by feature group
    importance = {}
    for name, group in FEATURE_GROUPS.items():
        group_grads = gradients[group["start"]:group["end"]]
        importance[name] = float(np.mean(group_grads))

    return importance


# ============================================================
# Main Ablation Study
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="V5 FOK Probe Ablation Study")
    parser.add_argument("--data_dir", type=str, default="data/training/acc_v5")
    parser.add_argument("--output_dir", type=str, default="results/ablation_v5")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs per experiment for stability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading training data...")
    all_features = []
    all_wrong_labels = []
    all_utility_labels = []

    for benchmark in ["gsm8k", "mmlu", "hellaswag"]:
        tensor_path = os.path.join(args.data_dir, f"{benchmark}_tensors.pt")
        if os.path.exists(tensor_path):
            data = torch.load(tensor_path, map_location="cpu")
            all_features.append(data['features'])
            all_wrong_labels.append(data['wrong_labels'])
            all_utility_labels.append(data['utility_labels'])
            print(f"  Loaded {len(data['features'])} samples from {benchmark}")

    if not all_features:
        print("ERROR: No training data found!")
        return

    features = torch.cat(all_features, dim=0)
    wrong_labels = torch.cat(all_wrong_labels, dim=0)
    utility_labels = torch.cat(all_utility_labels, dim=0)

    print(f"\nTotal samples: {len(features)}")
    print(f"Feature dimensions: {features.shape[1]}")
    print(f"Wrong rate: {wrong_labels.mean():.3f}")
    print(f"Utility positive rate: {(utility_labels > 0).float().mean():.3f}")

    # Sanitize and normalize — same preprocessing as training script
    # This prevents NaN loss from propagating through all ablation experiments
    combined = sanitize_and_normalize(
        {'features': features, 'wrong_labels': wrong_labels,
         'conflict_labels': torch.zeros_like(wrong_labels),
         'utility_labels': utility_labels},
        verbose=True
    )
    features = combined['features']
    wrong_labels = combined['wrong_labels']
    utility_labels = combined['utility_labels']

    results = {}

    # ========================================
    # 1. FULL MODEL (baseline)
    # ========================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: Full Model (all 45 features)")
    print(f"{'='*60}")

    full_metrics_list = []
    for run in range(args.num_runs):
        torch.manual_seed(args.seed + run)
        metrics = train_probe(
            features, wrong_labels, utility_labels,
            input_dim=45, num_epochs=args.num_epochs,
            device=args.device, verbose=(run == 0)
        )
        full_metrics_list.append(metrics)

    full_metrics = {k: np.mean([m[k] for m in full_metrics_list]) for k in full_metrics_list[0]}
    full_std = {k: np.std([m[k] for m in full_metrics_list]) for k in full_metrics_list[0]}

    print(f"\nFull model results (mean +/- std over {args.num_runs} runs):")
    print(f"  Wrong accuracy:     {full_metrics['wrong_accuracy']:.3f} +/- {full_std['wrong_accuracy']:.3f}")
    print(f"  Utility correlation: {full_metrics['utility_correlation']:.3f} +/- {full_std['utility_correlation']:.3f}")
    print(f"  Routing rate:       {full_metrics['routing_rate']:.3f} +/- {full_std['routing_rate']:.3f}")
    print(f"  Routed utility:     {full_metrics['routed_utility']:.3f} +/- {full_std['routed_utility']:.3f}")

    results['full_model'] = {'mean': full_metrics, 'std': full_std}

    # ========================================
    # 2. LEAVE-ONE-OUT (remove each group)
    # ========================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Leave-One-Out (remove each feature group)")
    print(f"{'='*60}")

    leave_one_out_results = {}

    for group_name, group_info in FEATURE_GROUPS.items():
        print(f"\n  Removing: {group_name} ({group_info['dims']} dims) - {group_info['desc']}")

        masked_features = mask_feature_group(features, group_name)

        group_metrics_list = []
        for run in range(args.num_runs):
            torch.manual_seed(args.seed + run)
            metrics = train_probe(
                masked_features, wrong_labels, utility_labels,
                input_dim=45, num_epochs=args.num_epochs,
                device=args.device
            )
            group_metrics_list.append(metrics)

        group_metrics = {k: np.mean([m[k] for m in group_metrics_list]) for k in group_metrics_list[0]}
        group_std = {k: np.std([m[k] for m in group_metrics_list]) for k in group_metrics_list[0]}

        # Compute impact (negative = removing hurts = feature is important)
        impact = {
            'wrong_acc_impact': group_metrics['wrong_accuracy'] - full_metrics['wrong_accuracy'],
            'utility_corr_impact': group_metrics['utility_correlation'] - full_metrics['utility_correlation'],
            'routing_rate_impact': group_metrics['routing_rate'] - full_metrics['routing_rate'],
        }

        print(f"    Wrong acc: {group_metrics['wrong_accuracy']:.3f} (impact: {impact['wrong_acc_impact']:+.3f})")
        print(f"    Util corr: {group_metrics['utility_correlation']:.3f} (impact: {impact['utility_corr_impact']:+.3f})")
        print(f"    Route rate: {group_metrics['routing_rate']:.3f} (impact: {impact['routing_rate_impact']:+.3f})")

        leave_one_out_results[group_name] = {
            'mean': group_metrics,
            'std': group_std,
            'impact': impact,
        }

    results['leave_one_out'] = leave_one_out_results

    # ========================================
    # 3. ISOLATION TEST (only each group)
    # ========================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: Isolation Test (only each feature group)")
    print(f"{'='*60}")

    isolation_results = {}

    for group_name, group_info in FEATURE_GROUPS.items():
        print(f"\n  Only: {group_name} ({group_info['dims']} dims)")

        isolated_features = select_feature_groups(features, [group_name])

        group_metrics_list = []
        for run in range(args.num_runs):
            torch.manual_seed(args.seed + run)
            metrics = train_probe(
                isolated_features, wrong_labels, utility_labels,
                input_dim=group_info['dims'], num_epochs=args.num_epochs,
                device=args.device
            )
            group_metrics_list.append(metrics)

        group_metrics = {k: np.mean([m[k] for m in group_metrics_list]) for k in group_metrics_list[0]}

        print(f"    Wrong acc: {group_metrics['wrong_accuracy']:.3f}")
        print(f"    Util corr: {group_metrics['utility_correlation']:.3f}")
        print(f"    Route rate: {group_metrics['routing_rate']:.3f}")

        isolation_results[group_name] = {'mean': group_metrics}

    results['isolation'] = isolation_results

    # ========================================
    # 4. CATEGORY-LEVEL ABLATION
    # ========================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 4: Category-Level Ablation")
    print(f"{'='*60}")

    category_results = {}

    for cat_name, cat_groups in CATEGORY_GROUPS.items():
        total_dims = sum(FEATURE_GROUPS[g]['dims'] for g in cat_groups)
        print(f"\n  Category: {cat_name} ({total_dims} dims)")
        print(f"    Groups: {', '.join(cat_groups)}")

        # Test with ONLY this category
        cat_features = select_feature_groups(features, cat_groups)

        cat_metrics_list = []
        for run in range(args.num_runs):
            torch.manual_seed(args.seed + run)
            metrics = train_probe(
                cat_features, wrong_labels, utility_labels,
                input_dim=total_dims, num_epochs=args.num_epochs,
                device=args.device
            )
            cat_metrics_list.append(metrics)

        cat_metrics = {k: np.mean([m[k] for m in cat_metrics_list]) for k in cat_metrics_list[0]}

        print(f"    Wrong acc: {cat_metrics['wrong_accuracy']:.3f}")
        print(f"    Util corr: {cat_metrics['utility_correlation']:.3f}")
        print(f"    Route rate: {cat_metrics['routing_rate']:.3f}")

        category_results[cat_name] = {'mean': cat_metrics, 'dims': total_dims}

    results['categories'] = category_results

    # ========================================
    # 5. GRADIENT-BASED IMPORTANCE
    # ========================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 5: Gradient-Based Feature Importance")
    print(f"{'='*60}")

    importance = compute_gradient_importance(features, utility_labels, device=args.device)

    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nFeature importance ranking:")
    for rank, (name, score) in enumerate(sorted_importance, 1):
        print(f"  {rank}. {name}: {score:.6f} ({FEATURE_GROUPS[name]['dims']} dims)")

    results['gradient_importance'] = importance

    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*60}")

    print(f"\nFull model: wrong_acc={full_metrics['wrong_accuracy']:.3f}, "
          f"util_corr={full_metrics['utility_correlation']:.3f}, "
          f"route_rate={full_metrics['routing_rate']:.3f}")

    print(f"\n--- Leave-One-Out Impact (negative = feature is important) ---")
    sorted_loo = sorted(
        leave_one_out_results.items(),
        key=lambda x: x[1]['impact']['utility_corr_impact']
    )
    for name, data in sorted_loo:
        impact = data['impact']
        print(f"  {name:30s}: util_corr_impact={impact['utility_corr_impact']:+.3f}, "
              f"wrong_acc_impact={impact['wrong_acc_impact']:+.3f}")

    print(f"\n--- Isolation Test (higher = more useful alone) ---")
    sorted_iso = sorted(
        isolation_results.items(),
        key=lambda x: x[1]['mean']['utility_correlation'],
        reverse=True
    )
    for name, data in sorted_iso:
        m = data['mean']
        print(f"  {name:30s}: util_corr={m['utility_correlation']:.3f}, "
              f"wrong_acc={m['wrong_accuracy']:.3f}")

    print(f"\n--- Category Comparison ---")
    for cat_name, data in category_results.items():
        m = data['mean']
        print(f"  {cat_name:25s} ({data['dims']:2d} dims): "
              f"util_corr={m['utility_correlation']:.3f}, "
              f"wrong_acc={m['wrong_accuracy']:.3f}")

    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")

    # Find most important features (largest negative impact when removed)
    important = [name for name, data in sorted_loo[:3]]
    print(f"\nMost important features (removing hurts most):")
    for name in important:
        impact = leave_one_out_results[name]['impact']
        print(f"  - {name}: util_corr drops by {abs(impact['utility_corr_impact']):.3f}")

    # Find least important features (removing doesn't hurt or helps)
    unimportant = [name for name, data in sorted_loo[-3:]]
    print(f"\nLeast important features (safe to remove):")
    for name in unimportant:
        impact = leave_one_out_results[name]['impact']
        print(f"  - {name}: util_corr changes by {impact['utility_corr_impact']:+.3f}")

    # V4 vs V5 comparison
    v4_groups = ["baseline_dynamic", "baseline_early_entropy"]
    v4_dims = sum(FEATURE_GROUPS[g]['dims'] for g in v4_groups)
    v5_new_groups = [g for g in FEATURE_GROUPS if g not in v4_groups]

    print(f"\nV4 features ({v4_dims} dims) vs V5 new features ({45 - v4_dims} dims):")
    if 'V4_baseline_only' in category_results:
        v4_corr = category_results['V4_baseline_only']['mean']['utility_correlation']
        print(f"  V4 features alone: util_corr={v4_corr:.3f}")
    print(f"  Full V5 model:     util_corr={full_metrics['utility_correlation']:.3f}")
    print(f"  Improvement:       {full_metrics['utility_correlation'] - category_results.get('V4_baseline_only', {}).get('mean', {}).get('utility_correlation', 0):.3f}")

    # Save results
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
