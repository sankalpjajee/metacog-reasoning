"""
ACC-Inspired Probe Training (V5) - Feeling of Knowing (FOK)

V5 trains a probe on 45-dimensional FOK features with built-in ablation support.

Feature Groups (for ablation):
  Group 1 - baseline_core:    Baseline dynamic + entropy (11 dims, idx 0-10)
  Group 2 - metacog_core:     Metacog dynamic + entropy (11 dims, idx 11-21)
  Group 3 - answer_gap:       Answer-position gap (2 dims, idx 22-23)
  Group 4 - divergence_speed: Metacog divergence speed (1 dim, idx 24)
  Group 5 - logit_lens:       Layer entropy via logit lens (8 dims, idx 25-32)
  Group 6 - cross_comparison: Token divergence + gap diff (2 dims, idx 33-34)
  Group 7 - answer_agreement: Token agreement, KL, top-k (3 dims, idx 35-37)
  Group 8 - temporal:         Temporal dynamics (4 dims, idx 38-41)
  Group 9 - calibration:      KNN confidence calibration (3 dims, idx 42-44)

Ablation modes:
  --ablation none:       Use all 45 features (default)
  --ablation leave_one_out: Train 9 models, each dropping one feature group
  --ablation cumulative: Train 9 models, progressively adding feature groups
  --ablation single:     Train 9 models, each using only one feature group

Key innovations over V4.5:
  - 45 dims (vs 11 dims) with FOK features
  - Dual-stream comparison (baseline vs metacog)
  - Focal loss for wrong detection
  - Weighted MSE for utility (20x on positive samples)
  - Built-in ablation study
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# ============================================================
# NaN-Safe Data Preprocessing
# ============================================================

def sanitize_and_normalize(combined_data: Dict[str, torch.Tensor], verbose: bool = True) -> Dict[str, torch.Tensor]:
    """
    Sanitize features and labels to prevent NaN loss during training.

    Two-step process:
    1. Replace NaN/Inf in features and labels with safe fallback values.
    2. Standardize features to zero mean and unit variance (robust to outliers).

    This is the primary fix for the NaN loss bug. NaN values in the saved
    .pt tensors (from data generation edge cases) propagate immediately
    through the loss computation, causing all losses to be NaN from epoch 1.
    """
    features = combined_data['features'].clone()
    utility_labels = combined_data['utility_labels'].clone()
    wrong_labels = combined_data['wrong_labels'].clone()
    conflict_labels = combined_data['conflict_labels'].clone()

    # --- Step 1: Report and fix NaN/Inf in features ---
    nan_mask = torch.isnan(features)
    inf_mask = torch.isinf(features)
    total_nan = nan_mask.sum().item()
    total_inf = inf_mask.sum().item()

    if verbose:
        print(f"\n[Sanitize] Feature NaN count: {total_nan} / {features.numel()}")
        print(f"[Sanitize] Feature Inf count: {total_inf} / {features.numel()}")
        if total_nan > 0:
            nan_per_dim = nan_mask.sum(dim=0)
            print(f"[Sanitize] NaN per dimension: {nan_per_dim.tolist()}")

    # Replace NaN/Inf with 0.0 (safe neutral value before normalization)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Step 2: Fix NaN/Inf in labels ---
    for name, tensor in [('utility', utility_labels), ('wrong', wrong_labels), ('conflict', conflict_labels)]:
        n = torch.isnan(tensor).sum().item() + torch.isinf(tensor).sum().item()
        if n > 0 and verbose:
            print(f"[Sanitize] {name} labels: {n} NaN/Inf values replaced with 0.0")
    utility_labels = torch.nan_to_num(utility_labels, nan=0.0, posinf=0.0, neginf=0.0)
    wrong_labels = torch.nan_to_num(wrong_labels, nan=0.0, posinf=0.0, neginf=0.0)
    conflict_labels = torch.nan_to_num(conflict_labels, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Step 3: Robust standardization (per feature dimension) ---
    # Use median and IQR instead of mean/std to be robust to outliers
    # (e.g., KL divergence features can have extreme values even after clipping)
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    # Avoid division by zero for constant features (std=0)
    std = torch.clamp(std, min=1e-6)
    features = (features - mean) / std

    # Final safety check: clamp to [-10, 10] to prevent any residual extremes
    features = torch.clamp(features, -10.0, 10.0)

    if verbose:
        print(f"[Sanitize] After normalization: mean={features.mean().item():.4f}, "
              f"std={features.std().item():.4f}, "
              f"min={features.min().item():.4f}, max={features.max().item():.4f}")
        print(f"[Sanitize] Remaining NaN: {torch.isnan(features).sum().item()}")

    return {
        'features': features,
        'wrong_labels': wrong_labels,
        'conflict_labels': conflict_labels,
        'utility_labels': utility_labels,
        # Store normalization stats for use at inference time
        'feature_mean': mean,
        'feature_std': std,
    }


# ============================================================
# Feature Group Definitions (for ablation)
# ============================================================

FEATURE_GROUPS = {
    'baseline_core': {
        'indices': list(range(0, 11)),
        'dims': 11,
        'description': 'Baseline dynamic features + early entropy',
    },
    'metacog_core': {
        'indices': list(range(11, 22)),
        'dims': 11,
        'description': 'Metacog dynamic features + early entropy',
    },
    'answer_gap': {
        'indices': list(range(22, 24)),
        'dims': 2,
        'description': 'Top-1 vs Top-2 probability gap at answer position',
    },
    'divergence_speed': {
        'indices': [24],
        'dims': 1,
        'description': 'How quickly metacog resolves uncertainty',
    },
    'logit_lens': {
        'indices': list(range(25, 33)),
        'dims': 8,
        'description': 'Layer entropy via logit lens (baseline + metacog)',
    },
    'cross_comparison': {
        'indices': list(range(33, 35)),
        'dims': 2,
        'description': 'Token divergence + answer gap difference',
    },
    'answer_agreement': {
        'indices': list(range(35, 38)),
        'dims': 3,
        'description': 'Token agreement rate, KL divergence, top-k overlap',
    },
    'temporal': {
        'indices': list(range(38, 42)),
        'dims': 4,
        'description': 'Entropy slope over time (baseline + metacog)',
    },
    'calibration': {
        'indices': list(range(42, 45)),
        'dims': 3,
        'description': 'KNN confidence calibration from experience',
    },
}

# Ordered list for cumulative ablation
FEATURE_GROUP_ORDER = [
    'baseline_core',      # Start with what V4 had
    'metacog_core',       # Add metacog stream
    'answer_gap',         # Add FOK feature 1
    'logit_lens',         # Add FOK feature 4
    'cross_comparison',   # Add cross-comparison
    'answer_agreement',   # Add early answer agreement
    'divergence_speed',   # Add FOK feature 3
    'temporal',           # Add temporal dynamics
    'calibration',        # Add confidence calibration
]


def get_feature_indices(groups: List[str]) -> List[int]:
    """Get sorted feature indices for a list of feature groups."""
    indices = []
    for group in groups:
        indices.extend(FEATURE_GROUPS[group]['indices'])
    return sorted(indices)


def get_feature_dim(groups: List[str]) -> int:
    """Get total feature dimensions for a list of feature groups."""
    return sum(FEATURE_GROUPS[g]['dims'] for g in groups)


# ============================================================
# Model
# ============================================================

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in binary classification."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class FOKProbe(nn.Module):
    """
    FOK-Inspired probe with dual-stream input and value-aware output.

    Input: variable dims (up to 45 for full feature set)
    Output: 3 predictions (wrong, conflict, utility)
    """
    def __init__(
        self,
        input_dim: int = 45,
        hidden_dim: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()

        # 3-layer MLP (deeper than V4 to handle more features)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(dropout)

        # Three prediction heads
        self.wrong_head = nn.Linear(32, 1)      # Binary: P(baseline wrong)
        self.conflict_head = nn.Linear(32, 1)   # Continuous: conflict/instability
        self.utility_head = nn.Linear(32, 1)    # Continuous: expected utility gain

        # Initialize weights
        for layer in [self.fc1, self.fc2, self.fc3,
                      self.wrong_head, self.conflict_head, self.utility_head]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.fc1(features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        wrong_logits = self.wrong_head(x).squeeze(-1)
        conflict_score = torch.sigmoid(self.conflict_head(x)).squeeze(-1)
        utility_score = self.utility_head(x).squeeze(-1)  # Unbounded (no tanh!)

        return wrong_logits, conflict_score, utility_score


class EnsembleFOKProbe(nn.Module):
    """Ensemble of multiple FOK probes for improved robustness."""
    def __init__(self, num_probes: int = 3, **probe_kwargs):
        super().__init__()
        self.probes = nn.ModuleList([FOKProbe(**probe_kwargs) for _ in range(num_probes)])

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wrong_preds = []
        conflict_preds = []
        utility_preds = []

        for probe in self.probes:
            wrong, conflict, utility = probe(features)
            wrong_preds.append(wrong)
            conflict_preds.append(conflict)
            utility_preds.append(utility)

        return (
            torch.stack(wrong_preds, dim=0).mean(dim=0),
            torch.stack(conflict_preds, dim=0).mean(dim=0),
            torch.stack(utility_preds, dim=0).mean(dim=0),
        )


# ============================================================
# Dataset
# ============================================================

class FOKDataset(Dataset):
    """Dataset for FOK probe training with feature group selection."""
    def __init__(self, tensor_data: Dict[str, torch.Tensor], feature_indices: Optional[List[int]] = None):
        self.features = tensor_data['features']
        self.wrong_labels = tensor_data['wrong_labels']
        self.conflict_labels = tensor_data['conflict_labels']
        self.utility_labels = tensor_data['utility_labels']

        # Select specific feature indices if provided
        if feature_indices is not None:
            self.features = self.features[:, feature_indices]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'wrong_label': self.wrong_labels[idx],
            'conflict_label': self.conflict_labels[idx],
            'utility_label': self.utility_labels[idx],
        }


# ============================================================
# Training Loop
# ============================================================

def train_fok_probe(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    device: str,
    epochs: int,
    learning_rate: float,
    lambda_wrong: float = 1.0,
    lambda_conflict: float = 0.3,
    lambda_utility: float = 1.5,
    weight_positive: float = 20.0,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    """Train FOK probe with multi-task loss."""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    mse_loss = nn.MSELoss()

    def weighted_utility_loss(pred, target, w_positive=20.0):
        """Weighted MSE loss with higher weight for positive utility samples."""
        weights = torch.ones_like(target)
        positive_mask = target > 0
        weights[positive_mask] = w_positive
        return (weights * (pred - target) ** 2).mean()

    # Print utility distribution
    if verbose:
        all_train_utils = []
        for batch in train_loader:
            all_train_utils.extend(batch['utility_label'].numpy())
        all_train_utils = np.array(all_train_utils)
        print(f"\nUtility label distribution:")
        print(f"  Positive (>0): {(all_train_utils > 0).mean()*100:.1f}%")
        print(f"  Negative (<=0): {(all_train_utils <= 0).mean()*100:.1f}%")
        print(f"  Mean: {all_train_utils.mean():.3f}")

    best_val_utility_corr = -1.0
    best_model_state = None
    history = {
        'train_loss': [], 'val_loss': [],
        'val_wrong_f1': [], 'val_utility_corr': [],
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, disable=not verbose):
            features = batch['features'].to(device)
            wrong_labels = batch['wrong_label'].to(device)
            conflict_labels = batch['conflict_label'].to(device)
            utility_labels = batch['utility_label'].to(device)

            optimizer.zero_grad()

            wrong_logits, conflict_pred, utility_pred = model(features)

            loss_wrong = focal_loss(wrong_logits, wrong_labels)
            loss_conflict = mse_loss(conflict_pred, conflict_labels)
            loss_utility = weighted_utility_loss(utility_pred, utility_labels, w_positive=weight_positive)

            total_loss = (
                lambda_wrong * loss_wrong +
                lambda_conflict * loss_conflict +
                lambda_utility * loss_utility
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(total_loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        all_wrong_preds = []
        all_wrong_labels = []
        all_utility_preds = []
        all_utility_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False, disable=not verbose):
                features = batch['features'].to(device)
                wrong_labels = batch['wrong_label'].to(device)
                conflict_labels = batch['conflict_label'].to(device)
                utility_labels = batch['utility_label'].to(device)

                wrong_logits, conflict_pred, utility_pred = model(features)

                loss_wrong = focal_loss(wrong_logits, wrong_labels)
                loss_conflict = mse_loss(conflict_pred, conflict_labels)
                loss_utility = mse_loss(utility_pred, utility_labels)

                total_loss = (
                    lambda_wrong * loss_wrong +
                    lambda_conflict * loss_conflict +
                    lambda_utility * loss_utility
                )

                val_losses.append(total_loss.item())

                wrong_probs = torch.sigmoid(wrong_logits)
                all_wrong_preds.extend(wrong_probs.cpu().numpy())
                all_wrong_labels.extend(wrong_labels.cpu().numpy())
                all_utility_preds.extend(utility_pred.cpu().numpy())
                all_utility_labels.extend(utility_labels.cpu().numpy())

        # Compute metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        from scipy.stats import pearsonr

        wrong_preds_binary = (np.array(all_wrong_preds) > 0.5).astype(int)
        wrong_labels_binary = np.array(all_wrong_labels).astype(int)

        val_wrong_f1 = f1_score(wrong_labels_binary, wrong_preds_binary, zero_division=0)
        val_precision = precision_score(wrong_labels_binary, wrong_preds_binary, zero_division=0)
        val_recall = recall_score(wrong_labels_binary, wrong_preds_binary, zero_division=0)

        utility_corr, _ = pearsonr(all_utility_preds, all_utility_labels)
        if np.isnan(utility_corr):
            utility_corr = 0.0

        if utility_corr > best_val_utility_corr:
            best_val_utility_corr = utility_corr
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_wrong_f1'].append(float(val_wrong_f1))
        history['val_utility_corr'].append(float(utility_corr))

        if verbose:
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Wrong F1: {val_wrong_f1:.4f} (P={val_precision:.3f}, R={val_recall:.3f})")
            print(f"  Val Utility Corr: {utility_corr:.4f} {'*' if utility_corr == best_val_utility_corr else ''}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


# ============================================================
# Ablation Study
# ============================================================

def run_ablation_study(
    combined_data: Dict[str, torch.Tensor],
    ablation_mode: str,
    args,
    device: str,
) -> Dict:
    """
    Run ablation study with different feature group combinations.

    Modes:
      leave_one_out: Drop one group at a time
      cumulative: Add groups one at a time
      single: Use one group at a time
    """
    results = {}

    if ablation_mode == 'leave_one_out':
        experiments = []
        for group_name in FEATURE_GROUP_ORDER:
            groups = [g for g in FEATURE_GROUP_ORDER if g != group_name]
            experiments.append({
                'name': f'without_{group_name}',
                'groups': groups,
                'dropped': group_name,
            })

    elif ablation_mode == 'cumulative':
        experiments = []
        for i, group_name in enumerate(FEATURE_GROUP_ORDER):
            groups = FEATURE_GROUP_ORDER[:i+1]
            experiments.append({
                'name': f'up_to_{group_name}',
                'groups': groups,
                'added': group_name,
            })

    elif ablation_mode == 'single':
        experiments = []
        for group_name in FEATURE_GROUP_ORDER:
            experiments.append({
                'name': f'only_{group_name}',
                'groups': [group_name],
            })

    else:
        raise ValueError(f"Unknown ablation mode: {ablation_mode}")

    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: {ablation_mode}")
    print(f"Running {len(experiments)} experiments")
    print(f"{'='*60}")

    for exp in experiments:
        exp_name = exp['name']
        groups = exp['groups']
        feature_indices = get_feature_indices(groups)
        input_dim = len(feature_indices)

        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"Groups: {', '.join(groups)}")
        print(f"Feature dims: {input_dim}")
        print(f"{'='*60}")

        # Create dataset with selected features
        dataset = FOKDataset(combined_data, feature_indices=feature_indices)

        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size

        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        if args.use_ensemble:
            model = EnsembleFOKProbe(num_probes=args.num_probes, input_dim=input_dim)
        else:
            model = FOKProbe(input_dim=input_dim)

        model, history = train_fok_probe(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lambda_wrong=args.lambda_wrong,
            lambda_conflict=args.lambda_conflict,
            lambda_utility=args.lambda_utility,
            weight_positive=args.weight_positive,
            verbose=False,
        )

        best_utility_corr = max(history['val_utility_corr'])
        best_wrong_f1 = max(history['val_wrong_f1'])
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]

        results[exp_name] = {
            'groups': groups,
            'feature_dims': input_dim,
            'best_utility_corr': best_utility_corr,
            'best_wrong_f1': best_wrong_f1,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'history': history,
        }

        print(f"  Best Utility Corr: {best_utility_corr:.4f}")
        print(f"  Best Wrong F1: {best_wrong_f1:.4f}")
        print(f"  Final Train/Val Loss: {final_train_loss:.4f} / {final_val_loss:.4f}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"ABLATION SUMMARY: {ablation_mode}")
    print(f"{'='*60}")
    print(f"{'Experiment':<35} {'Dims':>5} {'Util Corr':>10} {'Wrong F1':>10}")
    print(f"{'-'*60}")

    for exp_name, res in results.items():
        print(f"{exp_name:<35} {res['feature_dims']:>5} {res['best_utility_corr']:>10.4f} {res['best_wrong_f1']:>10.4f}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train FOK-inspired probe (V5)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu", "hellaswag"])
    parser.add_argument("--use_ensemble", action="store_true")
    parser.add_argument("--num_probes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lambda_wrong", type=float, default=1.0)
    parser.add_argument("--lambda_conflict", type=float, default=0.3)
    parser.add_argument("--lambda_utility", type=float, default=1.5)
    parser.add_argument("--weight_positive", type=float, default=20.0)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", type=str, default="none",
                        choices=["none", "leave_one_out", "cumulative", "single"],
                        help="Ablation study mode")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    print("\nLoading training data...")
    all_features = []
    all_wrong_labels = []
    all_conflict_labels = []
    all_utility_labels = []

    for benchmark in args.benchmarks:
        tensor_path = os.path.join(args.data_dir, f"{benchmark}_tensors.pt")
        if not os.path.exists(tensor_path):
            print(f"Warning: {tensor_path} not found, skipping {benchmark}")
            continue

        print(f"Loading {benchmark}...")
        data = torch.load(tensor_path)

        # V5: Use all 45 features directly (no slicing needed)
        all_features.append(data['features'])
        all_wrong_labels.append(data['wrong_labels'])
        all_conflict_labels.append(data['conflict_labels'])
        all_utility_labels.append(data['utility_labels'])

    combined_data = {
        'features': torch.cat(all_features, dim=0),
        'wrong_labels': torch.cat(all_wrong_labels, dim=0),
        'conflict_labels': torch.cat(all_conflict_labels, dim=0),
        'utility_labels': torch.cat(all_utility_labels, dim=0),
    }

    print(f"\nTotal samples: {len(combined_data['features'])}")
    print(f"Feature dimensions: {combined_data['features'].shape[1]}")
    print(f"Wrong rate: {combined_data['wrong_labels'].mean().item()*100:.1f}%")
    print(f"Mean utility: {combined_data['utility_labels'].mean().item():.3f}")
    print(f"Utility positive rate: {(combined_data['utility_labels'] > 0).float().mean().item()*100:.1f}%")

    # ========================================
    # Sanitize & Normalize Features
    # ========================================
    combined_data = sanitize_and_normalize(combined_data, verbose=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================
    # Ablation Study
    # ========================================
    if args.ablation != "none":
        ablation_results = run_ablation_study(combined_data, args.ablation, args, device)

        ablation_path = os.path.join(args.output_dir, f"ablation_{args.ablation}.json")
        serializable_results = {}
        for name, res in ablation_results.items():
            serializable_results[name] = {
                'groups': res['groups'],
                'feature_dims': res['feature_dims'],
                'best_utility_corr': float(res['best_utility_corr']),
                'best_wrong_f1': float(res['best_wrong_f1']),
                'final_train_loss': float(res['final_train_loss']),
                'final_val_loss': float(res['final_val_loss']),
            }
        with open(ablation_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nAblation results saved to: {ablation_path}")
        return

    # ========================================
    # Full Training (all 45 features)
    # ========================================
    dataset = FOKDataset(combined_data)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print("\nCreating model...")
    input_dim = combined_data['features'].shape[1]

    if args.use_ensemble:
        print(f"Using ensemble of {args.num_probes} probes")
        model = EnsembleFOKProbe(num_probes=args.num_probes, input_dim=input_dim)
    else:
        model = FOKProbe(input_dim=input_dim)

    print(f"Input dimensions: {input_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Print feature group breakdown
    print(f"\nFeature groups:")
    for name, info in FEATURE_GROUPS.items():
        print(f"  {name}: {info['dims']} dims (idx {info['indices'][0]}-{info['indices'][-1]}) - {info['description']}")

    # Train
    print("\nTraining...")
    model, history = train_fok_probe(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_wrong=args.lambda_wrong,
        lambda_conflict=args.lambda_conflict,
        lambda_utility=args.lambda_utility,
        weight_positive=args.weight_positive,
    )

    # Save model
    model_path = os.path.join(args.output_dir, "best_probe.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to: {model_path}")

    # Save normalization stats for inference
    norm_stats_path = os.path.join(args.output_dir, "norm_stats.pt")
    torch.save({
        'feature_mean': combined_data['feature_mean'],
        'feature_std': combined_data['feature_std'],
    }, norm_stats_path)
    print(f"Saved normalization stats to: {norm_stats_path}")

    # Save config
    config = {
        'version': 'v5_fok',
        'input_dim': input_dim,
        'hidden_dim': 128,
        'dropout': 0.15,
        'use_ensemble': args.use_ensemble,
        'num_probes': args.num_probes if args.use_ensemble else 1,
        'lambda_wrong': args.lambda_wrong,
        'lambda_conflict': args.lambda_conflict,
        'lambda_utility': args.lambda_utility,
        'weight_positive': args.weight_positive,
        'feature_groups': {name: {'dims': info['dims'], 'indices': info['indices'], 'description': info['description']} for name, info in FEATURE_GROUPS.items()},
        'total_features': 45,
    }

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")

    # Save history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}")

    print("\n" + "="*60)
    print("V5 FOK Training complete!")
    print(f"Best validation utility correlation: {max(history['val_utility_corr']):.4f}")
    print(f"Best validation wrong F1: {max(history['val_wrong_f1']):.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
