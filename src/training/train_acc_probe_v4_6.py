"""
ACC-Inspired Probe Training (V4.6)

ABLATION: Test if weighted loss helps or hurts routing.

Changes from V4.1:
- ✅ Removed tanh activation (allows unbounded utility predictions)
- ❌ NO weighted loss (standard MSE, unlike V4.5 which uses 20x weight)

This isolates the effect of tanh removal alone.

Features: Dynamic (9 dims) + early entropy (2 dims) = 11 dims total

Purpose: Compare V4.6 (no weight) vs V4.5 (20x weight) to see if
weighted loss is necessary or if tanh removal alone is sufficient.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


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


class ACCProbe(nn.Module):
    """
    ACC-Inspired probe with compact input and value-aware output.
    
    Input: 11 dims (dynamic + entropy only, no compressed hidden)
    Output: 3 predictions (wrong, conflict, utility)
    
    Much smaller than V2/V3 (16,389 dims) for better generalization.
    """
    def __init__(
        self,
        input_dim: int = 11,  # 9 dynamic + 2 entropy (no compressed hidden!)
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Compact 2-layer MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 64)
        
        # Three prediction heads
        self.wrong_head = nn.Linear(64, 1)      # Binary: P(baseline wrong)
        self.conflict_head = nn.Linear(64, 1)   # Continuous: conflict/instability
        self.utility_head = nn.Linear(64, 1)    # Continuous: expected utility gain
        
        # Initialize weights
        for layer in [self.fc1, self.fc2, self.wrong_head, self.conflict_head, self.utility_head]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Shared encoder
        x = self.fc1(features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Three predictions
        wrong_logits = self.wrong_head(x).squeeze(-1)
        conflict_score = torch.sigmoid(self.conflict_head(x)).squeeze(-1)  # [0, 1]
        utility_score = self.utility_head(x).squeeze(-1)  # No tanh! Unbounded output
        
        return wrong_logits, conflict_score, utility_score


class EnsembleACCProbe(nn.Module):
    """Ensemble of multiple ACC probes for improved robustness."""
    def __init__(self, num_probes: int = 3, **probe_kwargs):
        super().__init__()
        self.probes = nn.ModuleList([ACCProbe(**probe_kwargs) for _ in range(num_probes)])
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wrong_preds = []
        conflict_preds = []
        utility_preds = []
        
        for probe in self.probes:
            wrong, conflict, utility = probe(features)
            wrong_preds.append(wrong)
            conflict_preds.append(conflict)
            utility_preds.append(utility)
        
        # Average predictions
        ensemble_wrong = torch.stack(wrong_preds, dim=0).mean(dim=0)
        ensemble_conflict = torch.stack(conflict_preds, dim=0).mean(dim=0)
        ensemble_utility = torch.stack(utility_preds, dim=0).mean(dim=0)
        
        return ensemble_wrong, ensemble_conflict, ensemble_utility


class ACCDataset(Dataset):
    """Dataset for ACC probe training."""
    def __init__(self, tensor_data: Dict[str, torch.Tensor]):
        self.features = tensor_data['features']
        self.wrong_labels = tensor_data['wrong_labels']
        self.conflict_labels = tensor_data['conflict_labels']
        self.utility_labels = tensor_data['utility_labels']
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'wrong_label': self.wrong_labels[idx],
            'conflict_label': self.conflict_labels[idx],
            'utility_label': self.utility_labels[idx],
        }


def train_acc_probe(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    device: str,
    epochs: int,
    learning_rate: float,
    lambda_wrong: float = 1.0,
    lambda_conflict: float = 0.3,
    lambda_utility: float = 1.5,  # Higher weight for utility (main routing signal)
) -> Tuple[nn.Module, Dict]:
    """Train ACC probe with multi-task loss."""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    mse_loss = nn.MSELoss()
    
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
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            features = batch['features'].to(device)
            wrong_labels = batch['wrong_label'].to(device)
            conflict_labels = batch['conflict_label'].to(device)
            utility_labels = batch['utility_label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            wrong_logits, conflict_pred, utility_pred = model(features)
            
            # Multi-task loss
            loss_wrong = focal_loss(wrong_logits, wrong_labels)
            loss_conflict = mse_loss(conflict_pred, conflict_labels)
            loss_utility = mse_loss(utility_pred, utility_labels)
            
            total_loss = (
                lambda_wrong * loss_wrong +
                lambda_conflict * loss_conflict +
                lambda_utility * loss_utility
            )
            
            # Backward pass
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
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                features = batch['features'].to(device)
                wrong_labels = batch['wrong_label'].to(device)
                conflict_labels = batch['conflict_label'].to(device)
                utility_labels = batch['utility_label'].to(device)
                
                # Forward pass
                wrong_logits, conflict_pred, utility_pred = model(features)
                
                # Multi-task loss
                loss_wrong = focal_loss(wrong_logits, wrong_labels)
                loss_conflict = mse_loss(conflict_pred, conflict_labels)
                loss_utility = mse_loss(utility_pred, utility_labels)
                
                total_loss = (
                    lambda_wrong * loss_wrong +
                    lambda_conflict * loss_conflict +
                    lambda_utility * loss_utility
                )
                
                val_losses.append(total_loss.item())
                
                # Collect predictions
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
        
        # Utility correlation (key metric for value-aware routing)
        utility_corr, _ = pearsonr(all_utility_preds, all_utility_labels)
        if np.isnan(utility_corr):
            utility_corr = 0.0
        
        # Save best model (by utility correlation, not just F1)
        if utility_corr > best_val_utility_corr:
            best_val_utility_corr = utility_corr
            best_model_state = model.state_dict().copy()
        
        # Log
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_wrong_f1'].append(val_wrong_f1)
        history['val_utility_corr'].append(utility_corr)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Wrong F1: {val_wrong_f1:.4f} (P={val_precision:.3f}, R={val_recall:.3f})")
        print(f"  Val Utility Corr: {utility_corr:.4f} {'*' if utility_corr == best_val_utility_corr else ''}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train ACC-inspired probe (V4)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu", "hellaswag"])
    parser.add_argument("--use_ensemble", action="store_true")
    parser.add_argument("--num_probes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lambda_wrong", type=float, default=1.0)
    parser.add_argument("--lambda_conflict", type=float, default=0.3)
    parser.add_argument("--lambda_utility", type=float, default=1.5)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data from all benchmarks
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
        
        # V4.1: Extract only dynamic features + early entropy (skip compressed hidden)
        full_features = data['features']  # Shape: [N, 267]
        dynamic_entropy = full_features[:, 256:]  # Skip first 256 dims (compressed hidden)
        # dynamic_entropy shape: [N, 11] (9 dynamic + 2 entropy)
        
        all_features.append(dynamic_entropy)
        all_wrong_labels.append(data['wrong_labels'])
        all_conflict_labels.append(data['conflict_labels'])
        all_utility_labels.append(data['utility_labels'])
    
    # Combine all data
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
    
    # Create dataset
    dataset = ACCDataset(combined_data)
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print("\nCreating model...")
    input_dim = combined_data['features'].shape[1]
    
    if args.use_ensemble:
        print(f"Using ensemble of {args.num_probes} probes")
        model = EnsembleACCProbe(num_probes=args.num_probes, input_dim=input_dim)
    else:
        model = ACCProbe(input_dim=input_dim)
    
    print(f"Input dimensions: {input_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nTraining...")
    model, history = train_acc_probe(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_wrong=args.lambda_wrong,
        lambda_conflict=args.lambda_conflict,
        lambda_utility=args.lambda_utility,
    )
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_path = os.path.join(args.output_dir, "best_probe.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to: {model_path}")
    
    # Save config
    config = {
        'input_dim': input_dim,
        'hidden_dim': 128,
        'dropout': 0.1,
        'use_ensemble': args.use_ensemble,
        'num_probes': args.num_probes if args.use_ensemble else 1,
        'lambda_wrong': args.lambda_wrong,
        'lambda_conflict': args.lambda_conflict,
        'lambda_utility': args.lambda_utility,
    }
    
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")
    
    # Save history (convert numpy types to Python types for JSON serialization)
    history_serializable = {
        'train_loss': [float(x) for x in history['train_loss']],
        'val_loss': [float(x) for x in history['val_loss']],
        'val_wrong_f1': [float(x) for x in history['val_wrong_f1']],
        'val_utility_corr': [float(x) for x in history['val_utility_corr']],
    }
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    print(f"Saved training history to: {history_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation utility correlation: {max(history['val_utility_corr']):.4f}")
    print(f"Best validation wrong F1: {max(history['val_wrong_f1']):.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
