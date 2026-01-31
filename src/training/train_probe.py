#!/usr/bin/env python3
"""
Train the confidence probe on generated training data.

This script:
1. Loads hidden states and labels from generated training data
2. Trains a 2-layer MLP probe to predict confidence labels
3. Saves the trained probe for use in evaluation

Usage:
    python -m src.training.train_probe \
        --data_dir data/training/probe_data \
        --output_dir models/confidence_probe \
        --epochs 20 \
        --batch_size 64
"""

import argparse
import json
import os
import random
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np


class ConfidenceProbe(nn.Module):
    """
    Two-layer MLP probe for predicting confidence from hidden states.
    
    Input: hidden_state (4096 dimensions for Llama-3.1-8B)
    Output: logit for binary classification (0=high confidence, 1=low confidence)
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layer1 = nn.Linear(hidden_size, intermediate_size)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(intermediate_size, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_state: Tensor of shape (batch_size, hidden_size)
        
        Returns:
            logits: Tensor of shape (batch_size, 1)
        """
        x = self.layer1(hidden_state)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.layer2(x)
        return logits
    
    def predict_confidence(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict confidence label and probability.
        
        Args:
            hidden_state: Tensor of shape (batch_size, hidden_size)
        
        Returns:
            labels: Predicted labels (0=high confidence, 1=low confidence)
            probs: Probability of low confidence
        """
        logits = self.forward(hidden_state)
        probs = torch.sigmoid(logits)
        labels = (probs > 0.5).long().squeeze(-1)
        return labels, probs.squeeze(-1)


class ProbeDataset(Dataset):
    """Dataset for training the confidence probe."""
    
    def __init__(self, hidden_states: torch.Tensor, labels: torch.Tensor):
        self.hidden_states = hidden_states
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.hidden_states[idx], self.labels[idx]


def load_training_data(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Load training data from generated files."""
    
    tensors_path = os.path.join(data_dir, "training_tensors.pt")
    metadata_path = os.path.join(data_dir, "training_metadata.json")
    
    print(f"Loading training tensors from: {tensors_path}")
    data = torch.load(tensors_path)
    hidden_states = data['hidden_states']
    labels = data['labels']
    
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(labels)} samples")
    print(f"Hidden state shape: {hidden_states.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return hidden_states, labels, metadata


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Compute class weights for imbalanced data."""
    n_samples = len(labels)
    n_positive = labels.sum().item()
    n_negative = n_samples - n_positive
    
    # Weight for positive class (low confidence)
    # Higher weight for minority class
    if n_positive > 0 and n_negative > 0:
        pos_weight = n_negative / n_positive
    else:
        pos_weight = 1.0
    
    print(f"Class distribution: {n_negative} high confidence, {n_positive} low confidence")
    print(f"Positive class weight: {pos_weight:.2f}")
    
    return torch.tensor([pos_weight])


def train_epoch(
    model: ConfidenceProbe,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for hidden_states, labels in dataloader:
        hidden_states = hidden_states.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        
        logits = model(hidden_states).squeeze(-1)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        
        # Calculate accuracy
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += len(labels)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: ConfidenceProbe,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float, Dict]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for hidden_states, labels in dataloader:
            hidden_states = hidden_states.to(device)
            labels = labels.float().to(device)
            
            logits = model(hidden_states).squeeze(-1)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * len(labels)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            
            correct += (preds == labels.long()).sum().item()
            total += len(labels)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    # Calculate per-class metrics
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
    }
    
    return avg_loss, accuracy, metrics


def train_probe(
    data_dir: str,
    output_dir: str,
    hidden_size: int = 4096,
    intermediate_size: int = 256,
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Train the confidence probe."""
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    hidden_states, labels, metadata = load_training_data(data_dir)
    
    # Create dataset
    dataset = ProbeDataset(hidden_states, labels)
    
    # Split into train/val
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Compute class weights
    pos_weight = compute_class_weights(labels)
    
    # Initialize model
    model = ConfidenceProbe(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dropout=dropout
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    best_val_f1 = 0.0
    best_epoch = 0
    history = []
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_metrics = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            **val_metrics
        })
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}, "
              f"P={val_metrics['precision']:.4f}, R={val_metrics['recall']:.4f}, "
              f"F1={val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': {
                    'hidden_size': hidden_size,
                    'intermediate_size': intermediate_size,
                    'dropout': dropout,
                }
            }, os.path.join(output_dir, "best_probe.pt"))
            
            print(f"  *** New best model (F1={best_val_f1:.4f}) ***")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'dropout': dropout,
        }
    }, os.path.join(output_dir, "final_probe.pt"))
    
    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest epoch: {best_epoch}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"\nSaved to: {output_dir}")
    print(f"  - best_probe.pt: Best model checkpoint")
    print(f"  - final_probe.pt: Final model checkpoint")
    print(f"  - training_history.json: Training metrics")


def main():
    parser = argparse.ArgumentParser(description="Train confidence probe")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/training/probe_data",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/confidence_probe",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=4096,
        help="Hidden state size (4096 for Llama-3.1-8B)"
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=256,
        help="Intermediate layer size"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    train_probe(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
