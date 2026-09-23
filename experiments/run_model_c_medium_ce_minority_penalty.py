import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from src.data.dataset import get_dataloaders
from train_augmented_model_c import AugmentedECGDataset
from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU
from experiments.evaluate_patient_wise import evaluate_model_patient_wise, get_m1_device


class CEMinorityPenaltyLoss(nn.Module):
    """
    Cross Entropy Loss + Minority-Class Auxiliary Penalty (Classes 1: SVEB/A & 3: F/VT).
    
    Formula:
        L_CE = mean(CE_per_sample)
        L_minority = mean(CE_per_sample[target in {1, 3}])
        L_total = L_CE + 0.10 * L_minority
    """

    def __init__(self, lmbda: float = 0.10, minority_classes: list = [1, 3], reduction: str = 'mean'):
        super().__init__()
        self.lmbda = lmbda
        self.minority_classes = minority_classes
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_per_sample = F.cross_entropy(inputs, targets, reduction='none')
        base_ce = ce_per_sample.mean()

        minority_mask = torch.zeros_like(targets, dtype=torch.bool)
        for c in self.minority_classes:
            minority_mask = minority_mask | (targets == c)

        if minority_mask.any():
            minority_penalty = ce_per_sample[minority_mask].mean()
        else:
            minority_penalty = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)

        total_loss = base_ce + self.lmbda * minority_penalty
        return total_loss


def run_experiment_ce_minority_penalty(num_epochs: int = 20):
    device = get_m1_device()
    print("\n=================================================================")
    print("   EXPERIMENT B — MEDIUM CNN WITH CE + MINORITY PENALTY (lmbda=0.10)")
    print("=================================================================")
    print(f"[*] Target Acceleration Device: {device}")
    print(f"[*] CNN Architecture: Conv1D(1, 64, 5) -> Conv1D(64, 128, 5) -> Conv1D(128, 128, 3)")
    print(f"[*] BiGRU Architecture: 2-layer BiGRU, hidden_size=64 per direction")
    print(f"[*] Online Augmentation: Scaling [0.85, 1.15], Noise N(0, 0.02), Shift [-5, +5]")
    print(f"[*] Loss Function: CEMinorityPenaltyLoss(lmbda=0.10, minority_classes=[1, 3])\n")

    base_train_loader, val_loader, test_loader, config = get_dataloaders("config.yaml")

    augmented_train_dataset = AugmentedECGDataset(base_train_loader.dataset, is_train=True)
    train_loader = DataLoader(
        augmented_train_dataset,
        batch_size=base_train_loader.batch_size,
        shuffle=True,
        num_workers=0
    )

    model = GenericHybrid1DBiCNNGRU(
        in_channels=1,
        cnn_channels=[64, 128, 128],
        kernel_sizes=[5, 5, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.2076,
        num_classes=5
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Trainable Parameter Count: {trainable_params:,}")

    # ONLY CHANGE: Replace nn.CrossEntropyLoss() with CEMinorityPenaltyLoss(lmbda=0.10, minority_classes=[1, 3])
    criterion = CEMinorityPenaltyLoss(lmbda=0.10, minority_classes=[1, 3])
    optimizer = optim.Adam(model.parameters(), lr=0.002018, weight_decay=2.35e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc = 0.0
    checkpoint_path = Path("checkpoints/ecg_ce_minority_penalty_best.pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("[*] Starting Training on Augmented DS1 Set...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch in train_loader:
            if len(batch) == 4:
                b2d, b1d, brr, targets = batch
            else:
                b2d, b1d, targets = batch

            b1d = b1d.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(b1d)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    b2d, b1d, brr, targets = batch
                else:
                    b2d, b1d, targets = batch

                b1d = b1d.to(device)
                targets = targets.to(device)
                outputs = model(b1d)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = (val_correct / val_total) * 100.0
        scheduler.step(val_acc)

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "trainable_params": trainable_params,
                    "cnn_channels": [64, 128, 128],
                    "loss_function": "CEMinorityPenaltyLoss(lmbda=0.10, minority_classes=[1, 3])"
                },
                checkpoint_path
            )

    print(f"\n[✓] CE + Minority Penalty Experiment Training Complete! Best Val Acc: {best_val_acc:.2f}%")
    print(f"[*] Checkpoint saved to '{checkpoint_path}'")

    # Evaluate on DS2
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    summary, patient_breakdown = evaluate_model_patient_wise(model)

    results_file = Path("experiments/results_ecg_ce_minority_penalty.json")
    save_summary = {
        "model_name": "Experiment B (Medium CNN + CE Minority Penalty lmbda=0.10)",
        "trainable_params": trainable_params,
        "size_mb": (trainable_params * 4) / (1024.0 * 1024.0),
        "val_acc": best_val_acc,
        "ds2_accuracy": summary["accuracy"],
        "ds2_weighted_f1": summary["weighted_f1"],
        "ds2_macro_f1": summary["macro_f1"],
        "precision": summary["precision"].tolist(),
        "recall": summary["recall"].tolist(),
        "f1": summary["f1"].tolist(),
        "support": summary["support"].tolist(),
        "patient_stats": summary["patient_stats"],
        "confusion_matrix": summary["confusion_matrix"].tolist(),
        "confusion_matrix_norm": summary["confusion_matrix_norm"].tolist()
    }
    with open(results_file, "w") as f:
        json.dump(save_summary, f, indent=2)

    print(f"[✓] Saved CE Minority Penalty evaluation results to '{results_file}'")
    return save_summary


if __name__ == "__main__":
    run_experiment_ce_minority_penalty(num_epochs=20)
