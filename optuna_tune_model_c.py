import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report
from typing import Dict, Any, Tuple

from src.data.dataset import get_dataloaders
from src.utils.metrics import compute_metrics


# =====================================================================
# 1. HARDWARE SELECTION (Apple Mac M1 Acceleration)
# =====================================================================
def get_m1_device() -> torch.device:
    """Selects Apple Silicon MPS acceleration if available, fallback to CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


# =====================================================================
# 2. LOSS FUNCTIONS (Focal Loss & Weighted Cross-Entropy)
# =====================================================================
class PyTorchFocalLoss(nn.Module):
    """
    PyTorch Focal Loss implementation for severe class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


# =====================================================================
# 3. DYNAMIC MODEL SPECIFICATION (Model C: Hybrid1DBiCNNGRU)
# =====================================================================
class DynamicHybrid1DBiCNNGRU(nn.Module):
    """
    Flexible Model C (Hybrid1DBiCNNGRU) supporting dynamic architectural parameters
    for Optuna hyperparameter optimization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_out_channels: int = 32,
        conv_kernel_size: int = 5,
        gru_hidden_size: int = 64,
        gru_num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 5
    ):
        super().__init__()

        padding = conv_kernel_size // 2

        # 3-Stage 1D CNN Backbone
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, conv_out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_out_channels, conv_out_channels * 2, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(conv_out_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(conv_out_channels * 2, conv_out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_out_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        gru_input_dim = conv_out_channels * 2

        # Bidirectional GRU Recurrent Layer
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Dense Classifier Head (Bi-GRU output = gru_hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        # 1D CNN Feature Extractor
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Permute to (Batch, Time, Channels) for GRU
        x = x.permute(0, 2, 1)

        # Bi-GRU temporal processing
        gru_out, _ = self.gru(x)

        # Global Average Temporal Pooling across timesteps
        pooled = torch.mean(gru_out, dim=1)

        # Dense classification
        logits = self.classifier(pooled)
        return logits


# =====================================================================
# 4. OPTUNA OBJECTIVE FUNCTION & TUNING LOOP
# =====================================================================
def objective(trial: optuna.Trial, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> float:
    """
    Optuna Trial Objective Function optimizing for Validation Macro F1-Score on Model C.
    Includes MedianPruner early stopping and Apple Silicon MPS memory clearing.
    """
    # -----------------------------------------------------------------
    # A. Suggest Hyperparameters
    # -----------------------------------------------------------------
    conv_out_channels = trial.suggest_categorical('conv_out_channels', [16, 32, 64])
    conv_kernel_size = trial.suggest_categorical('conv_kernel_size', [3, 5, 7])
    gru_hidden_size = trial.suggest_categorical('gru_hidden_size', [32, 64, 128])
    gru_num_layers = trial.suggest_int('gru_num_layers', 1, 2)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    loss_type = trial.suggest_categorical('loss_type', ['focal', 'weighted_ce'])

    # -----------------------------------------------------------------
    # B. Instantiate Model & Move to Device (MPS/CPU)
    # -----------------------------------------------------------------
    model = DynamicHybrid1DBiCNNGRU(
        in_channels=1,
        conv_out_channels=conv_out_channels,
        conv_kernel_size=conv_kernel_size,
        gru_hidden_size=gru_hidden_size,
        gru_num_layers=gru_num_layers,
        dropout=dropout,
        num_classes=5
    ).to(device)

    # -----------------------------------------------------------------
    # C. Loss Function Setup
    # -----------------------------------------------------------------
    alpha_weights = torch.tensor([1.0, 10.0, 3.0, 10.0, 50.0], dtype=torch.float32).to(device)

    if loss_type == 'focal':
        focal_gamma = trial.suggest_float('focal_gamma', 1.5, 3.5)
        criterion = PyTorchFocalLoss(alpha=alpha_weights, gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=alpha_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    num_epochs = 15
    best_val_macro_f1 = 0.0

    # -----------------------------------------------------------------
    # D. Epoch Training & Evaluation Loop
    # -----------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        # 1. Train Step
        model.train()
        for batch in train_loader:
            if len(batch) == 4:
                batch_2d, batch_1d, batch_rr, targets = batch
            else:
                batch_2d, batch_1d, targets = batch

            batch_1d = batch_1d.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_1d)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # 2. Validation Step (Macro F1-Score Primary Metric)
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    batch_2d, batch_1d, batch_rr, targets = batch
                else:
                    batch_2d, batch_1d, targets = batch

                batch_1d = batch_1d.to(device)
                targets = targets.to(device)

                outputs = model(batch_1d)
                preds = outputs.argmax(dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # Compute Macro F1-Score
        val_macro_f1 = float(f1_score(val_targets, val_preds, average="macro", zero_division=0))
        scheduler.step(val_macro_f1)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1

        # 3. Report to Optuna for MedianPruner Early Stopping
        trial.report(val_macro_f1, epoch)

        # Handle pruning based on validation Macro F1
        if trial.should_prune():
            if device.type == "mps":
                torch.mps.empty_cache()
            raise optuna.exceptions.TrialPruned()

    # -----------------------------------------------------------------
    # E. Memory Cleanup for Mac M1
    # -----------------------------------------------------------------
    del model, optimizer, scheduler, criterion
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return best_val_macro_f1


# =====================================================================
# 5. MAIN EXECUTION PIPELINE
# =====================================================================
def run_optuna_study(n_trials: int = 10):
    """Launches Optuna Study with MedianPruner targeting Validation Macro F1-Score."""
    device = get_m1_device()
    print(f"\n=================================================================")
    print(f"    OPTUNA HYPERPARAMETER TUNING FOR MODEL C (Hybrid1DBiCNNGRU)  ")
    print(f"=================================================================")
    print(f"[*] Target Hardware Device: {device}")
    print(f"[*] Optimization Target Metric: Validation Macro F1-Score (MAXIMIZE)")
    print(f"[*] Total Trials Requested: {n_trials}\n")

    # Load Data Loaders
    train_loader, val_loader, test_loader, config = get_dataloaders("config.yaml")

    # Configure MedianPruner
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    # Create Optuna Study
    study = optuna.create_study(
        study_name="model_c_bi_cnn_gru_optimization",
        direction="maximize",
        pruner=pruner
    )

    # Launch Tuning Study
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("\n[✓] Optuna Study Complete!")
    print(f"[*] Best Trial Number: #{study.best_trial.number}")
    print(f"[*] Best Validation Macro F1-Score: {study.best_value:.4f}")
    print(f"\n[*] Best Hyperparameters Found:")
    for key, val in study.best_trial.params.items():
        print(f"    - {key}: {val}")

    return study


if __name__ == "__main__":
    run_optuna_study(n_trials=10)
