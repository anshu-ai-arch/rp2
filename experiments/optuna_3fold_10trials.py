import os
import sys
import gc
import time
import json
import csv
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import optuna
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Tuple, List, Dict

sys.path.append(os.getcwd())

from experiments.run_model_c_medium_kfold_minority import (
    load_ds1_record_data,
    DS1_ALL_PATIENTS,
    ArrhythmiaDatasetRaw,
    compute_active_metrics
)
from train_augmented_model_c import AugmentedECGDataset
from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU
from experiments.evaluate_patient_wise import get_m1_device

# Exact 3 validation folds specified for this experiment
EXPLORATORY_3FOLDS = {
    1: ['207', '118', '106', '112'],
    2: ['208', '209', '114', '115'],
    3: ['223', '201', '109', '122']
}


def get_3fold_dataloaders(
    record_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    val_fold_idx: int,
    batch_size: int = 256,
    sampling_alpha: float = 0.5
) -> Tuple[DataLoader, DataLoader, Dict[int, int], Dict[int, int], np.ndarray, float]:
    val_records = EXPLORATORY_3FOLDS[val_fold_idx]
    train_records = [r for r in DS1_ALL_PATIENTS if r not in val_records]

    # Combine beats for train records
    train_1d_list = [record_data[r][0] for r in train_records]
    train_y_list = [record_data[r][1] for r in train_records]

    X_train = np.concatenate(train_1d_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)

    # Combine beats for val records
    val_1d_list = [record_data[r][0] for r in val_records]
    val_y_list = [record_data[r][1] for r in val_records]

    X_val = np.concatenate(val_1d_list, axis=0)
    y_val = np.concatenate(val_y_list, axis=0)

    # Count class frequencies in train
    class_counts_train = {c: int(np.sum(y_train == c)) for c in range(5)}
    class_counts_val = {c: int(np.sum(y_val == c)) for c in range(5)}

    # Calculate per-class sampling weights w_c = (1 / N_c)^alpha for c in 0..3, w_4 = 0.0
    class_weights = np.zeros(5, dtype=np.float64)
    for c in range(4):
        if class_counts_train[c] > 0:
            class_weights[c] = (1.0 / class_counts_train[c]) ** sampling_alpha
        else:
            class_weights[c] = 0.0
    class_weights[4] = 0.0  # Q class weight explicitly ZERO

    # Sample weights and WeightedRandomSampler
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    # Calculate theoretical expected sampling proportions
    total_active_weight = sum(class_counts_train[c] * class_weights[c] for c in range(4))
    theo_props = {c: (class_counts_train[c] * class_weights[c]) / total_active_weight for c in range(5)}

    train_ds_raw = ArrhythmiaDatasetRaw(X_train, y_train)
    val_ds_raw = ArrhythmiaDatasetRaw(X_val, y_val)

    augmented_train_ds = AugmentedECGDataset(train_ds_raw, is_train=True)

    train_loader = DataLoader(
        augmented_train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds_raw,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, class_counts_train, class_counts_val, class_weights, theo_props


def train_eval_single_fold(
    record_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    fold_idx: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    sampling_alpha: float,
    device: torch.device,
    num_epochs: int = 20,
    save_checkpoint_path: Path = None
) -> Dict:
    train_loader, val_loader, train_counts, val_counts, class_weights, theo_props = get_3fold_dataloaders(
        record_data, val_fold_idx=fold_idx, batch_size=batch_size, sampling_alpha=sampling_alpha
    )

    # Initialize FRESH model from scratch for this fold
    model = GenericHybrid1DBiCNNGRU(
        in_channels=1,
        cnn_channels=[64, 128, 128],
        kernel_sizes=[5, 5, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.2076,
        num_classes=5
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=4)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_score = -999.0
    best_metrics = {}
    best_epoch = 0
    best_conf_mat = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_total = 0.0, 0

        for batch in train_loader:
            _, b1d, targets = batch
            b1d = b1d.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(b1d)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            train_total += targets.size(0)

        # Evaluate on validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                _, b1d, targets = batch
                b1d = b1d.to(device)
                outputs = model(b1d)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(targets.numpy())

        all_preds = np.concatenate(val_preds, axis=0)
        all_targets = np.concatenate(val_targets, axis=0)

        metrics = compute_active_metrics(all_preds, all_targets)
        scheduler.step(metrics["final_score"])

        # Compute active confusion matrix
        active_mask = (all_targets != 4)
        active_p = all_preds[active_mask]
        active_t = all_targets[active_mask]
        
        conf_mat = np.zeros((4, 4), dtype=int)
        for t, p in zip(active_t, active_p):
            if t < 4 and p < 4:
                conf_mat[t, p] += 1

        if metrics["final_score"] > best_score:
            best_score = metrics["final_score"]
            best_metrics = metrics
            best_epoch = epoch
            best_conf_mat = conf_mat.tolist()

            if save_checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "metrics": metrics,
                        "confusion_matrix": best_conf_mat,
                        "hyperparameters": {
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "batch_size": batch_size,
                            "sampling_alpha": sampling_alpha,
                            "fold": fold_idx
                        }
                    },
                    save_checkpoint_path
                )

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold_idx": fold_idx,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "metrics": best_metrics,
        "confusion_matrix": best_conf_mat,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "class_weights": class_weights.tolist(),
        "theo_props": theo_props
    }


def run_optuna_study():
    device = get_m1_device()
    print("\n=================================================================")
    print("   3-FOLD EXPLORATORY OPTUNA STUDY (10 TRIALS, DS1 PATIENTS ONLY)  ")
    print("=================================================================")
    print(f"[*] Target Acceleration Device: {device}")

    # Load DS1 data
    print("[*] Loading DS1 patient record data...")
    record_data = load_ds1_record_data("config.yaml")

    checkpoints_dir = Path("checkpoints/ecg_optuna_3fold_10trials")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    study_results = []
    best_study_score = -999.0
    best_trial_number = -1

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_study_score, best_trial_number

        lr = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        sampling_alpha = trial.suggest_float("sampling_alpha", 0.0, 1.0)
        batch_size = trial.suggest_categorical("batch_size", [128, 256])

        print(f"\n=================================================================")
        print(f"Starting Trial {trial.number + 1}/10 | lr={lr:.6f} | wd={weight_decay:.6e} | alpha={sampling_alpha:.4f} | bs={batch_size}")
        print("=================================================================")

        fold_scores = []
        fold_results = {}

        for fold_idx in range(1, 4):
            print(f" -> Trial {trial.number + 1}/10 | Fold {fold_idx}/3 (20 Epochs)...")
            ckpt_path = checkpoints_dir / f"trial_{trial.number + 1}_fold_{fold_idx}_best.pth"
            res = train_eval_single_fold(
                record_data=record_data,
                fold_idx=fold_idx,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                sampling_alpha=sampling_alpha,
                device=device,
                num_epochs=20,
                save_checkpoint_path=ckpt_path
            )
            fold_scores.append(res["best_score"])
            fold_results[f"fold_{fold_idx}"] = res
            print(f"    Fold {fold_idx}/3 Best Score: {res['best_score']:.4f} (Epoch {res['best_epoch']}) | Acc: {res['metrics']['active_accuracy']:.2f}% | Macro F1: {res['metrics']['active_macro_f1']:.2f}% | Min F1: {res['metrics']['minority_macro_f1']:.2f}% | N Rec: {res['metrics']['normal_recall']:.2f}%")

        mean_trial_score = float(np.mean(fold_scores))
        std_trial_score = float(np.std(fold_scores))

        mean_macro_f1 = float(np.mean([fold_results[f"fold_{k}"]["metrics"]["active_macro_f1"] for k in range(1, 4)]))
        mean_minority_f1 = float(np.mean([fold_results[f"fold_{k}"]["metrics"]["minority_macro_f1"] for k in range(1, 4)]))
        mean_n_recall = float(np.mean([fold_results[f"fold_{k}"]["metrics"]["normal_recall"] for k in range(1, 4)]))
        mean_accuracy = float(np.mean([fold_results[f"fold_{k}"]["metrics"]["active_accuracy"] for k in range(1, 4)]))

        trial_summary = {
            "trial_number": trial.number + 1,
            "objective_score": mean_trial_score,
            "std_objective_score": std_trial_score,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "sampling_alpha": sampling_alpha,
            "batch_size": batch_size,
            "mean_macro_f1": mean_macro_f1,
            "mean_minority_f1": mean_minority_f1,
            "mean_n_recall": mean_n_recall,
            "mean_accuracy": mean_accuracy,
            "fold_results": fold_results
        }
        study_results.append(trial_summary)

        print(f"\n[✓] Trial {trial.number + 1}/10 Complete | Mean Score: {mean_trial_score:.4f} | Alpha: {sampling_alpha:.4f} | LR: {lr:.6f} | WD: {weight_decay:.6e} | BS: {batch_size}")
        print(f" -> Mean Active Macro F1: {mean_macro_f1:.2f}% | Mean Minority F1: {mean_minority_f1:.2f}% | Mean N Recall: {mean_n_recall:.2f}%")

        if mean_trial_score > best_study_score:
            best_study_score = mean_trial_score
            best_trial_number = trial.number + 1

        return mean_trial_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=10)

    print("\n=================================================================")
    print("             OPTUNA 10-TRIAL STUDY COMPLETE                      ")
    print("=================================================================")
    print(f"[*] Best Trial: Trial {best_trial_number} (Mean Score: {best_study_score:.4f})")
    print(f"[*] Best Parameters:")
    best_params = study.best_params
    for k, v in best_params.items():
        print(f"    - {k}: {v}")

    # Save summary CSV
    csv_file = Path("experiments/optuna_3fold_10trials_summary.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Trial", "Objective Score", "Std Score", "Learning Rate", "Weight Decay",
            "Sampling Alpha", "Batch Size", "Mean Active Macro F1", "Mean Minority Macro F1",
            "Mean N Recall", "Mean Active Accuracy"
        ])
        for t in study_results:
            writer.writerow([
                t["trial_number"], f"{t['objective_score']:.4f}", f"{t['std_objective_score']:.4f}",
                f"{t['learning_rate']:.6e}", f"{t['weight_decay']:.6e}", f"{t['sampling_alpha']:.4f}",
                t["batch_size"], f"{t['mean_macro_f1']:.2f}", f"{t['mean_minority_f1']:.2f}",
                f"{t['mean_n_recall']:.2f}", f"{t['mean_accuracy']:.2f}"
            ])
    print(f"[✓] Saved trial summary CSV to '{csv_file}'")

    # Save complete JSON report
    json_file = Path("experiments/results_ecg_optuna_3fold_10trials.json")
    with open(json_file, "w") as f:
        json.dump(study_results, f, indent=2)
    print(f"[✓] Saved full results JSON to '{json_file}'")

    return study_results, best_params


if __name__ == "__main__":
    run_optuna_study()
