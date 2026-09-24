"""
Short 2-Fold Patient-Level Stratified Optuna Pilot (Alpha Search).

RESEARCH GOAL:
Run a short 2-fold patient-level stratified Optuna pilot study to search sampling_alpha
in [0.0, 1.0] along with learning_rate, weight_decay, and batch_size, evaluating whether
alpha tuning improves minority-class performance while preserving Normal recall.

PILOT EXPERIMENT SPECIFICATION:
1. DS1 patient set (22 records) partitioned into 2 stratified validation folds (11 patients each):
   - Fold 1: ['101', '112', '114', '115', '118', '119', '122', '207', '208', '220', '230']
   - Fold 2: ['106', '108', '109', '116', '124', '201', '203', '205', '209', '215', '223']
2. DS2 test set is COMPLETELY LOCKED AND UNTOUCHED.
3. Model: GenericHybrid1DBiCNNGRU (257,541 parameters).
4. Optuna Search Space:
   - sampling_alpha: float in [0.0, 1.0]
   - learning_rate:  float in [5e-4, 5e-3] (log scale)
   - weight_decay:   float in [1e-6, 1e-3] (log scale)
   - batch_size:     categorical in [128, 256]
5. Max 10 epochs per fold per trial.
6. Global Wall-Clock Limit: Default 3600 seconds (1 hour). Stop launching new trials after timeout.
7. Active classes: N (0), SVEB/A (1), VEB/PVC (2), F/VT (3).
8. Q class (4) is 100% EXCLUDED from training loss (ignore_index=4), sampler (w_4=0.0), and active metrics.
"""

import os
import sys
import gc
import time
import json
import psutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import optuna
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Tuple, List, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import get_device
from experiments.run_model_c_medium_kfold_minority import (
    load_ds1_record_data,
    DS1_ALL_PATIENTS,
    ArrhythmiaDatasetRaw,
    compute_active_metrics
)
from train_augmented_model_c import AugmentedECGDataset
from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU

# Locked DS2 Test Recordings (Must NEVER enter CV)
DS2_TEST_RECORDINGS = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234"
]

# Patient-Level Stratified 2-Fold Partition
PILOT_2FOLDS = {
    1: ['101', '112', '114', '115', '118', '119', '122', '207', '208', '220', '230'],
    2: ['106', '108', '109', '116', '124', '201', '203', '205', '209', '215', '223']
}


def print_cuda_info():
    """
    Prints CUDA device status and information.
    """
    print("\n=================================================================")
    print("                CUDA & HARDWARE ENVIRONMENT CHECK                ")
    print("=================================================================")
    cuda_avail = torch.cuda.is_available()
    print(f"[*] torch.cuda.is_available(): {cuda_avail}")
    if cuda_avail:
        device_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[*] Device Count:               {device_count}")
        print(f"[*] GPU Model Name:             {gpu_name}")
    elif torch.backends.mps.is_available():
        print("[*] Device Type:                 Apple Silicon Acceleration (MPS)")
    else:
        print("[*] Device Type:                 CPU Fallback")
    print("=================================================================\n")


def audit_fold_construction(record_data: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """
    Performs an explicit fold-balance audit and verifies structural research assertions.
    """
    print("=================================================================")
    print("        PATIENT-LEVEL STRATIFIED 2-FOLD FOLD-BALANCE AUDIT       ")
    print("=================================================================")

    all_val_patients = []
    patient_counts_per_fold = {}

    for fold_idx in range(1, 3):
        val_recs = PILOT_2FOLDS[fold_idx]
        train_recs = [r for r in DS1_ALL_PATIENTS if r not in val_recs]
        all_val_patients.extend(val_recs)

        # Assertion: Patient disjointness
        overlap = set(val_recs).intersection(set(train_recs))
        assert len(overlap) == 0, f"Patient overlap detected in Fold {fold_idx}: {overlap}"

        # Assertion: DS2 isolation
        ds2_overlap_val = set(val_recs).intersection(set(DS2_TEST_RECORDINGS))
        ds2_overlap_train = set(train_recs).intersection(set(DS2_TEST_RECORDINGS))
        assert len(ds2_overlap_val) == 0, f"DS2 leakage in Fold {fold_idx} val: {ds2_overlap_val}"
        assert len(ds2_overlap_train) == 0, f"DS2 leakage in Fold {fold_idx} train: {ds2_overlap_train}"

        # Beat counts for validation and train
        val_y = np.concatenate([record_data[r][1] for r in val_recs], axis=0)
        train_y = np.concatenate([record_data[r][1] for r in train_recs], axis=0)

        val_counts = [int(np.sum(val_y == c)) for c in range(5)]
        train_counts = [int(np.sum(train_y == c)) for c in range(5)]

        patient_counts_per_fold[fold_idx] = (train_counts, val_counts)

        print(f"\n[Fold {fold_idx}/2 Audit]")
        print(f" -> Validation Patient Records ({len(val_recs)}): {sorted(val_recs)}")
        print(f" -> Training Patient Records ({len(train_recs)}):   {sorted(train_recs)}")
        print(f" -> Val Beat Counts:   N={val_counts[0]:>5}, SVEB={val_counts[1]:>4}, VEB={val_counts[2]:>4}, F/VT={val_counts[3]:>4}, Q={val_counts[4]:>2} | Total={len(val_y):,}")
        print(f" -> Train Beat Counts: N={train_counts[0]:>5}, SVEB={train_counts[1]:>4}, VEB={train_counts[2]:>4}, F/VT={train_counts[3]:>4}, Q={train_counts[4]:>2} | Total={len(train_y):,}")

    # Assertion: Every DS1 patient appears in exactly one validation fold
    assert sorted(all_val_patients) == sorted(DS1_ALL_PATIENTS), "Validation fold union does not match DS1 patient set!"
    assert len(all_val_patients) == len(set(all_val_patients)), "Duplicate patient assigned across validation folds!"

    print("\n[✓] ALL FOLD-BALANCE AND STRUCTURAL AUDIT ASSERTIONS PASSED CLEANLY!")
    print("=================================================================\n")
    return patient_counts_per_fold


def run_sanity_check(device: torch.device):
    """
    Executes a forward/backward pass sanity test.
    """
    print("[*] Running Forward/Backward Sanity Check...")
    model = GenericHybrid1DBiCNNGRU(
        in_channels=1,
        cnn_channels=[64, 128, 128],
        kernel_sizes=[5, 5, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.2076,
        num_classes=5
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count == 257541, f"Model parameter count mismatch! Expected 257,541, got {param_count}"

    criterion = nn.CrossEntropyLoss(ignore_index=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001184, weight_decay=7.114476e-4)

    dummy_input = torch.randn(128, 1, 256, device=device)
    dummy_targets = torch.randint(0, 5, (128,), device=device)

    model.train()
    optimizer.zero_grad()
    outputs = model(dummy_input)
    assert outputs.shape == (128, 5), f"Output shape mismatch! Expected (128, 5), got {outputs.shape}"

    loss = criterion(outputs, dummy_targets)
    assert torch.isfinite(loss), "Loss is not finite!"

    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Gradient for {name} is not finite!"

    optimizer.step()
    print("    [✓] Sanity Check Passed: Finite Loss, Finite Gradients, Correct Output Shape.")
    del model, optimizer
    gc.collect()


def train_eval_fold(
    fold_idx: int,
    hp: Dict,
    record_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    checkpoints_dir: Path,
    trial_number: int
) -> Tuple[float, Dict, int, List[List[int]]]:
    """
    Trains and evaluates 1 fold for up to 10 epochs.
    Returns (best_fold_score, best_metrics, best_epoch, best_confusion_matrix).
    """
    val_records = PILOT_2FOLDS[fold_idx]
    train_records = [r for r in DS1_ALL_PATIENTS if r not in val_records]

    train_1d_list = [record_data[r][0] for r in train_records]
    train_y_list = [record_data[r][1] for r in train_records]
    X_train = np.concatenate(train_1d_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)

    val_1d_list = [record_data[r][0] for r in val_records]
    val_y_list = [record_data[r][1] for r in val_records]
    X_val = np.concatenate(val_1d_list, axis=0)
    y_val = np.concatenate(val_y_list, axis=0)

    counts_train = {c: int(np.sum(y_train == c)) for c in range(5)}
    class_weights = np.zeros(5, dtype=np.float64)
    for c in range(4):
        if counts_train[c] > 0:
            class_weights[c] = (1.0 / counts_train[c]) ** hp["sampling_alpha"]
        else:
            class_weights[c] = 0.0
    class_weights[4] = 0.0  # Q class weight ZERO

    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_ds_raw = ArrhythmiaDatasetRaw(X_train, y_train)
    val_ds_raw = ArrhythmiaDatasetRaw(X_val, y_val)
    augmented_train_ds = AugmentedECGDataset(train_ds_raw, is_train=True)

    train_loader = DataLoader(
        augmented_train_ds,
        batch_size=hp["batch_size"],
        sampler=sampler,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds_raw,
        batch_size=hp["batch_size"],
        shuffle=False,
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

    criterion = nn.CrossEntropyLoss(ignore_index=4)
    optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    ckpt_path = checkpoints_dir / f"trial_{trial_number}_fold_{fold_idx}_best.pth"

    best_score = -999.0
    best_metrics = {}
    best_epoch = 0
    best_conf_mat = None

    for epoch in range(1, 11):  # Max 10 epochs for pilot
        model.train()
        for batch in train_loader:
            _, b1d, targets = batch
            b1d = b1d.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(b1d)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "confusion_matrix": best_conf_mat,
                    "hyperparameters": hp
                },
                ckpt_path
            )

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_score, best_metrics, best_epoch, best_conf_mat


def run_optuna_pilot(max_runtime_seconds: float = 3600.0, audit_only: bool = False):
    device = get_device()
    print_cuda_info()

    print("=================================================================")
    print("   2-FOLD PATIENT-LEVEL STRATIFIED OPTUNA PILOT (ALPHA SEARCH)   ")
    print("=================================================================")
    print(f"[*] Target Acceleration Device: {device}")
    print(f"[*] Max Runtime Limit:          {max_runtime_seconds:.0f} seconds ({max_runtime_seconds/3600.0:.2f} hours)")

    record_data = load_ds1_record_data(str(PROJECT_ROOT / "config.yaml"))
    patient_counts_per_fold = audit_fold_construction(record_data)
    run_sanity_check(device)

    if audit_only:
        print("\n[!] AUDIT ONLY MODE COMPLETE. Stopping before Optuna study execution.")
        return

    checkpoints_dir = PROJECT_ROOT / "checkpoints/optuna_alpha_2fold_pilot"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    start_wall_time = time.time()
    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        # Check wall-clock timeout
        elapsed = time.time() - start_wall_time
        if elapsed >= max_runtime_seconds:
            print(f"\n[!] Global wall-clock limit reached ({elapsed:.1f}s >= {max_runtime_seconds:.0f}s). Stopping study.")
            trial.study.stop()
            raise optuna.TrialPruned("Global wall-clock timeout reached.")

        sampling_alpha = trial.suggest_float("sampling_alpha", 0.0, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256])

        hp = {
            "sampling_alpha": sampling_alpha,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size
        }

        fold_scores = []
        fold_metrics = {}

        for fold_idx in range(1, 3):
            score, metrics, epoch, conf_mat = train_eval_fold(
                fold_idx, hp, record_data, device, checkpoints_dir, trial.number
            )
            fold_scores.append(score)
            fold_metrics[f"fold_{fold_idx}"] = {
                "score": score,
                "metrics": metrics,
                "best_epoch": epoch,
                "confusion_matrix": conf_mat
            }

        mean_objective = float(np.mean(fold_scores))
        mean_acc = float(np.mean([fold_metrics[f"fold_{k}"]["metrics"]["active_accuracy"] for k in range(1, 3)]))
        mean_macro_f1 = float(np.mean([fold_metrics[f"fold_{k}"]["metrics"]["active_macro_f1"] for k in range(1, 3)]))
        mean_minority_f1 = float(np.mean([fold_metrics[f"fold_{k}"]["metrics"]["minority_macro_f1"] for k in range(1, 3)]))
        mean_n_recall = float(np.mean([fold_metrics[f"fold_{k}"]["metrics"]["normal_recall"] for k in range(1, 3)]))

        t_elapsed = time.time() - start_wall_time

        trial_rec = {
            "trial_number": trial.number,
            "hyperparameters": hp,
            "mean_objective": mean_objective,
            "mean_active_accuracy": mean_acc,
            "mean_active_macro_f1": mean_macro_f1,
            "mean_minority_macro_f1": mean_minority_f1,
            "mean_normal_recall": mean_n_recall,
            "elapsed_seconds": t_elapsed,
            "fold_metrics": fold_metrics
        }
        trial_records.append(trial_rec)

        print(f"\n[Trial #{trial.number} Completed in {t_elapsed:.1f}s]")
        print(f" -> Params: alpha={sampling_alpha:.4f}, lr={learning_rate:.6f}, wd={weight_decay:.6e}, bs={batch_size}")
        print(f" -> Mean Objective: {mean_objective:.4f} | Acc: {mean_acc:.2f}% | Macro F1: {mean_macro_f1:.2f}% | Min F1: {mean_minority_f1:.2f}% | N Rec: {mean_n_recall:.2f}%")

        return mean_objective

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

    try:
        study.optimize(objective, n_trials=50, timeout=max_runtime_seconds)
    except Exception as e:
        print(f"[*] Study optimization stopped: {e}")

    total_runtime = time.time() - start_wall_time
    limit_reached = total_runtime >= max_runtime_seconds

    completed_trials = [t for t in trial_records if t["mean_objective"] is not None]

    if len(completed_trials) == 0:
        print("\n[!] No trials were completed before timeout!")
        return

    best_trial_rec = max(completed_trials, key=lambda t: t["mean_objective"])

    print("\n=================================================================")
    print("        OPTUNA 2-FOLD STRATIFIED PILOT STUDY RESULTS REPORT       ")
    print("=================================================================")
    print(f"[*] Total Completed Trials: {len(completed_trials)}")
    print(f"[*] Total Runtime:          {total_runtime:.1f} seconds ({total_runtime/3600.0:.2f} hours)")
    print(f"[*] Runtime Limit Reached:  {limit_reached}")
    print(f"\n[*] BEST TRIAL (# {best_trial_rec['trial_number']}):")
    print(f"    - alpha:                {best_trial_rec['hyperparameters']['sampling_alpha']:.4f}")
    print(f"    - learning_rate:        {best_trial_rec['hyperparameters']['learning_rate']:.6f}")
    print(f"    - weight_decay:         {best_trial_rec['hyperparameters']['weight_decay']:.6e}")
    print(f"    - batch_size:           {best_trial_rec['hyperparameters']['batch_size']}")
    print(f"    - Best Objective Score: {best_trial_rec['mean_objective']:.4f}")
    print(f"    - Mean Active Macro F1: {best_trial_rec['mean_active_macro_f1']:.2f}%")
    print(f"    - Mean Minority Macro F1:{best_trial_rec['mean_minority_macro_f1']:.2f}%")
    print(f"    - Mean Normal Recall:   {best_trial_rec['mean_normal_recall']:.2f}%")
    print(f"    - Mean Active Accuracy: {best_trial_rec['mean_active_accuracy']:.2f}%")
    print("=================================================================\n")

    results_data = {
        "experiment": "Short 2-Fold Patient-Level Stratified Optuna Pilot (Alpha Search)",
        "total_completed_trials": len(completed_trials),
        "total_runtime_seconds": total_runtime,
        "max_runtime_seconds": max_runtime_seconds,
        "limit_reached": limit_reached,
        "best_trial": best_trial_rec,
        "all_trials": trial_records
    }

    results_json = PROJECT_ROOT / "experiments/results_optuna_alpha_2fold_pilot.json"
    with open(results_json, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"[✓] Successfully saved pilot evaluation results JSON to '{results_json}'")
    return results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Short 2-Fold Patient-Level Stratified Optuna Pilot (Alpha Search)")
    parser.add_argument("--max-runtime", type=float, default=3600.0, help="Maximum wall-clock runtime in seconds (default: 3600)")
    parser.add_argument("--audit-only", action="store_true", help="Run fold construction audit and sanity checks without starting Optuna study")
    args = parser.parse_args()

    run_optuna_pilot(max_runtime_seconds=args.max_runtime, audit_only=args.audit_only)
