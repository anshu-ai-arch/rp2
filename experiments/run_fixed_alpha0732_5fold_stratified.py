"""
Fixed-Hyperparameter 5-Fold Patient-Level Stratified Cross-Validation Experiment.

RESEARCH GOAL:
Evaluate whether the minority-aware sampler (alpha = 0.7320) combined with fixed hyperparameters
(lr = 0.001184, wd = 7.114476e-4, batch_size = 128, epochs = 20) achieves consistent minority class
gains across 5 patient-level stratified validation folds.

STRICT RESEARCH CONSTRAINTS:
1. Patient-level 5-fold partition: No patient/record split across train & validation.
2. DS1 patient set (22 records) partitioned into 5 validation folds:
   - Fold 1: ['115', '119', '122', '207', '230'] (5 patients)
   - Fold 2: ['101', '112', '118', '208', '220'] (5 patients)
   - Fold 3: ['114', '124', '205', '223']      (4 patients)
   - Fold 4: ['108', '109', '201', '215']      (4 patients)
   - Fold 5: ['106', '116', '203', '209']      (4 patients)
3. DS2 test set is COMPLETELY LOCKED AND UNTOUCHED.
4. Active classes: N (0), SVEB/A (1), VEB/PVC (2), F/VT (3).
5. Q class (4) is 100% EXCLUDED from training loss (ignore_index=4), sampler (w_4=0.0), and active metrics.
6. GenericHybrid1DBiCNNGRU architecture (257,541 parameters) freshly initialized per fold.
7. Validation sampling remains NATURAL and unweighted.
"""

import os
import sys
import gc
import time
import json
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
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

# Patient-Level Stratified 5-Fold Partition
FIXED_5FOLDS = {
    1: ['115', '119', '122', '207', '230'],
    2: ['101', '112', '118', '208', '220'],
    3: ['114', '124', '205', '223'],
    4: ['108', '109', '201', '215'],
    5: ['106', '116', '203', '209']
}

FIXED_HP = {
    "sampling_alpha": 0.7320,
    "learning_rate": 0.001184,
    "weight_decay": 7.114476e-4,
    "batch_size": 128,
    "epochs": 20
}


def audit_fold_construction(record_data: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """
    Performs an explicit fold-balance audit and verifies all structural research assertions.
    """
    print("\n=================================================================")
    print("        PATIENT-LEVEL STRATIFIED 5-FOLD FOLD-BALANCE AUDIT       ")
    print("=================================================================")

    all_val_patients = []
    patient_counts_per_fold = {}

    for fold_idx in range(1, 6):
        val_recs = FIXED_5FOLDS[fold_idx]
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

        # Beat counts for validation fold
        val_y = np.concatenate([record_data[r][1] for r in val_recs], axis=0)
        train_y = np.concatenate([record_data[r][1] for r in train_recs], axis=0)

        val_counts = [int(np.sum(val_y == c)) for c in range(5)]
        train_counts = [int(np.sum(train_y == c)) for c in range(5)]

        patient_counts_per_fold[fold_idx] = (train_counts, val_counts)

        print(f"\n[Fold {fold_idx}/5 Audit]")
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


def run_fixed_alpha0732_5fold_experiment(audit_only: bool = False):
    device = get_device()
    print("\n=================================================================")
    print("   FIXED-HYPERPARAMETER 5-FOLD STRATIFIED ECG CV EVALUATION     ")
    print("=================================================================")
    print(f"[*] Target Acceleration Device: {device}")
    print(f"[*] Fixed Hyperparameters:")
    print(f"    - sampling_alpha: {FIXED_HP['sampling_alpha']}")
    print(f"    - learning_rate:  {FIXED_HP['learning_rate']}")
    print(f"    - weight_decay:   {FIXED_HP['weight_decay']}")
    print(f"    - batch_size:     {FIXED_HP['batch_size']}")
    print(f"    - epochs:         {FIXED_HP['epochs']}")

    # Verification assertions for parameters
    assert FIXED_HP["sampling_alpha"] == 0.7320, "sampling_alpha mismatch!"
    assert FIXED_HP["learning_rate"] == 0.001184, "learning_rate mismatch!"
    assert FIXED_HP["weight_decay"] == 7.114476e-4, "weight_decay mismatch!"
    assert FIXED_HP["batch_size"] == 128, "batch_size mismatch!"
    assert FIXED_HP["epochs"] == 20, "epochs mismatch!"

    print("\n[*] Loading DS1 patient record data...")
    record_data = load_ds1_record_data(str(PROJECT_ROOT / "config.yaml"))

    # Perform Audit
    patient_counts_per_fold = audit_fold_construction(record_data)

    # Verify model architecture & parameter count
    dummy_model = GenericHybrid1DBiCNNGRU(
        in_channels=1,
        cnn_channels=[64, 128, 128],
        kernel_sizes=[5, 5, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.2076,
        num_classes=5
    )
    param_count = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    assert param_count == 257541, f"Model parameter count mismatch! Expected 257,541, got {param_count}"
    print(f"[✓] GenericHybrid1DBiCNNGRU Parameter Count Verified: {param_count:,}")
    del dummy_model

    if audit_only:
        print("\n[!] AUDIT ONLY MODE COMPLETE. Stopping before full 5-fold training.")
        return

    checkpoints_dir = PROJECT_ROOT / "checkpoints/ecg_fixed_alpha0732_5fold"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    fold_results = {}
    start_time = time.time()
    process = psutil.Process(os.getpid())

    for fold_idx in range(1, 6):
        val_records = FIXED_5FOLDS[fold_idx]
        train_records = [r for r in DS1_ALL_PATIENTS if r not in val_records]

        print(f"\n=================================================================")
        print(f"                  EXECUTING FOLD {fold_idx}/5                    ")
        print(f"=================================================================")
        print(f" -> Validation Patient Records: {sorted(val_records)}")
        print(f" -> Training Patient Records ({len(train_records)} records): {sorted(train_records)}")

        train_1d_list = [record_data[r][0] for r in train_records]
        train_y_list = [record_data[r][1] for r in train_records]
        X_train = np.concatenate(train_1d_list, axis=0)
        y_train = np.concatenate(train_y_list, axis=0)

        val_1d_list = [record_data[r][0] for r in val_records]
        val_y_list = [record_data[r][1] for r in val_records]
        X_val = np.concatenate(val_1d_list, axis=0)
        y_val = np.concatenate(val_y_list, axis=0)

        counts_train = {c: int(np.sum(y_train == c)) for c in range(5)}
        counts_val = {c: int(np.sum(y_val == c)) for c in range(5)}

        class_weights = np.zeros(5, dtype=np.float64)
        for c in range(4):
            if counts_train[c] > 0:
                class_weights[c] = (1.0 / counts_train[c]) ** FIXED_HP["sampling_alpha"]
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
            batch_size=FIXED_HP["batch_size"],
            sampler=sampler,
            num_workers=0
        )
        val_loader = DataLoader(
            val_ds_raw,
            batch_size=FIXED_HP["batch_size"],
            shuffle=False,
            num_workers=0
        )

        # Fresh model initialization per fold
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
        optimizer = optim.Adam(model.parameters(), lr=FIXED_HP["learning_rate"], weight_decay=FIXED_HP["weight_decay"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        ckpt_path = checkpoints_dir / f"ecg_fixed_alpha0732_5fold_fold{fold_idx}_best.pth"

        best_score = -999.0
        best_metrics = {}
        best_epoch = 0
        best_conf_mat = None
        epoch_logs = []

        print(f"\n[*] Starting Training Fold {fold_idx}/5 (20 Epochs)...")
        for epoch in range(1, FIXED_HP["epochs"] + 1):
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

            epoch_train_loss = train_loss / train_total

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

            epoch_log = {
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "metrics": metrics,
                "confusion_matrix": conf_mat.tolist()
            }
            epoch_logs.append(epoch_log)

            print(f"Fold {fold_idx} | Epoch [{epoch:02d}/20] - Loss: {epoch_train_loss:.4f} | Score: {metrics['final_score']:.4f} | Acc: {metrics['active_accuracy']:.2f}% | Macro F1: {metrics['active_macro_f1']:.2f}% | Min F1: {metrics['minority_macro_f1']:.2f}% | N Rec: {metrics['normal_recall']:.2f}%")

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
                        "hyperparameters": FIXED_HP
                    },
                    ckpt_path
                )

        fold_results[f"fold_{fold_idx}"] = {
            "fold_idx": fold_idx,
            "train_records": sorted(train_records),
            "val_records": sorted(val_records),
            "counts_train": counts_train,
            "counts_val": counts_val,
            "class_weights": class_weights.tolist(),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "metrics": best_metrics,
            "confusion_matrix": best_conf_mat,
            "epoch_logs": epoch_logs
        }

        print(f"\n[✓] Fold {fold_idx}/5 Completed! Best Score: {best_score:.4f} at Epoch {best_epoch}")

        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    mem_mb = process.memory_info().rss / (1024.0 * 1024.0)

    accs = [fold_results[f"fold_{k}"]["metrics"]["active_accuracy"] for k in range(1, 6)]
    macro_f1s = [fold_results[f"fold_{k}"]["metrics"]["active_macro_f1"] for k in range(1, 6)]
    minority_f1s = [fold_results[f"fold_{k}"]["metrics"]["minority_macro_f1"] for k in range(1, 6)]
    n_recs = [fold_results[f"fold_{k}"]["metrics"]["normal_recall"] for k in range(1, 6)]
    n_f1s = [fold_results[f"fold_{k}"]["metrics"]["per_class"][0]["f1"] for k in range(1, 6)]
    sveb_f1s = [fold_results[f"fold_{k}"]["metrics"]["per_class"][1]["f1"] for k in range(1, 6)]
    veb_f1s = [fold_results[f"fold_{k}"]["metrics"]["per_class"][2]["f1"] for k in range(1, 6)]
    f_f1s = [fold_results[f"fold_{k}"]["metrics"]["per_class"][3]["f1"] for k in range(1, 6)]
    scores = [fold_results[f"fold_{k}"]["best_score"] for k in range(1, 6)]

    aggregate_summary = {
        "mean_active_accuracy": float(np.mean(accs)), "std_active_accuracy": float(np.std(accs)),
        "mean_active_macro_f1": float(np.mean(macro_f1s)), "std_active_macro_f1": float(np.std(macro_f1s)),
        "mean_minority_macro_f1": float(np.mean(minority_f1s)), "std_minority_macro_f1": float(np.std(minority_f1s)),
        "mean_n_recall": float(np.mean(n_recs)), "std_n_recall": float(np.std(n_recs)),
        "mean_n_f1": float(np.mean(n_f1s)), "std_n_f1": float(np.std(n_f1s)),
        "mean_sveb_f1": float(np.mean(sveb_f1s)), "std_sveb_f1": float(np.std(sveb_f1s)),
        "mean_veb_f1": float(np.mean(veb_f1s)), "std_veb_f1": float(np.std(veb_f1s)),
        "mean_f_f1": float(np.mean(f_f1s)), "std_f_f1": float(np.std(f_f1s)),
        "mean_fold_score": float(np.mean(scores)), "std_fold_score": float(np.std(scores)),
        "elapsed_sec": elapsed_time,
        "mem_mb": mem_mb
    }

    print("\n=================================================================")
    print("        AGGREGATE 5-FOLD PATIENT-LEVEL STRATIFIED RESULTS       ")
    print("=================================================================")
    print(f"[*] Mean Fold Score:       {aggregate_summary['mean_fold_score']:.4f} ± {aggregate_summary['std_fold_score']:.4f}")
    print(f"[*] Mean Active Accuracy:  {aggregate_summary['mean_active_accuracy']:.2f}% ± {aggregate_summary['std_active_accuracy']:.2f}%")
    print(f"[*] Mean Active Macro F1:  {aggregate_summary['mean_active_macro_f1']:.2f}% ± {aggregate_summary['std_active_macro_f1']:.2f}%")
    print(f"[*] Mean Minority Macro F1:{aggregate_summary['mean_minority_macro_f1']:.2f}% ± {aggregate_summary['std_minority_macro_f1']:.2f}%")
    print(f"[*] Mean Normal (N) Recall:{aggregate_summary['mean_n_recall']:.2f}% ± {aggregate_summary['std_n_recall']:.2f}%")
    print(f"[*] Mean Normal (N) F1:    {aggregate_summary['mean_n_f1']:.2f}% ± {aggregate_summary['std_n_f1']:.2f}%")
    print(f"[*] Mean SVEB/A F1:        {aggregate_summary['mean_sveb_f1']:.2f}% ± {aggregate_summary['std_sveb_f1']:.2f}%")
    print(f"[*] Mean VEB/PVC F1:       {aggregate_summary['mean_veb_f1']:.2f}% ± {aggregate_summary['std_veb_f1']:.2f}%")
    print(f"[*] Mean F/VT F1:          {aggregate_summary['mean_f_f1']:.2f}% ± {aggregate_summary['std_f_f1']:.2f}%")
    print("=================================================================\n")

    save_data = {
        "experiment": "Fixed-Hyperparameter 5-Fold Patient-Level Stratified CV (alpha=0.7320, lr=0.001184, wd=7.114476e-4, bs=128)",
        "hyperparameters": FIXED_HP,
        "aggregate_summary": aggregate_summary,
        "fold_results": fold_results
    }

    results_json = PROJECT_ROOT / "experiments/results_ecg_fixed_alpha0732_5fold_stratified.json"
    with open(results_json, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"[✓] Successfully saved 5-fold evaluation results JSON to '{results_json}'")
    return save_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run 5-Fold Patient-Level Stratified ECG Evaluation")
    parser.add_argument("--audit-only", action="store_true", help="Run fold construction audit and assertions without training")
    args = parser.parse_args()

    run_fixed_alpha0732_5fold_experiment(audit_only=args.audit_only)
