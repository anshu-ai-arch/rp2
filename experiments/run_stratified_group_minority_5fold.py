"""
Stratified Group K-Fold + Minority-Aware Training ECG Experiment.

RESEARCH GOAL:
Improve recognition of minority classes SVEB/A (1) and F/VT (3) while preserving Normal (0)
performance, using patient-level Stratified Group K-Fold cross-validation (5 folds) and
training-only minority-aware sampling (alpha = 0.7320).

STRICT RESEARCH CONSTRAINTS:
1. Patient-level Grouping: No patient/record split across train & validation.
2. DS1 patient set (22 records) partitioned into 5 validation folds.
3. DS2 test set (22 records) is COMPLETELY LOCKED AND UNTOUCHED.
4. Active classes: N (0), SVEB/A (1), VEB/PVC (2), F/VT (3).
5. Q class (4) is 100% EXCLUDED from training loss (ignore_index=4), sampler (w_4=0.0), and active metrics.
6. GenericHybrid1DBiCNNGRU architecture (257,541 parameters) freshly initialized per fold.
7. Validation sampling remains NATURAL and unweighted.
8. Two-Phase Execution: Phase A (Audit-Only) must pass before Phase B (Full 5-Fold Training).
"""

import os
import sys
import gc
import time
import json
import argparse
import subprocess
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

# Validated Patient-Level Stratified 5-Fold Partition
STRATIFIED_GROUP_5FOLDS = {
    1: ['115', '119', '122', '207', '230'],
    2: ['101', '112', '118', '208', '220'],
    3: ['114', '124', '205', '223'],
    4: ['108', '109', '201', '215'],
    5: ['106', '116', '203', '209']
}

HYPERPARAMETERS = {
    "sampling_alpha": 0.7320,
    "learning_rate": 0.002018,
    "weight_decay": 2.35e-5,
    "batch_size": 256,
    "epochs": 20
}

EXPECTED_DS1_TOTALS = {
    "N": 45813,
    "SVEB": 943,
    "VEB": 3786,
    "FVT": 898,
    "Q": 8,
    "Total": 51448
}


def run_phase_a_audit(record_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """
    Executes Phase A Audit:
    1. Patient-wise beat label table for all 22 DS1 records.
    2. 5-fold patient assignment table.
    3. Per-fold N / SVEB / VEB / F / Q counts.
    4. Pooled validation totals compared against expected benchmark.
    5. Data leakage & patient isolation assertions.
    """
    print("\n=================================================================")
    print("      PHASE A: STRATIFIED GROUP 5-FOLD CV DATASET AUDIT          ")
    print("=================================================================")

    # 1. Patient-wise Table
    patient_table = []
    print("\n[1] Patient-Wise Beat Count Table (DS1 - 22 Patients):")
    print(f"{'Patient':<8} | {'N':>6} | {'SVEB/A':>6} | {'VEB/PVC':>7} | {'F/VT':>6} | {'Q':>3} | {'Total':>6} | {'SVEB+F/VT':>9}")
    print("-" * 75)

    tot_n, tot_sveb, tot_veb, tot_fvt, tot_q, tot_all = 0, 0, 0, 0, 0, 0

    for rec in sorted(DS1_ALL_PATIENTS):
        y = record_data[rec][1]
        n_c = int(np.sum(y == 0))
        sveb_c = int(np.sum(y == 1))
        veb_c = int(np.sum(y == 2))
        fvt_c = int(np.sum(y == 3))
        q_c = int(np.sum(y == 4))
        rec_tot = len(y)
        min_tot = sveb_c + fvt_c

        tot_n += n_c
        tot_sveb += sveb_c
        tot_veb += veb_c
        tot_fvt += fvt_c
        tot_q += q_c
        tot_all += rec_tot

        patient_table.append({
            "patient": rec,
            "N": n_c,
            "SVEB": sveb_c,
            "VEB": veb_c,
            "FVT": fvt_c,
            "Q": q_c,
            "Total": rec_tot,
            "SVEB_plus_FVT": min_tot
        })
        print(f"{rec:<8} | {n_c:>6} | {sveb_c:>6} | {veb_c:>7} | {fvt_c:>6} | {q_c:>3} | {rec_tot:>6} | {min_tot:>9}")

    print("-" * 75)
    print(f"{'DS1 TOTAL':<8} | {tot_n:>6} | {tot_sveb:>6} | {tot_veb:>7} | {tot_fvt:>6} | {tot_q:>3} | {tot_all:>6} | {tot_sveb+tot_fvt:>9}")
    print("=" * 75)

    # 2. 5-Fold Patient Assignment & Beat Counts
    fold_audit_details = {}
    all_val_patients = []
    pooled_counts = [0, 0, 0, 0, 0]

    print("\n[2] Per-Fold Validation Patient Assignment & Beat Counts:")
    for fold_idx in range(1, 6):
        val_recs = STRATIFIED_GROUP_5FOLDS[fold_idx]
        train_recs = [r for r in DS1_ALL_PATIENTS if r not in val_recs]
        all_val_patients.extend(val_recs)

        # Leakage assertion 1: Patient disjointness
        overlap = set(val_recs).intersection(set(train_recs))
        assert len(overlap) == 0, f"Patient overlap detected in Fold {fold_idx}: {overlap}"

        # Leakage assertion 2: DS2 isolation
        ds2_overlap_val = set(val_recs).intersection(set(DS2_TEST_RECORDINGS))
        ds2_overlap_train = set(train_recs).intersection(set(DS2_TEST_RECORDINGS))
        assert len(ds2_overlap_val) == 0, f"DS2 leakage in Fold {fold_idx} val: {ds2_overlap_val}"
        assert len(ds2_overlap_train) == 0, f"DS2 leakage in Fold {fold_idx} train: {ds2_overlap_train}"

        # Compute validation fold beat counts
        val_y = np.concatenate([record_data[r][1] for r in val_recs], axis=0)
        train_y = np.concatenate([record_data[r][1] for r in train_recs], axis=0)

        val_c = [int(np.sum(val_y == c)) for c in range(5)]
        train_c = [int(np.sum(train_y == c)) for c in range(5)]

        for c in range(5):
            pooled_counts[c] += val_c[c]

        fold_audit_details[f"fold_{fold_idx}"] = {
            "val_patients": sorted(val_recs),
            "train_patients": sorted(train_recs),
            "val_counts": val_c,
            "train_counts": train_c,
            "val_total": len(val_y),
            "train_total": len(train_y)
        }

        print(f"\n[Fold {fold_idx}/5]")
        print(f" -> Validation Patients ({len(val_recs)}): {sorted(val_recs)}")
        print(f" -> Val Beat Counts:   N={val_c[0]:>5}, SVEB={val_c[1]:>4}, VEB={val_c[2]:>4}, F/VT={val_c[3]:>4}, Q={val_c[4]:>2} | Total={len(val_y):,}")
        print(f" -> Train Beat Counts: N={train_c[0]:>5}, SVEB={train_c[1]:>4}, VEB={train_c[2]:>4}, F/VT={train_c[3]:>4}, Q={train_c[4]:>2} | Total={len(train_y):,}")

    # 3. Partition Completeness Assertions
    assert sorted(all_val_patients) == sorted(DS1_ALL_PATIENTS), "Validation union mismatch with DS1 patient set!"
    assert len(all_val_patients) == len(set(all_val_patients)), "Duplicate patient assigned across validation folds!"
    assert sum(pooled_counts) == 51448, f"Pooled beat total mismatch! Expected 51,448, got {sum(pooled_counts)}"

    # 4. Compare against Expected DS1 Benchmark
    print("\n=================================================================")
    print("            POOLED VALIDATION TOTALS VS EXPECTED BENCHMARK        ")
    print("=================================================================")
    print(f"Class        | Expected  | Actual    | Status")
    print("-" * 50)
    print(f"Normal (N)   | {EXPECTED_DS1_TOTALS['N']:>9,} | {pooled_counts[0]:>9,} | {'MATCH' if pooled_counts[0] == EXPECTED_DS1_TOTALS['N'] else 'MISMATCH'}")
    print(f"SVEB/A       | {EXPECTED_DS1_TOTALS['SVEB']:>9,} | {pooled_counts[1]:>9,} | {'MATCH' if pooled_counts[1] == EXPECTED_DS1_TOTALS['SVEB'] else 'MISMATCH'}")
    print(f"VEB/PVC      | {EXPECTED_DS1_TOTALS['VEB']:>9,} | {pooled_counts[2]:>9,} | {'MATCH' if pooled_counts[2] == EXPECTED_DS1_TOTALS['VEB'] else 'MISMATCH'}")
    print(f"F/VT         | {EXPECTED_DS1_TOTALS['FVT']:>9,} | {pooled_counts[3]:>9,} | {'MATCH' if pooled_counts[3] == EXPECTED_DS1_TOTALS['FVT'] else 'MISMATCH'}")
    print(f"Q Class      | {EXPECTED_DS1_TOTALS['Q']:>9,} | {pooled_counts[4]:>9,} | {'MATCH' if pooled_counts[4] == EXPECTED_DS1_TOTALS['Q'] else 'MISMATCH'}")
    print("-" * 50)
    print(f"TOTAL        | {EXPECTED_DS1_TOTALS['Total']:>9,} | {sum(pooled_counts):>9,} | {'MATCH' if sum(pooled_counts) == EXPECTED_DS1_TOTALS['Total'] else 'MISMATCH'}")
    print("=================================================================")

    # Stop & Report Discrepancy if any check fails
    mismatches = []
    if pooled_counts[0] != EXPECTED_DS1_TOTALS['N']: mismatches.append("N")
    if pooled_counts[1] != EXPECTED_DS1_TOTALS['SVEB']: mismatches.append("SVEB")
    if pooled_counts[2] != EXPECTED_DS1_TOTALS['VEB']: mismatches.append("VEB")
    if pooled_counts[3] != EXPECTED_DS1_TOTALS['FVT']: mismatches.append("FVT")
    if pooled_counts[4] != EXPECTED_DS1_TOTALS['Q']: mismatches.append("Q")

    if mismatches:
        print(f"\n[!] CRITICAL DISCREPANCY DETECTED in classes: {mismatches}")
        print("[!] Stopping execution as required by research protocol.")
        raise ValueError(f"Dataset total mismatch detected for classes {mismatches}")

    print("\n[✓] ALL PHASE A AUDIT CHECKS AND LEAKAGE ASSERTIONS PASSED CLEANLY!")
    print("=================================================================\n")

    audit_summary = {
        "ds1_patients_count": len(DS1_ALL_PATIENTS),
        "ds2_patients_count": len(DS2_TEST_RECORDINGS),
        "patient_table": patient_table,
        "fold_audit_details": fold_audit_details,
        "pooled_counts": {
            "N": pooled_counts[0],
            "SVEB": pooled_counts[1],
            "VEB": pooled_counts[2],
            "FVT": pooled_counts[3],
            "Q": pooled_counts[4],
            "Total": sum(pooled_counts)
        },
        "leakage_checks": {
            "patient_disjointness": True,
            "ds2_isolation": True,
            "partition_completeness": True,
            "zero_overlap": True
        }
    }
    return audit_summary


def run_phase_b_training(record_data: Dict[str, Tuple[np.ndarray, np.ndarray]], audit_summary: Dict) -> Dict:
    """
    Executes Phase B Training Protocol:
    Trains 5-fold CV using GenericHybrid1DBiCNNGRU with WeightedRandomSampler (alpha=0.7320)
    for training folds ONLY, and natural evaluation for validation folds.
    """
    device = get_device()
    print("\n=================================================================")
    print("      PHASE B: STRATIFIED GROUP 5-FOLD CV MODEL TRAINING         ")
    print("=================================================================")
    print(f"[*] Target Acceleration Device: {device}")
    print(f"[*] Hyperparameters:")
    print(f"    - sampling_alpha: {HYPERPARAMETERS['sampling_alpha']}")
    print(f"    - learning_rate:  {HYPERPARAMETERS['learning_rate']}")
    print(f"    - weight_decay:   {HYPERPARAMETERS['weight_decay']}")
    print(f"    - batch_size:     {HYPERPARAMETERS['batch_size']}")
    print(f"    - epochs:         {HYPERPARAMETERS['epochs']}")

    checkpoints_dir = PROJECT_ROOT / "checkpoints/stratified_group_minority_5fold"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    fold_results = {}
    val_preds_all_folds = []
    val_targets_all_folds = []

    start_time = time.time()
    process = psutil.Process(os.getpid())

    for fold_idx in range(1, 6):
        val_records = STRATIFIED_GROUP_5FOLDS[fold_idx]
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

        # Compute Class Weights for WeightedRandomSampler (Training Only)
        class_weights = np.zeros(5, dtype=np.float64)
        for c in range(4):
            if counts_train[c] > 0:
                class_weights[c] = (1.0 / counts_train[c]) ** HYPERPARAMETERS["sampling_alpha"]
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
            batch_size=HYPERPARAMETERS["batch_size"],
            sampler=sampler,
            num_workers=0
        )
        val_loader = DataLoader(
            val_ds_raw,
            batch_size=HYPERPARAMETERS["batch_size"],
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
        optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMETERS["learning_rate"], weight_decay=HYPERPARAMETERS["weight_decay"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        ckpt_path = checkpoints_dir / f"stratified_group_fold{fold_idx}_best.pth"

        best_score = -999.0
        best_metrics = {}
        best_epoch = 0
        best_val_preds = None

        print(f"\n[*] Training Fold {fold_idx}/5 ({HYPERPARAMETERS['epochs']} Epochs)...")
        for epoch in range(1, HYPERPARAMETERS["epochs"] + 1):
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

            print(f"Fold {fold_idx} | Epoch [{epoch:02d}/20] - Loss: {epoch_train_loss:.4f} | Score: {metrics['final_score']:.4f} | Acc: {metrics['active_accuracy']:.2f}% | Macro F1: {metrics['active_macro_f1']:.2f}% | Min F1: {metrics['minority_macro_f1']:.2f}% | N Rec: {metrics['normal_recall']:.2f}%")

            if metrics["final_score"] > best_score:
                best_score = metrics["final_score"]
                best_metrics = metrics
                best_epoch = epoch
                best_val_preds = all_preds
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "hyperparameters": HYPERPARAMETERS
                }, ckpt_path)

        val_preds_all_folds.append(best_val_preds)
        val_targets_all_folds.append(y_val)

        fold_results[f"fold_{fold_idx}"] = {
            "fold_idx": fold_idx,
            "train_records": sorted(train_records),
            "val_records": sorted(val_records),
            "counts_train": counts_train,
            "counts_val": counts_val,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "metrics": best_metrics
        }

        print(f"\n[✓] Fold {fold_idx}/5 Completed! Best Score: {best_score:.4f} at Epoch {best_epoch}")

        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    mem_mb = process.memory_info().rss / (1024.0 * 1024.0)

    # Compute Pooled Validation Metrics across all 5 Folds
    pooled_preds = np.concatenate(val_preds_all_folds, axis=0)
    pooled_targets = np.concatenate(val_targets_all_folds, axis=0)
    assert len(pooled_preds) == 51448, f"Pooled beat count mismatch! Expected 51,448, got {len(pooled_preds)}"
    pooled_metrics = compute_active_metrics(pooled_preds, pooled_targets)

    # Compute Aggregate Stats across 5 Folds
    accs = [fold_results[f"fold_{k}"]["metrics"]["active_accuracy"] for k in range(1, 6)]
    macro_f1s = [fold_results[f"fold_{k}"]["metrics"]["active_macro_f1"] for k in range(1, 6)]
    minority_f1s = [fold_results[f"fold_{k}"]["metrics"]["minority_macro_f1"] for k in range(1, 6)]
    n_recs = [fold_results[f"fold_{k}"]["metrics"]["normal_recall"] for k in range(1, 6)]
    sveb_f1s = [fold_results[f"fold_{k}"]["metrics"]["per_class"][1]["f1"] for k in range(1, 6)]
    veb_f1s = [fold_results[f"fold_{k}"]["metrics"]["per_class"][2]["f1"] for k in range(1, 6)]
    f_f1s = [fold_results[f"fold_{k}"]["metrics"]["per_class"][3]["f1"] for k in range(1, 6)]

    aggregate_summary = {
        "mean_active_accuracy": float(np.mean(accs)), "std_active_accuracy": float(np.std(accs)),
        "mean_active_macro_f1": float(np.mean(macro_f1s)), "std_active_macro_f1": float(np.std(macro_f1s)),
        "mean_minority_macro_f1": float(np.mean(minority_f1s)), "std_minority_macro_f1": float(np.std(minority_f1s)),
        "mean_n_recall": float(np.mean(n_recs)), "std_n_recall": float(np.std(n_recs)),
        "mean_sveb_f1": float(np.mean(sveb_f1s)), "std_sveb_f1": float(np.std(sveb_f1s)),
        "mean_veb_f1": float(np.mean(veb_f1s)), "std_veb_f1": float(np.std(veb_f1s)),
        "mean_f_f1": float(np.mean(f_f1s)), "std_f_f1": float(np.std(f_f1s)),
        "elapsed_sec": elapsed_time,
        "mem_mb": mem_mb
    }

    print("\n=================================================================")
    print("     AGGREGATE STRATIFIED GROUP 5-FOLD MODEL EVALUATION          ")
    print("=================================================================")
    print(f"[*] Mean Active Accuracy:  {aggregate_summary['mean_active_accuracy']:.2f}% ± {aggregate_summary['std_active_accuracy']:.2f}%")
    print(f"[*] Mean Active Macro F1:  {aggregate_summary['mean_active_macro_f1']:.2f}% ± {aggregate_summary['std_active_macro_f1']:.2f}%")
    print(f"[*] Mean Minority Macro F1:{aggregate_summary['mean_minority_macro_f1']:.2f}% ± {aggregate_summary['std_minority_macro_f1']:.2f}%")
    print(f"[*] Mean Normal (N) Recall:{aggregate_summary['mean_n_recall']:.2f}% ± {aggregate_summary['std_n_recall']:.2f}%")
    print(f"[*] Mean SVEB/A F1:        {aggregate_summary['mean_sveb_f1']:.2f}% ± {aggregate_summary['std_sveb_f1']:.2f}%")
    print(f"[*] Mean VEB/PVC F1:       {aggregate_summary['mean_veb_f1']:.2f}% ± {aggregate_summary['std_veb_f1']:.2f}%")
    print(f"[*] Mean F/VT F1:          {aggregate_summary['mean_f_f1']:.2f}% ± {aggregate_summary['std_f_f1']:.2f}%")
    print("-----------------------------------------------------------------")
    print(f"[*] Pooled DS1 Validation Score:       {pooled_metrics['final_score']:.4f}")
    print(f"[*] Pooled Active Macro F1:             {pooled_metrics['active_macro_f1']:.2f}%")
    print(f"[*] Pooled Minority Macro F1:           {pooled_metrics['minority_macro_f1']:.2f}%")
    print("=================================================================\n")

    # Get git hash
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT)).decode("utf-8").strip()
    except Exception:
        git_hash = "N/A"

    save_data = {
        "experiment": "Stratified Group K-Fold + Minority-Aware Training (alpha=0.7320, lr=0.002018, wd=2.35e-5, bs=256)",
        "git_commit": git_hash,
        "hyperparameters": HYPERPARAMETERS,
        "audit_summary": audit_summary,
        "aggregate_summary": aggregate_summary,
        "pooled_metrics": pooled_metrics,
        "fold_results": fold_results
    }

    results_json = PROJECT_ROOT / "experiments/results_stratified_group_minority_5fold.json"
    with open(results_json, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"[✓] Successfully saved experiment results JSON to '{results_json}'")

    # Save artifact summary
    artifacts_dir = PROJECT_ROOT / "artifacts/stratified_group_minority_cv"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_artifact = artifacts_dir / "audit_and_summary.json"
    with open(summary_artifact, "w") as f:
        json.dump({"audit_summary": audit_summary, "aggregate_summary": aggregate_summary}, f, indent=2)

    return save_data


def run_experiment(audit_only: bool = False):
    print("\nLoading DS1 patient record data...")
    record_data = load_ds1_record_data(str(PROJECT_ROOT / "config.yaml"))

    # Phase A: Audit
    audit_summary = run_phase_a_audit(record_data)

    if audit_only:
        print("[!] AUDIT ONLY MODE COMPLETE. Stopping before Phase B model training.")
        return

    # Phase B: Training
    run_phase_b_training(record_data, audit_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stratified Group 5-Fold Minority-Aware ECG Experiment")
    parser.add_argument("--audit-only", action="store_true", help="Execute Phase A Audit only and exit before model training")
    args = parser.parse_args()

    run_experiment(audit_only=args.audit_only)
