"""
Minority-Aware Patient-Level 5-Fold Stratified Cross-Validation Experiment.

RESEARCH GOAL:
Reduce minority-class (SVEB/A and F/VT) validation variance across patient folds by constructing
a minority-aware patient-level stratification partition. Evaluates stability across repeated random
seeds [0, 1, 2, 3, 4] with pooled DS1 beat-level evaluation per seed.

STRICT RESEARCH CONSTRAINTS:
1. Patient-level 5-fold partition: No patient/record split across train & validation.
2. DS1 patient set (22 records) partitioned into 5 validation folds using minority-aware sorting.
3. DS2 test set (22 records) is COMPLETELY LOCKED AND UNTOUCHED.
4. Active classes: N (0), SVEB/A (1), VEB/PVC (2), F/VT (3).
5. Q class (4) is 100% EXCLUDED from training loss (ignore_index=4), sampler (w_4=0.0), and active metrics.
6. GenericHybrid1DBiCNNGRU architecture (257,541 parameters) freshly initialized per fold.
7. Validation sampling remains NATURAL and unweighted.
8. Pooled DS1 evaluation per seed: concatenated validation predictions across all 5 folds.
"""

import os
import sys
import gc
import time
import json
import argparse
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

HYPERPARAMETERS = {
    "sampling_alpha": 0.7320,
    "learning_rate": 0.002018,
    "weight_decay": 2.35e-5,
    "batch_size": 256,
    "epochs": 20
}

SEEDS = [0, 1, 2, 3, 4]


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def construct_minority_aware_5folds(record_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, List[str]]:
    """
    Constructs 5 patient-level folds by sorting DS1 patients based on minority class counts
    (F/VT, SVEB/A, VEB/PVC) and distributing them round-robin across folds.
    """
    patient_stats = []
    for rec in DS1_ALL_PATIENTS:
        y = record_data[rec][1]
        n_sveb = int(np.sum(y == 1))
        n_veb = int(np.sum(y == 2))
        n_fvt = int(np.sum(y == 3))
        n_total = len(y)
        patient_stats.append({
            "rec": rec,
            "fvt": n_fvt,
            "sveb": n_sveb,
            "veb": n_veb,
            "total": n_total
        })

    # Sort descending by F/VT, then SVEB, then VEB, then total beats
    sorted_patients = sorted(
        patient_stats,
        key=lambda p: (p["fvt"], p["sveb"], p["veb"], p["total"]),
        reverse=True
    )

    folds = {1: [], 2: [], 3: [], 4: [], 5: []}
    for idx, p in enumerate(sorted_patients):
        fold_id = (idx % 5) + 1
        folds[fold_id].append(p["rec"])

    return folds


def audit_minority_aware_folds(record_data: Dict[str, Tuple[np.ndarray, np.ndarray]], folds: Dict[int, List[str]]):
    """
    Verifies patient disjointness, total beat count preservation, DS2 isolation,
    and displays minority beat distribution per validation fold.
    """
    print("\n=================================================================")
    print("      MINORITY-AWARE PATIENT-LEVEL 5-FOLD FOLD-BALANCE AUDIT     ")
    print("=================================================================")

    all_val_patients = []
    total_val_beats = 0

    for fold_idx in range(1, 6):
        val_recs = folds[fold_idx]
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
        total_val_beats += len(val_y)

        val_counts = [int(np.sum(val_y == c)) for c in range(5)]
        train_counts = [int(np.sum(train_y == c)) for c in range(5)]

        print(f"\n[Fold {fold_idx}/5 Audit]")
        print(f" -> Validation Patient Records ({len(val_recs)}): {sorted(val_recs)}")
        print(f" -> Val Beat Counts:   N={val_counts[0]:>5}, SVEB={val_counts[1]:>4}, VEB={val_counts[2]:>4}, F/VT={val_counts[3]:>4}, Q={val_counts[4]:>2} | Total={len(val_y):,}")
        print(f" -> Train Beat Counts: N={train_counts[0]:>5}, SVEB={train_counts[1]:>4}, VEB={train_counts[2]:>4}, F/VT={train_counts[3]:>4}, Q={train_counts[4]:>2} | Total={len(train_y):,}")

    # Assertions
    assert sorted(all_val_patients) == sorted(DS1_ALL_PATIENTS), "Validation fold union does not match DS1 patient set!"
    assert len(all_val_patients) == len(set(all_val_patients)), "Duplicate patient assigned across validation folds!"
    assert total_val_beats == 51448, f"Total pooled validation beat count mismatch! Expected 51,448, got {total_val_beats}"

    print(f"\n[✓] Total Pooled Validation Beats across 5 Folds: {total_val_beats:,}")
    print("[✓] ALL FOLD-BALANCE AND STRUCTURAL AUDIT ASSERTIONS PASSED CLEANLY!")
    print("=================================================================\n")


def run_minority_aware_experiment(audit_only: bool = False, num_seeds: int = 5):
    device = get_device()
    print("\n=================================================================")
    print("   MINORITY-AWARE PATIENT 5-FOLD CV WITH SEED REPETITIONS       ")
    print("=================================================================")
    print(f"[*] Target Acceleration Device: {device}")
    print(f"[*] Hyperparameters:")
    print(f"    - sampling_alpha: {HYPERPARAMETERS['sampling_alpha']}")
    print(f"    - learning_rate:  {HYPERPARAMETERS['learning_rate']}")
    print(f"    - weight_decay:   {HYPERPARAMETERS['weight_decay']}")
    print(f"    - batch_size:     {HYPERPARAMETERS['batch_size']}")
    print(f"    - epochs:         {HYPERPARAMETERS['epochs']}")
    print(f"[*] Seed Repetitions ({num_seeds}): {SEEDS[:num_seeds]}")

    print("\n[*] Loading DS1 patient record data...")
    record_data = load_ds1_record_data(str(PROJECT_ROOT / "config.yaml"))

    # Construct minority-aware folds
    folds = construct_minority_aware_5folds(record_data)

    # Audit folds
    audit_minority_aware_folds(record_data, folds)

    # Verify model parameter count
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
        print("\n[!] AUDIT ONLY MODE COMPLETE. Stopping before seed runs.")
        return

    checkpoints_dir = PROJECT_ROOT / "checkpoints/minority_aware_patient_cv"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    active_seeds = SEEDS[:num_seeds]
    seed_pooled_results = {}
    all_seed_metrics = []

    start_time = time.time()
    process = psutil.Process(os.getpid())

    for seed in active_seeds:
        print(f"\n=================================================================")
        print(f"                     STARTING SEED RUN {seed}                   ")
        print(f"=================================================================")
        set_seed(seed)

        seed_val_preds_list = []
        seed_val_targets_list = []
        seed_fold_details = {}

        for fold_idx in range(1, 6):
            val_records = folds[fold_idx]
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
            counts_val = {c: int(np.sum(y_val == c)) for c in range(5)}

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

            best_score = -999.0
            best_metrics = {}
            best_epoch = 0
            best_val_preds = None

            ckpt_path = checkpoints_dir / f"minority_aware_seed{seed}_fold{fold_idx}_best.pth"

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

                if metrics["final_score"] > best_score:
                    best_score = metrics["final_score"]
                    best_metrics = metrics
                    best_epoch = epoch
                    best_val_preds = all_preds
                    torch.save({
                        "epoch": epoch,
                        "seed": seed,
                        "fold_idx": fold_idx,
                        "model_state_dict": model.state_dict(),
                        "metrics": metrics,
                        "hyperparameters": HYPERPARAMETERS
                    }, ckpt_path)

            print(f"Seed {seed} | Fold {fold_idx}/5 - Best Score: {best_score:.4f} at Epoch {best_epoch} | Acc: {best_metrics['active_accuracy']:.2f}% | Macro F1: {best_metrics['active_macro_f1']:.2f}% | Min F1: {best_metrics['minority_macro_f1']:.2f}%")

            seed_val_preds_list.append(best_val_preds)
            seed_val_targets_list.append(y_val)

            seed_fold_details[f"fold_{fold_idx}"] = {
                "val_records": sorted(val_records),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "metrics": best_metrics
            }

            del model, optimizer, scheduler, train_loader, val_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Compute Pooled Metrics for Seed
        pooled_preds = np.concatenate(seed_val_preds_list, axis=0)
        pooled_targets = np.concatenate(seed_val_targets_list, axis=0)
        assert len(pooled_preds) == 51448, f"Pooled beat count mismatch! Expected 51448, got {len(pooled_preds)}"

        pooled_seed_metrics = compute_active_metrics(pooled_preds, pooled_targets)
        print(f"\n[✓] Seed {seed} Pooled Evaluation Complete (51,448 beats):")
        print(f"    -> Pooled Score:            {pooled_seed_metrics['final_score']:.4f}")
        print(f"    -> Pooled Active Accuracy: {pooled_seed_metrics['active_accuracy']:.2f}%")
        print(f"    -> Pooled Active Macro F1: {pooled_seed_metrics['active_macro_f1']:.2f}%")
        print(f"    -> Pooled Minority Macro F1:{pooled_seed_metrics['minority_macro_f1']:.2f}%")

        seed_pooled_results[f"seed_{seed}"] = {
            "seed": seed,
            "pooled_metrics": pooled_seed_metrics,
            "fold_details": seed_fold_details
        }
        all_seed_metrics.append(pooled_seed_metrics)

    elapsed_time = time.time() - start_time
    mem_mb = process.memory_info().rss / (1024.0 * 1024.0)

    # Compute aggregate stats across seeds
    accs = [m["active_accuracy"] for m in all_seed_metrics]
    macro_f1s = [m["active_macro_f1"] for m in all_seed_metrics]
    minority_f1s = [m["minority_macro_f1"] for m in all_seed_metrics]
    n_recs = [m["normal_recall"] for m in all_seed_metrics]
    n_f1s = [m["per_class"][0]["f1"] for m in all_seed_metrics]

    sveb_precs = [m["per_class"][1]["precision"] for m in all_seed_metrics]
    sveb_recs = [m["per_class"][1]["recall"] for m in all_seed_metrics]
    sveb_f1s = [m["per_class"][1]["f1"] for m in all_seed_metrics]

    veb_precs = [m["per_class"][2]["precision"] for m in all_seed_metrics]
    veb_recs = [m["per_class"][2]["recall"] for m in all_seed_metrics]
    veb_f1s = [m["per_class"][2]["f1"] for m in all_seed_metrics]

    f_precs = [m["per_class"][3]["precision"] for m in all_seed_metrics]
    f_recs = [m["per_class"][3]["recall"] for m in all_seed_metrics]
    f_f1s = [m["per_class"][3]["f1"] for m in all_seed_metrics]

    aggregate_summary = {
        "num_seeds": len(active_seeds),
        "mean_active_accuracy": float(np.mean(accs)), "std_active_accuracy": float(np.std(accs)),
        "mean_active_macro_f1": float(np.mean(macro_f1s)), "std_active_macro_f1": float(np.std(macro_f1s)),
        "mean_minority_macro_f1": float(np.mean(minority_f1s)), "std_minority_macro_f1": float(np.std(minority_f1s)),
        "mean_n_recall": float(np.mean(n_recs)), "std_n_recall": float(np.std(n_recs)),
        "mean_n_f1": float(np.mean(n_f1s)), "std_n_f1": float(np.std(n_f1s)),
        "mean_sveb_precision": float(np.mean(sveb_precs)), "std_sveb_precision": float(np.std(sveb_precs)),
        "mean_sveb_recall": float(np.mean(sveb_recs)), "std_sveb_recall": float(np.std(sveb_recs)),
        "mean_sveb_f1": float(np.mean(sveb_f1s)), "std_sveb_f1": float(np.std(sveb_f1s)),
        "mean_veb_precision": float(np.mean(veb_precs)), "std_veb_precision": float(np.std(veb_precs)),
        "mean_veb_recall": float(np.mean(veb_recs)), "std_veb_recall": float(np.std(veb_recs)),
        "mean_veb_f1": float(np.mean(veb_f1s)), "std_veb_f1": float(np.std(veb_f1s)),
        "mean_f_precision": float(np.mean(f_precs)), "std_f_precision": float(np.std(f_precs)),
        "mean_f_recall": float(np.mean(f_recs)), "std_f_recall": float(np.std(f_recs)),
        "mean_f_f1": float(np.mean(f_f1s)), "std_f_f1": float(np.std(f_f1s)),
        "elapsed_sec": elapsed_time,
        "mem_mb": mem_mb
    }

    print("\n=================================================================")
    print("      AGGREGATE MINORITY-AWARE PATIENT CV POOLED RESULTS        ")
    print("=================================================================")
    print(f"[*] Seeds Evaluated:        {active_seeds}")
    print(f"[*] Mean Active Accuracy:  {aggregate_summary['mean_active_accuracy']:.2f}% ± {aggregate_summary['std_active_accuracy']:.2f}%")
    print(f"[*] Mean Active Macro F1:  {aggregate_summary['mean_active_macro_f1']:.2f}% ± {aggregate_summary['std_active_macro_f1']:.2f}%")
    print(f"[*] Mean Minority Macro F1:{aggregate_summary['mean_minority_macro_f1']:.2f}% ± {aggregate_summary['std_minority_macro_f1']:.2f}%")
    print(f"[*] Mean Normal (N) Recall:{aggregate_summary['mean_n_recall']:.2f}% ± {aggregate_summary['std_n_recall']:.2f}%")
    print(f"[*] Mean SVEB/A F1:        {aggregate_summary['mean_sveb_f1']:.2f}% ± {aggregate_summary['std_sveb_f1']:.2f}%")
    print(f"[*] Mean VEB/PVC F1:       {aggregate_summary['mean_veb_f1']:.2f}% ± {aggregate_summary['std_veb_f1']:.2f}%")
    print(f"[*] Mean F/VT F1:          {aggregate_summary['mean_f_f1']:.2f}% ± {aggregate_summary['std_f_f1']:.2f}%")
    print("=================================================================\n")

    save_data = {
        "experiment": "Minority-Aware Patient 5-Fold Stratified CV with Repeated Seeds (alpha=0.7320, lr=0.002018, wd=2.35e-5, bs=256)",
        "hyperparameters": HYPERPARAMETERS,
        "folds_assignment": {f"fold_{k}": folds[k] for k in range(1, 6)},
        "aggregate_summary": aggregate_summary,
        "seed_pooled_results": seed_pooled_results
    }

    results_json = PROJECT_ROOT / "experiments/results_minority_aware_patient_cv.json"
    with open(results_json, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"[✓] Successfully saved minority-aware CV results JSON to '{results_json}'")
    return save_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Minority-Aware Patient 5-Fold ECG Evaluation")
    parser.add_argument("--audit-only", action="store_true", help="Run fold construction audit and assertions without training")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds to run (1 to 5)")
    args = parser.parse_args()

    run_minority_aware_experiment(audit_only=args.audit_only, num_seeds=args.num_seeds)
