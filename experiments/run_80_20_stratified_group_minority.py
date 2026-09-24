"""
80/20 Patient Holdout + Stratified Group 5-Fold + Minority-Aware CE Training ECG Experiment.

RESEARCH GOAL:
Evaluate whether patient-level stratified group cross-validation combined with minority-aware
Cross-Entropy training improves minority-class recognition (SVEB/A and F/Fusion) while
preserving Normal-class performance and overall accuracy, evaluated on a completely unseen
20% patient-level final test pool.

AUTHORITATIVE DETERMINISTIC SPLIT (Seed 25488):
- Total Patients: 47 MIT-BIH records (107,652 total beats)
- Development Pool: 38 Patients (86,868 beats, ~80.7%)
- Final Test Pool: 9 Patients (20,784 beats, ~19.3%)
- Test Class Fractions: N=20.39%, SVEB=20.22%, VEB=20.45%, F/VT=28.30%
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
import wfdb
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Tuple, List, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import get_device
from src.data.preprocessor import ECGPreprocessor
from src.data.beat_extractor import ECGBeatExtractor, AAMI_MAPPING
from experiments.run_model_c_medium_kfold_minority import ArrhythmiaDatasetRaw, compute_active_metrics
from train_augmented_model_c import AugmentedECGDataset
from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU

# Locked Deterministic Split Seed
SPLIT_SEED = 25488

HYPERPARAMETERS = {
    "sampling_alpha": 0.7320,
    "learning_rate": 0.002018,
    "weight_decay": 2.35e-5,
    "batch_size": 256,
    "epochs": 20
}


def load_all_47_patient_records(config_path: Path) -> Tuple[List[str], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Loads raw 1D beat segments and AAMI labels for all 47 MIT-BIH records in data/raw/.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = PROJECT_ROOT / config["data"]["data_dir"]
    if not raw_dir.exists():
        raise FileNotFoundError(f"[!] Dataset directory not found at: '{raw_dir}'")

    all_records = sorted(list(set(f.stem for f in raw_dir.glob("*.atr"))))
    assert len(all_records) == 47, f"Expected 47 records in data/raw/, found {len(all_records)}"

    preprocessor = ECGPreprocessor(
        raw_fs=config["data"]["raw_sampling_rate"],
        target_fs=config["data"]["target_sampling_rate"],
        lowcut=config["data"]["lowcut"],
        highcut=config["data"]["highcut"],
        filter_order=config["data"]["filter_order"],
        segment_duration_sec=config["data"]["segment_duration_sec"],
        matrix_rows=config["data"]["matrix_rows"],
        matrix_cols=config["data"]["matrix_cols"],
        use_filtering=config["data"].get("use_filtering", True),
    )

    beat_extractor = ECGBeatExtractor(
        beat_window_size=config["data"].get("beat_window_size", 256),
        matrix_rows=config["data"].get("matrix_rows", 16),
        matrix_cols=config["data"].get("matrix_cols", 16)
    )

    record_data = {}
    for rec_id in all_records:
        rec_path = raw_dir / rec_id
        record = wfdb.rdrecord(str(rec_path))
        ecg_signal = record.p_signal[:, 0]

        processed_signal = preprocessor.bandpass_filter(ecg_signal)
        resampled_signal = preprocessor.resample_signal(processed_signal)

        b_1d, _, _, b_y = beat_extractor.extract_beats_from_record(
            rec_path, resampled_signal,
            raw_fs=config["data"]["raw_sampling_rate"],
            target_fs=config["data"]["target_sampling_rate"]
        )
        record_data[rec_id] = (b_1d, b_y)

    return all_records, record_data


def get_authoritative_80_20_split(all_records: List[str]) -> Tuple[List[str], List[str]]:
    """
    Returns the single deterministic 38 Development / 9 Final Test split for Seed 25488.
    """
    rng = np.random.RandomState(SPLIT_SEED)
    shuffled = rng.permutation(all_records)
    cand_test = sorted([str(r) for r in shuffled[:9]])
    cand_dev = sorted([str(r) for r in shuffled[9:]])
    return cand_dev, cand_test


def construct_stratified_group_5folds(
    dev_patients: List[str],
    record_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Dict[int, List[str]]:
    """
    Constructs 5 patient-level folds from the 38 development patients,
    stratifying by minority class presence (F/VT, SVEB, VEB).
    """
    patient_stats = []
    for rec in dev_patients:
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


def compute_confusion_matrix_and_flows(all_preds: np.ndarray, all_targets: np.ndarray) -> Dict:
    """
    Computes 4x4 active confusion matrix and detailed misclassification flows.
    """
    active_mask = (all_targets != 4)
    active_p = all_preds[active_mask]
    active_t = all_targets[active_mask]

    conf_mat = np.zeros((4, 4), dtype=int)
    for t, p in zip(active_t, active_p):
        if t < 4 and p < 4:
            conf_mat[t, p] += 1

    class_names = ["N", "SVEB", "VEB", "FVT"]
    flows = {}
    for i, t_name in enumerate(class_names):
        for j, p_name in enumerate(class_names):
            if i != j:
                flows[f"{t_name}_to_{p_name}"] = int(conf_mat[i, j])

    # Summary of false positives: N beats misclassified as any minority class
    n_false_positives = int(conf_mat[0, 1] + conf_mat[0, 2] + conf_mat[0, 3])
    n_total_active = int(np.sum(conf_mat[0, :]))
    n_fp_rate = float(n_false_positives / n_total_active) if n_total_active > 0 else 0.0

    return {
        "confusion_matrix": conf_mat.tolist(),
        "misclassification_flows": flows,
        "normal_false_positives_count": n_false_positives,
        "normal_false_positive_rate": n_fp_rate
    }


def run_phase_a_audit(
    all_records: List[str],
    record_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Tuple[List[str], List[str], Dict[int, List[str]], Dict]:
    """
    Executes Phase 1 - Phase 4 Audit with locked deterministic seed 25488.
    """
    print("\n=================================================================")
    print("      PHASE A: AUTHORITATIVE 80/20 & 5-FOLD GROUP CV AUDIT        ")
    print("=================================================================")

    dev_patients, test_patients = get_authoritative_80_20_split(all_records)
    print(f"[*] Authoritative Deterministic Seed: {SPLIT_SEED}")
    print(f"[*] Development Pool ({len(dev_patients)} Patients): {dev_patients}")
    print(f"[*] Final Test Pool ({len(test_patients)} Patients):   {test_patients}")

    all_patient_stats = []
    total_dataset_counts = np.zeros(5, dtype=int)

    for rec in all_records:
        y = record_data[rec][1]
        pool_tag = "DEV" if rec in dev_patients else "TEST"
        c = [int(np.sum(y == cls_id)) for cls_id in range(5)]
        rec_tot = len(y)
        min_tot = c[1] + c[3]

        total_dataset_counts += np.array(c)
        all_patient_stats.append({
            "pool": pool_tag, "rec": rec,
            "N": c[0], "SVEB": c[1], "VEB": c[2], "FVT": c[3], "Q": c[4],
            "Total": rec_tot, "Minority": min_tot
        })

    dev_c = np.sum([np.array([item["N"], item["SVEB"], item["VEB"], item["FVT"], item["Q"]]) for item in all_patient_stats if item["pool"] == "DEV"], axis=0)
    test_c = np.sum([np.array([item["N"], item["SVEB"], item["VEB"], item["FVT"], item["Q"]]) for item in all_patient_stats if item["pool"] == "TEST"], axis=0)
    dataset_c = dev_c + test_c

    folds = construct_stratified_group_5folds(dev_patients, record_data)

    fold_audit_details = {}
    all_val_patients = []
    pooled_val_counts = np.zeros(5, dtype=int)

    for fold_idx in range(1, 6):
        val_recs = folds[fold_idx]
        train_recs = [r for r in dev_patients if r not in val_recs]
        all_val_patients.extend(val_recs)

        overlap_val_train = set(val_recs).intersection(set(train_recs))
        assert len(overlap_val_train) == 0, f"Patient overlap in Fold {fold_idx}: {overlap_val_train}"

        test_leakage_val = set(val_recs).intersection(set(test_patients))
        test_leakage_train = set(train_recs).intersection(set(test_patients))
        assert len(test_leakage_val) == 0, f"Final Test leakage in Fold {fold_idx} val: {test_leakage_val}"
        assert len(test_leakage_train) == 0, f"Final Test leakage in Fold {fold_idx} train: {test_leakage_train}"

        val_y = np.concatenate([record_data[r][1] for r in val_recs], axis=0)
        train_y = np.concatenate([record_data[r][1] for r in train_recs], axis=0)

        val_counts = np.array([np.sum(val_y == c) for c in range(5)], dtype=int)
        train_counts = np.array([np.sum(train_y == c) for c in range(5)], dtype=int)
        pooled_val_counts += val_counts

        fold_audit_details[f"fold_{fold_idx}"] = {
            "val_patients": sorted(val_recs),
            "train_patients": sorted(train_recs),
            "val_counts": val_counts.tolist(),
            "train_counts": train_counts.tolist(),
            "val_total": int(len(val_y)),
            "train_total": int(len(train_y))
        }

    overlap_dev_test = set(dev_patients).intersection(set(test_patients))
    assert len(overlap_dev_test) == 0, f"Dev/Test pool overlap detected: {overlap_dev_test}"
    assert sorted(dev_patients + test_patients) == sorted(all_records), "Union of Dev + Test does not equal all 47 records!"
    assert sorted(all_val_patients) == sorted(dev_patients), "Fold validation union does not equal Development pool!"
    assert len(all_val_patients) == len(set(all_val_patients)), "Duplicate patient assigned across validation folds!"
    assert np.array_equal(pooled_val_counts, dev_c), "Pooled validation beat counts do not match Development pool totals!"

    print("\n[✓] ALL PHASE 1 - PHASE 4 AUDIT ASSERTIONS AND LEAKAGE CHECKS PASSED CLEANLY!")
    print("=================================================================\n")

    audit_summary = {
        "split_seed": SPLIT_SEED,
        "all_records_count": len(all_records),
        "dev_patients_count": len(dev_patients),
        "test_patients_count": len(test_patients),
        "dev_patients": sorted(dev_patients),
        "test_patients": sorted(test_patients),
        "dataset_totals": dataset_c.tolist(),
        "dev_totals": dev_c.tolist(),
        "test_totals": test_c.tolist(),
        "fold_audit_details": fold_audit_details,
        "leakage_checks": {
            "dev_test_disjointness": True,
            "partition_completeness": True,
            "test_isolation": True,
            "fold_disjointness": True
        }
    }
    return dev_patients, test_patients, folds, audit_summary


def run_phase_b_5fold_cv(
    dev_patients: List[str],
    folds: Dict[int, List[str]],
    record_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    audit_summary: Dict
) -> Tuple[Dict, List[np.ndarray], List[np.ndarray]]:
    """
    Executes Phase B (5-Fold Development CV Training):
    Trains 5-fold CV using GenericHybrid1DBiCNNGRU with WeightedRandomSampler (alpha=0.7320)
    for training folds ONLY and natural evaluation for validation folds.
    """
    device = get_device()
    print("\n=================================================================")
    print("   PHASE B: DEVELOPMENT 5-FOLD MINORITY-AWARE MODEL TRAINING    ")
    print("=================================================================")
    print(f"[*] Target Acceleration Device: {device}")
    print(f"[*] Hyperparameters:")
    print(f"    - sampling_alpha: {HYPERPARAMETERS['sampling_alpha']}")
    print(f"    - learning_rate:  {HYPERPARAMETERS['learning_rate']}")
    print(f"    - weight_decay:   {HYPERPARAMETERS['weight_decay']}")
    print(f"    - batch_size:     {HYPERPARAMETERS['batch_size']}")
    print(f"    - epochs:         {HYPERPARAMETERS['epochs']}")

    checkpoints_dir = PROJECT_ROOT / "checkpoints/80_20_stratified_group_minority"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    fold_results = {}
    val_preds_all_folds = []
    val_targets_all_folds = []

    start_time = time.time()
    process = psutil.Process(os.getpid())

    for fold_idx in range(1, 6):
        val_records = folds[fold_idx]
        train_records = [r for r in dev_patients if r not in val_records]

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

        # Compute Training-Only Class Weights for WeightedRandomSampler
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

        ckpt_path = checkpoints_dir / f"80_20_cv_fold{fold_idx}_best.pth"

        best_score_tuple = (-999.0, -999.0, -999.0, -999.0)
        best_metrics = {}
        best_epoch = 0
        best_val_preds = None
        best_flow_analysis = None
        violates_normal_constraint = False

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
            scheduler.step(metrics["minority_macro_f1"])

            flow_analysis = compute_confusion_matrix_and_flows(all_preds, all_targets)
            n_recall = metrics["normal_recall"]
            min_f1 = metrics["minority_macro_f1"]
            macro_f1 = metrics["active_macro_f1"]
            acc = metrics["active_accuracy"]

            # Constrained Selection Objective:
            # Primary: Minority Macro F1 subject to Normal Recall >= 96.5% if feasible
            meets_n_constraint = (n_recall >= 96.5)
            cand_score_tuple = (1.0 if meets_n_constraint else 0.0, min_f1, macro_f1, n_recall)

            print(f"Fold {fold_idx} | Epoch [{epoch:02d}/20] - Loss: {epoch_train_loss:.4f} | Min F1: {min_f1:.2f}% | Macro F1: {macro_f1:.2f}% | Acc: {acc:.2f}% | N Rec: {n_recall:.2f}% {'[N-RECALL SAFE]' if meets_n_constraint else '[N-RECALL VIOLATION]'}")

            if cand_score_tuple > best_score_tuple:
                best_score_tuple = cand_score_tuple
                best_metrics = metrics
                best_epoch = epoch
                best_val_preds = all_preds
                best_flow_analysis = flow_analysis
                violates_normal_constraint = not meets_n_constraint

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "confusion_analysis": flow_analysis,
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
            "class_weights": class_weights.tolist(),
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "confusion_analysis": best_flow_analysis,
            "violates_normal_recall_constraint": violates_normal_constraint
        }

        print(f"\n[✓] Fold {fold_idx}/5 Completed! Best Epoch {best_epoch} | Min F1: {best_metrics['minority_macro_f1']:.2f}% | N Rec: {best_metrics['normal_recall']:.2f}%")

        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    mem_mb = process.memory_info().rss / (1024.0 * 1024.0)

    # Compute Pooled Validation Metrics across all 5 Folds
    pooled_preds = np.concatenate(val_preds_all_folds, axis=0)
    pooled_targets = np.concatenate(val_targets_all_folds, axis=0)
    assert len(pooled_preds) == 86868, f"Pooled beat count mismatch! Expected 86,868, got {len(pooled_preds)}"

    pooled_metrics = compute_active_metrics(pooled_preds, pooled_targets)
    pooled_flow_analysis = compute_confusion_matrix_and_flows(pooled_preds, pooled_targets)

    # Compute Aggregate Stats across 5 Folds
    accs = [fold_results[f"fold_{k}"]["best_metrics"]["active_accuracy"] for k in range(1, 6)]
    macro_f1s = [fold_results[f"fold_{k}"]["best_metrics"]["active_macro_f1"] for k in range(1, 6)]
    minority_f1s = [fold_results[f"fold_{k}"]["best_metrics"]["minority_macro_f1"] for k in range(1, 6)]
    n_recs = [fold_results[f"fold_{k}"]["best_metrics"]["normal_recall"] for k in range(1, 6)]
    sveb_f1s = [fold_results[f"fold_{k}"]["best_metrics"]["per_class"][1]["f1"] for k in range(1, 6)]
    veb_f1s = [fold_results[f"fold_{k}"]["best_metrics"]["per_class"][2]["f1"] for k in range(1, 6)]
    f_f1s = [fold_results[f"fold_{k}"]["best_metrics"]["per_class"][3]["f1"] for k in range(1, 6)]

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
    print(f"[*] Pooled DS1 Validation Active Accuracy: {pooled_metrics['active_accuracy']:.2f}%")
    print(f"[*] Pooled DS1 Validation Active Macro F1: {pooled_metrics['active_macro_f1']:.2f}%")
    print(f"[*] Pooled DS1 Validation Minority Macro F1:{pooled_metrics['minority_macro_f1']:.2f}%")
    print(f"[*] Pooled DS1 Validation Normal Recall:   {pooled_metrics['normal_recall']:.2f}%")
    print("=================================================================\n")

    cv_results = {
        "aggregate_summary": aggregate_summary,
        "pooled_metrics": pooled_metrics,
        "pooled_confusion_analysis": pooled_flow_analysis,
        "fold_results": fold_results
    }
    return cv_results, val_preds_all_folds, val_targets_all_folds


def run_experiment(audit_only: bool = False):
    config_path = PROJECT_ROOT / "config.yaml"
    print("\n[*] Loading all 47 MIT-BIH patient records...")
    all_records, record_data = load_all_47_patient_records(config_path)

    # Phase A Audit
    dev_patients, test_patients, folds, audit_summary = run_phase_a_audit(all_records, record_data)

    if audit_only:
        print("[!] AUDIT ONLY MODE COMPLETE. Stopping before Phase B model training.")
        return

    # Phase B 5-Fold Training
    cv_results, _, _ = run_phase_b_5fold_cv(dev_patients, folds, record_data, audit_summary)

    # Get Git Hash
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT)).decode("utf-8").strip()
    except Exception:
        git_hash = "N/A"

    save_data = {
        "experiment": "80_20 Patient Holdout + Stratified Group 5-Fold + Minority-Aware CE Training (alpha=0.7320, lr=0.002018, wd=2.35e-5, bs=256)",
        "git_commit": git_hash,
        "hyperparameters": HYPERPARAMETERS,
        "audit_summary": audit_summary,
        "cv_results": cv_results
    }

    results_json = PROJECT_ROOT / "experiments/results_80_20_stratified_group_minority.json"
    with open(results_json, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"[✓] Successfully saved Phase B experiment results JSON to '{results_json}'")

    artifacts_dir = PROJECT_ROOT / "artifacts/80_20_stratified_group_minority"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / "phase_b_summary.json", "w") as f:
        json.dump(save_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 80/20 Patient Holdout + Stratified Group 5-Fold ECG Experiment")
    parser.add_argument("--audit-only", action="store_true", help="Execute Phase A Audit only and exit before model training")
    args = parser.parse_args()

    run_experiment(audit_only=args.audit_only)
