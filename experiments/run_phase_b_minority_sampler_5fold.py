"""
Phase B: Minority-Aware WeightedRandomSampler 5-Fold Training for ECG Classification.

RESEARCH GOAL:
Evaluate whether training-only minority-aware sampling (alpha = 0.7320) with Cross-Entropy Loss
improves recognition of minority classes SVEB/A and F/VT while maintaining high Normal-class
recall (>= 96.5%) and overall accuracy across 5 patient-level stratified validation folds.

STRICT RESEARCH CONSTRAINTS:
1. Authoritative 80/20 Patient Split (Seed 25488):
   - Development Pool: 38 Patients
   - Final Test Pool: 9 Patients (100% UNTOUCHED, NEVER LOADED DURING PHASE B)
2. Locked 5-Fold Development CV Assignments:
   - Fold 1: ['105', '118', '207', '210', '215', '222', '230', '233']
   - Fold 2: ['101', '108', '112', '114', '117', '124', '208', '231']
   - Fold 3: ['102', '109', '116', '121', '201', '203', '214', '232']
   - Fold 4: ['103', '113', '115', '122', '200', '205', '212']
   - Fold 5: ['100', '104', '106', '107', '111', '219', '223']
3. GenericHybrid1DBiCNNGRU Architecture (257,541 parameters) initialized freshly per fold.
4. Preprocessing & Augmentation: 256-sample window [0,1] scaled; online augmentation ONLY on training fold.
5. Sampler: WeightedRandomSampler (alpha = 0.7320) ONLY on training fold. Validation is NATURAL.
6. Model Selection Objective: Prioritize Normal Recall >= 96.5%, then maximize Minority Macro F1.
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

# Authoritative Locked Development Patient Pool (38 Records)
DEVELOPMENT_PATIENTS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '109', '111', '112',
    '113', '114', '115', '116', '117', '118', '121', '122', '124', '200', '201',
    '202', '203', '205', '207', '208', '210', '212', '214', '215', '219', '222',
    '223', '230', '231', '232', '233'
]

# Authoritative Locked Final Test Patient Pool (9 Records - NEVER UNLOCKED IN PHASE B)
FINAL_TEST_PATIENTS = [
    '108', '119', '123', '209', '213', '220', '221', '228', '234'
]

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

HYPERPARAMETERS = {
    "sampling_alpha": 0.7320,
    "learning_rate": 0.002018,
    "weight_decay": 2.35e-5,
    "batch_size": 256,
    "epochs": 20
}


def make_json_serializable(obj):
    """
    Recursively converts NumPy scalars, arrays, and standard container types
    into standard Python JSON-serializable types (bool, int, float, list, dict).
    """
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [make_json_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    return obj



def resolve_data_directory(custom_data_dir: str = None) -> Path:
    """
    Resolves ECG dataset directory across local environments and Kaggle.
    Supports CLI arg, ECG_DATA_DIR env variable, Kaggle input path, or config.yaml default.
    """
    if custom_data_dir and Path(custom_data_dir).exists():
        return Path(custom_data_dir)

    env_dir = os.getenv("ECG_DATA_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    # Standard Kaggle dataset locations
    kaggle_paths = [
        Path("/kaggle/input/datasets/klmsathishkumar/mit-bih-arrhythmia-database/mitdb"),
        Path("/kaggle/input/datasets/klmsathishkumar/mit-bih-arrhythmia-database"),
        Path("/kaggle/input/mit-bih-arrhythmia-database/mitdb"),
        Path("/kaggle/input/mit-bih-arrhythmia-database")
    ]
    for kp in kaggle_paths:
        if kp.exists():
            return kp

    # Local config default
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        raw_dir = Path(cfg["data"]["data_dir"])
        data_dir = raw_dir if raw_dir.is_absolute() else (PROJECT_ROOT / raw_dir)
        if data_dir.exists():
            return data_dir

    raise FileNotFoundError("[!] Could not locate MIT-BIH dataset directory. Set ECG_DATA_DIR or pass --data-dir.")


def load_development_patient_records(data_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Loads raw 1D beat segments and AAMI labels ONLY for the 38 development patients.
    """
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "data": {
                "raw_sampling_rate": 360, "target_sampling_rate": 128,
                "lowcut": 0.5, "highcut": 50.0, "filter_order": 4,
                "segment_duration_sec": 10, "matrix_rows": 16, "matrix_cols": 16,
                "beat_window_size": 256, "use_filtering": True
            }
        }

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
    for rec_id in DEVELOPMENT_PATIENTS:
        rec_path = data_dir / rec_id
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

    return record_data


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

    n_false_positives = int(conf_mat[0, 1] + conf_mat[0, 2] + conf_mat[0, 3])
    n_total_active = int(np.sum(conf_mat[0, :]))
    n_fp_rate = float(n_false_positives / n_total_active) if n_total_active > 0 else 0.0

    return {
        "confusion_matrix": conf_mat.tolist(),
        "misclassification_flows": flows,
        "normal_false_positives_count": n_false_positives,
        "normal_false_positive_rate": n_fp_rate
    }


def validate_structural_integrity(data_dir: Path, record_data: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """
    Performs Phase B structural verification, parameter check, and leakage assertions without running training.
    """
    print("\n=================================================================")
    print("      PHASE B STRUCTURAL & CONFIGURATION VALIDATION AUDIT        ")
    print("=================================================================")
    print(f"[*] Dataset Directory Verified: {data_dir}")
    print(f"[*] Development Patients ({len(DEVELOPMENT_PATIENTS)}): {DEVELOPMENT_PATIENTS}")
    print(f"[*] Locked Final Test Patients ({len(FINAL_TEST_PATIENTS)} - UNTOUCHED): {FINAL_TEST_PATIENTS}")

    # Check 1: Patient Disjointness
    overlap_dev_test = set(DEVELOPMENT_PATIENTS).intersection(set(FINAL_TEST_PATIENTS))
    assert len(overlap_dev_test) == 0, f"Dev/Test pool overlap detected: {overlap_dev_test}"
    print("[✓] Dev and Test patient pools are 100% disjoint.")

    # Check 2: Fold Partition Verification
    folds = construct_stratified_group_5folds(DEVELOPMENT_PATIENTS, record_data)
    all_val_patients = []
    for fold_idx in range(1, 6):
        val_recs = folds[fold_idx]
        train_recs = [r for r in DEVELOPMENT_PATIENTS if r not in val_recs]
        all_val_patients.extend(val_recs)

        # Disjointness
        overlap_val_train = set(val_recs).intersection(set(train_recs))
        assert len(overlap_val_train) == 0, f"Patient overlap in Fold {fold_idx}: {overlap_val_train}"

        # Test isolation
        test_leakage_val = set(val_recs).intersection(set(FINAL_TEST_PATIENTS))
        test_leakage_train = set(train_recs).intersection(set(FINAL_TEST_PATIENTS))
        assert len(test_leakage_val) == 0, f"Test leakage in Fold {fold_idx} val: {test_leakage_val}"
        assert len(test_leakage_train) == 0, f"Test leakage in Fold {fold_idx} train: {test_leakage_train}"

        # Count beats
        val_y = np.concatenate([record_data[r][1] for r in val_recs], axis=0)
        train_y = np.concatenate([record_data[r][1] for r in train_recs], axis=0)

        counts_train = {c: int(np.sum(train_y == c)) for c in range(5)}
        counts_val = {c: int(np.sum(val_y == c)) for c in range(5)}

        # Verify Training Sampler Weight calculation (Training Only)
        class_weights = np.zeros(5, dtype=np.float64)
        for c in range(4):
            if counts_train[c] > 0:
                class_weights[c] = (1.0 / counts_train[c]) ** HYPERPARAMETERS["sampling_alpha"]
            else:
                class_weights[c] = 0.0
        class_weights[4] = 0.0  # Q class weight ZERO

        print(f" -> Fold {fold_idx}: Val ({len(val_recs)} recs, {len(val_y):,} beats) | Train ({len(train_recs)} recs, {len(train_y):,} beats)")
        print(f"    Train Sampler Weights (alpha={HYPERPARAMETERS['sampling_alpha']}): N={class_weights[0]:.6e}, SVEB={class_weights[1]:.6e}, VEB={class_weights[2]:.6e}, F/VT={class_weights[3]:.6e}, Q={class_weights[4]:.1f}")

    assert sorted(all_val_patients) == sorted(DEVELOPMENT_PATIENTS), "Validation fold union does not match Development pool!"
    assert len(all_val_patients) == len(set(all_val_patients)), "Duplicate patient assigned across validation folds!"
    print("[✓] 5-Fold patient assignments form a complete, disjoint partition of Development pool.")

    # Check 3: Model Architecture & Parameter Count
    model = GenericHybrid1DBiCNNGRU(
        in_channels=1,
        cnn_channels=[64, 128, 128],
        kernel_sizes=[5, 5, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.2076,
        num_classes=5
    )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count == 257541, f"Model parameter count mismatch! Expected 257,541, got {param_count}"
    print(f"[✓] GenericHybrid1DBiCNNGRU Trainable Parameter Count Verified: {param_count:,}")

    # Check 4: Verify criterion ignore_index=4
    criterion = nn.CrossEntropyLoss(ignore_index=4)
    assert criterion.ignore_index == 4, "CrossEntropyLoss ignore_index must be 4 for Q class!"
    print("[✓] CrossEntropyLoss ignore_index=4 verified (Q class excluded from active loss).")

    print("\n[✓] ALL PHASE B STRUCTURAL & CONFIGURATION CHECKS PASSED CLEANLY!")
    print("=================================================================\n")


def run_phase_b_training(
    record_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: Path
) -> Dict:
    """
    Executes full Phase B 5-Fold Training on Development Pool using WeightedRandomSampler (alpha=0.7320)
    on training folds ONLY and natural evaluation on validation folds.
    """
    device = get_device()
    print("\n=================================================================")
    print("   PHASE B: DEVELOPMENT 5-FOLD MINORITY-AWARE MODEL TRAINING    ")
    print("=================================================================")
    print(f"[*] Target Acceleration Device: {device}")
    print(f"[*] Output Directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results = {}
    val_preds_all_folds = []
    val_targets_all_folds = []

    start_time = time.time()
    process = psutil.Process(os.getpid())

    folds = construct_stratified_group_5folds(DEVELOPMENT_PATIENTS, record_data)
    for fold_idx in range(1, 6):
        fold_dir = output_dir / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_records = folds[fold_idx]
        train_records = [r for r in DEVELOPMENT_PATIENTS if r not in val_records]

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

        # Training-Only Class Weights for WeightedRandomSampler
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

        ckpt_path = fold_dir / "best_model.pth"
        if ckpt_path.exists():
            print(f"\n[*] Existing checkpoint found for Fold {fold_idx} at '{ckpt_path}'. Loading saved model state...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model = GenericHybrid1DBiCNNGRU(
                in_channels=1,
                cnn_channels=[64, 128, 128],
                kernel_sizes=[5, 5, 3],
                gru_hidden_size=64,
                gru_num_layers=2,
                dropout=0.2076,
                num_classes=5
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            val_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    _, b1d, _ = batch
                    b1d = b1d.to(device)
                    outputs = model(b1d)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    val_preds.append(preds)
            best_val_preds = np.concatenate(val_preds, axis=0)
            best_epoch = checkpoint.get("epoch", 0)
            best_metrics = checkpoint.get("metrics", compute_active_metrics(best_val_preds, y_val))
            best_flow_analysis = checkpoint.get("confusion_analysis", compute_confusion_matrix_and_flows(best_val_preds, y_val))

            with open(fold_dir / "metrics.json", "w") as f:
                json.dump(make_json_serializable({"best_epoch": best_epoch, "metrics": best_metrics}), f, indent=2)
            with open(fold_dir / "confusion_matrix.json", "w") as f:
                json.dump(make_json_serializable(best_flow_analysis), f, indent=2)

            val_preds_all_folds.append(best_val_preds)
            val_targets_all_folds.append(y_val)
            fold_results[f"fold_{fold_idx}"] = {
                "fold_idx": fold_idx,
                "train_records": sorted(train_records),
                "val_records": sorted(val_records),
                "counts_train": counts_train,
                "counts_val": counts_val,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "confusion_analysis": best_flow_analysis
            }
            print(f"[✓] Fold {fold_idx}/5 Recovered from Checkpoint! Best Epoch {best_epoch} | Min F1: {best_metrics['minority_macro_f1']:.2f}% | N Rec: {best_metrics['normal_recall']:.2f}%")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        best_score_tuple = (-999.0, -999.0, -999.0, -999.0)
        best_metrics = {}
        best_epoch = 0
        best_val_preds = None
        best_flow_analysis = None
        epoch_history = []

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

            meets_n_constraint = bool(n_recall >= 96.5)
            cand_score_tuple = (1.0 if meets_n_constraint else 0.0, min_f1, macro_f1, n_recall)

            epoch_log = {
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "metrics": metrics,
                "normal_recall_constraint_met": meets_n_constraint,
                "flow_analysis": flow_analysis
            }
            epoch_history.append(epoch_log)

            print(f"Fold {fold_idx} | Epoch [{epoch:02d}/20] - Loss: {epoch_train_loss:.4f} | Min F1: {min_f1:.2f}% | Macro F1: {macro_f1:.2f}% | Acc: {acc:.2f}% | N Rec: {n_recall:.2f}% {'[N-SAFE]' if meets_n_constraint else '[N-VIOLATION]'}")

            if cand_score_tuple > best_score_tuple:
                best_score_tuple = cand_score_tuple
                best_metrics = metrics
                best_epoch = epoch
                best_val_preds = all_preds
                best_flow_analysis = flow_analysis

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "confusion_analysis": flow_analysis,
                    "hyperparameters": HYPERPARAMETERS
                }, ckpt_path)

        val_preds_all_folds.append(best_val_preds)
        val_targets_all_folds.append(y_val)

        # Save Fold JSON Files
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(make_json_serializable({"best_epoch": best_epoch, "metrics": best_metrics}), f, indent=2)
        with open(fold_dir / "confusion_matrix.json", "w") as f:
            json.dump(make_json_serializable(best_flow_analysis), f, indent=2)
        with open(fold_dir / "history.json", "w") as f:
            json.dump(make_json_serializable(epoch_history), f, indent=2)

        fold_results[f"fold_{fold_idx}"] = {
            "fold_idx": fold_idx,
            "train_records": sorted(train_records),
            "val_records": sorted(val_records),
            "counts_train": counts_train,
            "counts_val": counts_val,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "confusion_analysis": best_flow_analysis
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
    pooled_metrics = compute_active_metrics(pooled_preds, pooled_targets)
    pooled_flow_analysis = compute_confusion_matrix_and_flows(pooled_preds, pooled_targets)

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

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT)).decode("utf-8").strip()
    except Exception:
        git_hash = "N/A"

    save_data = {
        "experiment": "Phase B Minority-Aware WeightedRandomSampler 5-Fold Training (alpha=0.7320, lr=0.002018, wd=2.35e-5, bs=256)",
        "git_commit": git_hash,
        "hyperparameters": HYPERPARAMETERS,
        "aggregate_summary": aggregate_summary,
        "pooled_metrics": pooled_metrics,
        "pooled_confusion_analysis": pooled_flow_analysis,
        "fold_results": fold_results
    }

    with open(output_dir / "pooled_results.json", "w") as f:
        json.dump(make_json_serializable(save_data), f, indent=2)

    print(f"[✓] Phase B 5-Fold Training Complete! Saved pooled results to '{output_dir / 'pooled_results.json'}'")
    return save_data


def main():
    parser = argparse.ArgumentParser(description="Phase B Minority-Aware ECG 5-Fold Training Script")
    parser.add_argument("--dry-run", action="store_true", help="Run lightweight structural validation and audit checks without model training")
    parser.add_argument("--audit-only", action="store_true", help="Alias for --dry-run")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to MIT-BIH dataset directory")
    parser.add_argument("--output-dir", type=str, default="experiments/results_phase_b", help="Output directory for Phase B checkpoints and JSON results")
    args = parser.parse_args()

    data_dir = resolve_data_directory(args.data_dir)
    print(f"[*] Resolved ECG Dataset Directory: '{data_dir}'")

    record_data = load_development_patient_records(data_dir)
    validate_structural_integrity(data_dir, record_data)

    if args.dry_run or args.audit_only:
        print("[!] DRY-RUN / AUDIT-ONLY MODE COMPLETE. Exiting before model training.")
        return

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    run_phase_b_training(record_data, output_dir)


if __name__ == "__main__":
    main()
