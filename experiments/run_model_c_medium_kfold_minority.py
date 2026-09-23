import os
import sys
import gc
import time
import json
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, List, Dict
import wfdb

sys.path.append(os.getcwd())

from src.data.preprocessor import ECGPreprocessor
from src.data.beat_extractor import ECGBeatExtractor
from train_augmented_model_c import AugmentedECGDataset
from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU
from experiments.evaluate_patient_wise import get_m1_device

# Exact patient-disjoint fold assignment for DS1 (22 records)
FOLD_PATIENTS = {
    1: ['207', '118', '106', '112'],
    2: ['208', '209', '114', '115'],
    3: ['223', '201', '109', '122'],
    4: ['205', '203', '119', '124', '101'],
    5: ['215', '220', '108', '116', '230']
}

DS1_ALL_PATIENTS = [rec for fold_recs in FOLD_PATIENTS.values() for rec in fold_recs]


class ArrhythmiaDatasetRaw(Dataset):
    """
    PyTorch Dataset wrapping raw ECG beat segments, matrices, RR features, and labels.
    """
    def __init__(self, segments_1d: np.ndarray, labels: np.ndarray):
        self.segments_1d = torch.tensor(segments_1d, dtype=torch.float32).unsqueeze(1)
        self.matrices_2d = torch.zeros((len(labels), 1, 16, 16), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.matrices_2d[idx], self.segments_1d[idx], self.labels[idx]


def load_ds1_record_data(config_path: str = "config.yaml") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Loads raw 1D beat segments and labels grouped by record_id (patient) for DS1.
    Supports portable path resolution relative to PROJECT_ROOT or via ECG_DATA_DIR environment variable.
    """
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path

    if not cfg_path.exists():
        raise FileNotFoundError(f"[!] Configuration file not found at '{cfg_path}'")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # Allow environment variable override for Google Drive / custom dataset path
    env_data_dir = os.getenv("ECG_DATA_DIR")
    if env_data_dir:
        data_dir = Path(env_data_dir)
    else:
        raw_dir = Path(config["data"]["data_dir"])
        data_dir = raw_dir if raw_dir.is_absolute() else (project_root / raw_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"\n[!] Dataset directory not found at: '{data_dir}'\n"
            f"    Please ensure the MIT-BIH dataset is present at that location, or set the "
            f"environment variable 'ECG_DATA_DIR' (e.g. export ECG_DATA_DIR='/content/drive/MyDrive/mitdb')."
        )

    use_filtering = config["data"].get("use_filtering", True)

    preprocessor = ECGPreprocessor(
        raw_fs=config["data"]["raw_sampling_rate"],
        target_fs=config["data"]["target_sampling_rate"],
        lowcut=config["data"]["lowcut"],
        highcut=config["data"]["highcut"],
        filter_order=config["data"]["filter_order"],
        segment_duration_sec=config["data"]["segment_duration_sec"],
        matrix_rows=config["data"]["matrix_rows"],
        matrix_cols=config["data"]["matrix_cols"],
        use_filtering=use_filtering,
    )

    beat_extractor = ECGBeatExtractor(
        beat_window_size=config["data"].get("beat_window_size", 256),
        matrix_rows=config["data"].get("matrix_rows", 16),
        matrix_cols=config["data"].get("matrix_cols", 16)
    )

    record_data = {}
    for rec_id in DS1_ALL_PATIENTS:
        record_path = data_dir / rec_id
        record = wfdb.rdrecord(str(record_path))
        ecg_signal = record.p_signal[:, 0]

        if use_filtering:
            processed_signal = preprocessor.bandpass_filter(ecg_signal)
        else:
            processed_signal = ecg_signal

        resampled_signal = preprocessor.resample_signal(processed_signal)

        b_1d, _, _, b_y = beat_extractor.extract_beats_from_record(
            record_path, resampled_signal,
            raw_fs=config["data"]["raw_sampling_rate"],
            target_fs=config["data"]["target_sampling_rate"]
        )

        record_data[rec_id] = (b_1d, b_y)

    return record_data


def get_fold_dataloaders(
    record_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    val_fold_idx: int,
    batch_size: int = 256,
    sampling_alpha: float = 0.5
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[str, int]]:
    """
    Constructs patient-disjoint Train and Validation DataLoaders for fold val_fold_idx.
    """
    val_records = FOLD_PATIENTS[val_fold_idx]
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
    class_counts_train = {}
    for c in range(5):
        class_counts_train[c] = int(np.sum(y_train == c))

    # Calculate sampling weights w_c = (1 / N_c)^alpha for c in 0..3, w_4 = 0.0
    class_weights = np.zeros(5, dtype=np.float64)
    for c in range(4):
        if class_counts_train[c] > 0:
            class_weights[c] = (1.0 / class_counts_train[c]) ** sampling_alpha
        else:
            class_weights[c] = 0.0
    class_weights[4] = 0.0  # Q class weight explicitly ZERO

    # Assign sample weights
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    # Wrap in PyTorch Datasets
    train_ds_raw = ArrhythmiaDatasetRaw(X_train, y_train)
    val_ds_raw = ArrhythmiaDatasetRaw(X_val, y_val)

    # Wrap train dataset in Online Augmentation
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

    class_counts_val = {c: int(np.sum(y_val == c)) for c in range(5)}

    return train_loader, val_loader, class_counts_train, class_counts_val


def compute_active_metrics(all_preds: np.ndarray, all_targets: np.ndarray) -> Dict[str, float]:
    """
    Computes active-class evaluation metrics excluding Q (Class 4).
    """
    # Mask out any Q target beats if present
    active_mask = (all_targets != 4)
    preds = all_preds[active_mask]
    targets = all_targets[active_mask]

    total_active = len(targets)
    correct_active = np.sum(preds == targets)
    active_accuracy = (correct_active / total_active) * 100.0 if total_active > 0 else 0.0

    # Per-class precision, recall, f1 for classes 0, 1, 2, 3
    recalls, precisions, f1s = [], [], []
    per_class_metrics = {}

    for c in range(4):
        c_target_mask = (targets == c)
        c_pred_mask = (preds == c)

        tp = np.sum(c_target_mask & c_pred_mask)
        fn = np.sum(c_target_mask & (~c_pred_mask))
        fp = np.sum((~c_target_mask) & c_pred_mask)

        rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1)

        per_class_metrics[c] = {"precision": prec, "recall": rec, "f1": f1}

    active_macro_f1 = float(np.mean(f1s))
    minority_macro_f1 = float(np.mean(f1s[1:4]))
    normal_recall = recalls[0]

    # Calculate objective score with 96.50% Normal Recall floor penalty
    base_score = 0.50 * (active_macro_f1 / 100.0) + 0.30 * (minority_macro_f1 / 100.0) + 0.20 * (normal_recall / 100.0)
    recall_penalty = 5.0 * max(0.0, 0.9650 - (normal_recall / 100.0))
    final_score = base_score - recall_penalty

    return {
        "active_accuracy": active_accuracy,
        "active_macro_f1": active_macro_f1,
        "minority_macro_f1": minority_macro_f1,
        "normal_recall": normal_recall,
        "sveb_recall": recalls[1],
        "veb_recall": recalls[2],
        "f_recall": recalls[3],
        "base_score": base_score,
        "recall_penalty": recall_penalty,
        "final_score": final_score,
        "per_class": per_class_metrics
    }


def train_single_fold(
    record_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    fold_idx: int,
    lr: float = 0.002018,
    weight_decay: float = 2.35e-5,
    batch_size: int = 256,
    sampling_alpha: float = 0.5,
    num_epochs: int = 20,
    verbose: bool = True
) -> Dict:
    """
    Trains GenericHybrid1DBiCNNGRU on a single patient-disjoint fold.
    """
    device = get_m1_device()
    train_loader, val_loader, train_counts, val_counts = get_fold_dataloaders(
        record_data, val_fold_idx=fold_idx, batch_size=batch_size, sampling_alpha=sampling_alpha
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

    # Q class (4) ignored in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=4)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_score = -999.0
    best_metrics = {}

    start_time = time.time()
    process = psutil.Process(os.getpid())

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_total = 0.0, 0

        for batch in train_loader:
            if len(batch) == 4:
                _, b1d, _, targets = batch
            elif len(batch) == 2:
                b1d, targets = batch
            else:
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

        # Validation evaluation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    _, b1d, _, targets = batch
                elif len(batch) == 2:
                    b1d, targets = batch
                else:
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

        if verbose:
            print(f"Fold {fold_idx} | Epoch [{epoch:02d}/{num_epochs:02d}] - Train Loss: {train_loss/train_total:.4f} | Val Score: {metrics['final_score']:.4f} | Val Acc: {metrics['active_accuracy']:.2f}% | Macro F1: {metrics['active_macro_f1']:.2f}% | N Rec: {metrics['normal_recall']:.2f}%")

    elapsed_sec = time.time() - start_time
    mem_mb = process.memory_info().rss / (1024.0 * 1024.0)

    # Cleanup memory
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold_idx": fold_idx,
        "elapsed_sec": elapsed_sec,
        "mem_mb": mem_mb,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "best_metrics": best_metrics
    }


def run_benchmark_fold_1():
    print("\n=================================================================")
    print("      SINGLE BENCHMARK FOLD EXECUTION (Fold 1 of Trial 0)         ")
    print("=================================================================")
    print("[*] Loading DS1 records into memory...")
    record_data = load_ds1_record_data("config.yaml")

    print("[*] Running Fold 1 Benchmark (20 Epochs)...")
    res = train_single_fold(
        record_data=record_data,
        fold_idx=1,
        lr=0.002018,
        weight_decay=2.35e-5,
        batch_size=256,
        sampling_alpha=0.5,
        num_epochs=20,
        verbose=True
    )

    print("\n=================================================================")
    print("                 BENCHMARK FOLD RESULTS REPORT                   ")
    print("=================================================================")
    print(f"[✓] Benchmark Fold 1 Execution Completed Successfully!")
    print(f" -> Elapsed Time: {res['elapsed_sec']:.2f} seconds")
    print(f" -> Peak RAM Footprint: {res['mem_mb']:.2f} MB")
    print(f" -> Active Accuracy: {res['best_metrics']['active_accuracy']:.2f}%")
    print(f" -> Active Macro F1: {res['best_metrics']['active_macro_f1']:.2f}%")
    print(f" -> Minority Macro F1: {res['best_metrics']['minority_macro_f1']:.2f}%")
    print(f" -> Normal (N) Recall: {res['best_metrics']['normal_recall']:.2f}% (Baseline Floor: 96.50%)")
    print(f" -> SVEB/A Recall: {res['best_metrics']['sveb_recall']:.2f}%")
    print(f" -> VEB/PVC Recall: {res['best_metrics']['veb_recall']:.2f}%")
    print(f" -> F/VT Recall: {res['best_metrics']['f_recall']:.2f}%")
    print(f" -> Final Objective Score: {res['best_metrics']['final_score']:.4f}")
    print("=================================================================\n")

    return res


if __name__ == "__main__":
    run_benchmark_fold_1()
