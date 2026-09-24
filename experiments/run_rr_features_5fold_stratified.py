"""
Controlled ECG Research Experiment:
Patient-Level Stratified 5-Fold Cross-Validation with Fused Waveform + 4 RR-Interval Features

RESEARCH QUESTION:
Does adding 4 beat-timing/RR-derived features (pre_RR, post_RR, local_mean_RR, rr_ratio) to the
existing 1D ECG waveform representation improve recognition of minority ECG classes (especially SVEB/A)
under patient-level stratified 5-fold cross-validation while preserving Normal-class performance?

SINGLE CONTROLLED INTERVENTION:
- Input: 1D Waveform [B, 1, 256] + 4 Normalized RR Features [B, 4]
- 4 RR Features Definition:
  1. pre_RR: (R_peak[i] - R_peak[i-1]) / raw_fs (seconds)
  2. post_RR: (R_peak[i+1] - R_peak[i]) / raw_fs (seconds)
  3. local_mean_RR: Mean of up to 10 previous valid RR intervals (seconds)
  4. rr_ratio: pre_RR / local_mean_RR
- Boundary beats (e.g. idx == 0) with incomplete RR context are marked invalid and excluded (22 beats total).
- RR feature normalization: Fitted ONLY on training fold (mean & std), applied unchanged to validation fold.

STRICT RESEARCH CONSTRAINTS & ISOLATION:
1. Patient-level 5-fold partition reused exactly from established 5-fold CV experiment:
   - Fold 1: ['115', '119', '122', '207', '230'] (5 patients)
   - Fold 2: ['101', '112', '118', '208', '220'] (5 patients)
   - Fold 3: ['114', '124', '205', '223']      (4 patients)
   - Fold 4: ['108', '109', '201', '215']      (4 patients)
   - Fold 5: ['106', '116', '203', '209']      (4 patients)
2. DS1 patient set (22 records) only. DS2 test set is COMPLETELY LOCKED AND UNTOUCHED.
3. Active classes: N (0), SVEB/A (1), VEB/PVC (2), F/VT (3).
4. Q class (4) is 100% EXCLUDED from training loss (ignore_index=4) and active evaluation metrics.
5. Ordinary CrossEntropyLoss(ignore_index=4). NO focal loss, NO class weights, NO alpha sampling.
6. Hyperparameters: lr=0.002018, wd=2.35e-5, batch_size=256, epochs=20.
7. Architecture: Hybrid1DCNNBiGRU_RR (259,669 parameters).
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
import wfdb
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import get_device
from src.data.preprocessor import ECGPreprocessor
from src.data.beat_extractor import AAMI_MAPPING
from experiments.run_model_c_medium_kfold_minority import DS1_ALL_PATIENTS, compute_active_metrics

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
    "learning_rate": 0.002018,
    "weight_decay": 2.35e-5,
    "batch_size": 256,
    "epochs": 20
}


class Hybrid1DCNNBiGRU_RR(nn.Module):
    """
    Fused Architecture combining 1D CNN + BiGRU Waveform Representation with 4 RR-Interval Features.
    
    Waveform Branch:
        Input [B, 1, 256] -> Conv1D(64, k=5) -> Conv1D(128, k=5) -> Conv1D(128, k=3)
        -> BiGRU(hidden=64, 2 layers) -> Mean Pooling -> [B, 128]
    RR Feature Branch:
        Input [B, 4] -> Linear(4, 16) -> ReLU() -> [B, 16]
    Fusion:
        Concat([B, 128], [B, 16]) -> [B, 144]
        -> Linear(144, 128) -> ReLU() -> Dropout(0.2076) -> Linear(128, 5)
    
    Total Trainable Parameters: 259,669.
    """
    def __init__(self, in_channels=1, cnn_channels=[64, 128, 128], kernel_sizes=[5, 5, 3],
                 gru_hidden_size=64, gru_num_layers=2, dropout=0.2076, num_classes=5):
        super().__init__()
        conv_layers = []
        curr_ch = in_channels
        for out_ch, k_size in zip(cnn_channels, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(curr_ch, out_ch, kernel_size=k_size, padding=k_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
            curr_ch = out_ch
        self.cnn = nn.Sequential(*conv_layers)

        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.rr_branch = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(144, 128)
        self.relu_fc1 = nn.ReLU()
        self.drop_fc1 = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x_wave: torch.Tensor, x_rr: torch.Tensor) -> torch.Tensor:
        feat_cnn = self.cnn(x_wave)                     # [B, 128, 32]
        feat_cnn_perm = feat_cnn.permute(0, 2, 1)      # [B, 32, 128]
        gru_out, _ = self.gru(feat_cnn_perm)            # [B, 32, 128]
        wave_emb = torch.mean(gru_out, dim=1)           # [B, 128]

        rr_emb = self.rr_branch(x_rr)                   # [B, 16]
        fused = torch.cat([wave_emb, rr_emb], dim=1)    # [B, 144]

        h = self.drop_fc1(self.relu_fc1(self.fc1(fused)))
        logits = self.classifier(h)                     # [B, 5]
        return logits


class ECG_RR_Dataset(Dataset):
    """
    PyTorch Dataset storing 1D Waveforms, 4 RR Features, and AAMI Labels.
    """
    def __init__(self, waveforms_1d: np.ndarray, rr_features: np.ndarray, labels: np.ndarray):
        self.waveforms_1d = torch.tensor(waveforms_1d, dtype=torch.float32).unsqueeze(1) # [N, 1, 256]
        self.rr_features = torch.tensor(rr_features, dtype=torch.float32)                 # [N, 4]
        self.labels = torch.tensor(labels, dtype=torch.long)                               # [N]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.waveforms_1d[idx], self.rr_features[idx], self.labels[idx]


def extract_ds1_record_data_with_rr(config_path: str = "config.yaml") -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], Dict[str, Dict]]:
    """
    Extracts 1D waveforms and 4 RR features for all DS1 patient records.
    Filters out boundary beats where pre_RR or post_RR cannot be computed reliably.
    """
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    env_data_dir = os.getenv("ECG_DATA_DIR")
    if env_data_dir:
        data_dir = Path(env_data_dir)
    else:
        raw_dir = Path(config["data"]["data_dir"])
        data_dir = raw_dir if raw_dir.is_absolute() else (project_root / raw_dir)

    raw_fs = float(config["data"]["raw_sampling_rate"])
    target_fs = float(config["data"]["target_sampling_rate"])
    beat_window_size = int(config["data"].get("beat_window_size", 256))
    half_window = beat_window_size // 2

    preprocessor = ECGPreprocessor(
        raw_fs=raw_fs,
        target_fs=target_fs,
        lowcut=config["data"]["lowcut"],
        highcut=config["data"]["highcut"],
        filter_order=config["data"]["filter_order"],
        segment_duration_sec=config["data"]["segment_duration_sec"],
        matrix_rows=16,
        matrix_cols=16,
        use_filtering=config["data"].get("use_filtering", True),
    )

    record_data = {}
    record_stats = {}

    for rec_id in DS1_ALL_PATIENTS:
        record_path = data_dir / rec_id
        record = wfdb.rdrecord(str(record_path))
        ann = wfdb.rdann(str(record_path), 'atr')

        ecg_signal = record.p_signal[:, 0]
        processed_signal = preprocessor.bandpass_filter(ecg_signal)
        resampled_signal = preprocessor.resample_signal(processed_signal)

        r_peaks = ann.sample
        symbols = ann.symbol
        sig_len = len(resampled_signal)
        ratio = target_fs / raw_fs

        valid_indices = [idx for idx, sym in enumerate(symbols) if sym in AAMI_MAPPING]

        # Calculate all RR intervals in seconds
        all_rr_sec = np.diff(r_peaks) / raw_fs # length = len(r_peaks) - 1

        beats_1d = []
        rr_feats = []
        labels = []

        excluded_count = 0

        for idx in valid_indices:
            # Boundary beat check: need pre_RR (idx > 0) and post_RR (idx < len(r_peaks) - 1)
            if idx == 0 or idx >= len(r_peaks) - 1:
                excluded_count += 1
                continue

            pre_rr = (r_peaks[idx] - r_peaks[idx - 1]) / raw_fs
            post_rr = (r_peaks[idx + 1] - r_peaks[idx]) / raw_fs

            start_k = max(0, idx - 10)
            prev_rrs = all_rr_sec[start_k:idx]
            if len(prev_rrs) == 0:
                excluded_count += 1
                continue

            local_mean_rr = float(np.mean(prev_rrs))
            rr_ratio = float(pre_rr / (local_mean_rr + 1e-8))

            r_idx = int(r_peaks[idx] * ratio) if raw_fs != target_fs else r_peaks[idx]
            start_idx = r_idx - half_window
            end_idx = r_idx + half_window

            if start_idx < 0 or end_idx > sig_len:
                excluded_count += 1
                continue

            beat_wave = resampled_signal[start_idx:end_idx]
            if len(beat_wave) != beat_window_size:
                excluded_count += 1
                continue

            # Per-beat z-score normalization
            mean_val = np.mean(beat_wave)
            std_val = np.std(beat_wave)
            std_val = std_val if std_val > 1e-8 else 1.0
            norm_wave = (beat_wave - mean_val) / std_val

            beats_1d.append(norm_wave)
            rr_feats.append([pre_rr, post_rr, local_mean_rr, rr_ratio])
            labels.append(AAMI_MAPPING[symbols[idx]])

        record_data[rec_id] = (
            np.array(beats_1d, dtype=np.float32),
            np.array(rr_feats, dtype=np.float32),
            np.array(labels, dtype=np.int64)
        )
        record_stats[rec_id] = {
            "valid_beats": len(labels),
            "excluded_boundary_beats": excluded_count
        }

    return record_data, record_stats


def audit_and_verify_rr_setup(record_data, record_stats):
    """
    Performs assertions, boundary beat count report, and RR feature verification.
    """
    print("\n=================================================================")
    print("      WAVEFORM + RR FEATURES 5-FOLD CV AUDIT & SETUP CHECK       ")
    print("=================================================================")

    total_valid = sum(record_stats[r]["valid_beats"] for r in DS1_ALL_PATIENTS)
    total_excluded = sum(record_stats[r]["excluded_boundary_beats"] for r in DS1_ALL_PATIENTS)

    print(f"[*] Total DS1 Patients:             {len(DS1_ALL_PATIENTS)}")
    print(f"[*] Total Valid Beats (with RR):   {total_valid:,}")
    print(f"[*] Total Excluded Boundary Beats: {total_excluded} ({total_excluded/(total_valid+total_excluded)*100:.2f}%)")

    # Sample beat inspection
    sample_rec = "207"
    sample_rr = record_data[sample_rec][1][:3]
    print(f"\n[*] Sample RR Feature Vector Inspection (Record {sample_rec}, First 3 Beats):")
    for b_idx, rr in enumerate(sample_rr):
        print(f"    Beat {b_idx+1}: pre_RR = {rr[0]:.4f}s | post_RR = {rr[1]:.4f}s | local_mean_RR = {rr[2]:.4f}s | rr_ratio = {rr[3]:.4f}")

    all_val_patients = []
    for fold_idx in range(1, 6):
        val_recs = FIXED_5FOLDS[fold_idx]
        train_recs = [r for r in DS1_ALL_PATIENTS if r not in val_recs]
        all_val_patients.extend(val_recs)

        # Assertions
        assert len(set(val_recs).intersection(set(train_recs))) == 0, f"Patient overlap in Fold {fold_idx}"
        assert len(set(val_recs).intersection(set(DS2_TEST_RECORDINGS))) == 0, f"DS2 leakage in Fold {fold_idx}"

    assert sorted(all_val_patients) == sorted(DS1_ALL_PATIENTS), "Validation union mismatch!"
    print("\n[✓] ALL FOLD-BALANCE AND DISJOINTNESS ASSERTIONS PASSED CLEANLY!")


def run_sanity_check(device: torch.device):
    """
    Runs forward and backward sanity check for Hybrid1DCNNBiGRU_RR model.
    """
    print("\n[*] Running Hybrid1DCNNBiGRU_RR Forward/Backward Sanity Check...")
    model = Hybrid1DCNNBiGRU_RR().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    [✓] Model Parameter Count: {param_count:,} (Expected: 259,669)")
    assert param_count == 259669, f"Parameter count mismatch! Expected 259,669, got {param_count}"

    dummy_wave = torch.randn(128, 1, 256, device=device)
    dummy_rr = torch.randn(128, 4, device=device)
    dummy_targets = torch.randint(0, 5, (128,), device=device)

    criterion = nn.CrossEntropyLoss(ignore_index=4)
    optimizer = optim.Adam(model.parameters(), lr=0.002018, weight_decay=2.35e-5)

    model.train()
    optimizer.zero_grad()
    outputs = model(dummy_wave, dummy_rr)
    assert outputs.shape == (128, 5), f"Output shape mismatch! Expected (128, 5), got {outputs.shape}"

    loss = criterion(outputs, dummy_targets)
    assert torch.isfinite(loss), "Loss is not finite!"

    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Gradient for {name} is not finite!"

    optimizer.step()
    print("    [✓] Sanity Pass Passed: Output Shape [128, 5], Loss & Gradients Finite.")
    del model, optimizer
    gc.collect()


def run_rr_features_5fold_experiment(audit_only: bool = False):
    device = get_device()
    print("\n=================================================================")
    print("   PATIENT-LEVEL STRATIFIED 5-FOLD CV: WAVEFORM + RR FEATURES    ")
    print("=================================================================")
    print(f"[*] Target Acceleration Device: {device}")
    print(f"[*] Fixed Training Hyperparameters:")
    print(f"    - learning_rate: {FIXED_HP['learning_rate']}")
    print(f"    - weight_decay:  {FIXED_HP['weight_decay']}")
    print(f"    - batch_size:    {FIXED_HP['batch_size']}")
    print(f"    - epochs:        {FIXED_HP['epochs']}")

    print("\n[*] Extracting DS1 waveforms and 4 RR features...")
    record_data, record_stats = extract_ds1_record_data_with_rr("config.yaml")

    audit_and_verify_rr_setup(record_data, record_stats)
    run_sanity_check(device)

    if audit_only:
        print("\n[!] AUDIT ONLY MODE COMPLETE. Stopping before 5-fold training.")
        return

    checkpoints_dir = PROJECT_ROOT / "checkpoints/rr_features_5fold_stratified"
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

        # Assemble raw training and validation arrays
        X_train_wave = np.concatenate([record_data[r][0] for r in train_records], axis=0)
        X_train_rr_raw = np.concatenate([record_data[r][1] for r in train_records], axis=0)
        y_train = np.concatenate([record_data[r][2] for r in train_records], axis=0)

        X_val_wave = np.concatenate([record_data[r][0] for r in val_records], axis=0)
        X_val_rr_raw = np.concatenate([record_data[r][1] for r in val_records], axis=0)
        y_val = np.concatenate([record_data[r][2] for r in val_records], axis=0)

        # STRICT: Fit RR normalization statistics ONLY on training fold
        rr_mean = np.mean(X_train_rr_raw, axis=0)
        rr_std = np.std(X_train_rr_raw, axis=0)
        rr_std[rr_std < 1e-8] = 1.0

        # Apply training-fold mean/std to train and val
        X_train_rr_norm = (X_train_rr_raw - rr_mean) / rr_std
        X_val_rr_norm = (X_val_rr_raw - rr_mean) / rr_std

        # Verify no NaN or Inf
        assert not np.isnan(X_train_rr_norm).any() and not np.isinf(X_train_rr_norm).any(), "NaN/Inf in X_train_rr_norm!"
        assert not np.isnan(X_val_rr_norm).any() and not np.isinf(X_val_rr_norm).any(), "NaN/Inf in X_val_rr_norm!"

        counts_train = {c: int(np.sum(y_train == c)) for c in range(5)}
        counts_val = {c: int(np.sum(y_val == c)) for c in range(5)}

        train_ds = ECG_RR_Dataset(X_train_wave, X_train_rr_norm, y_train)
        val_ds = ECG_RR_Dataset(X_val_wave, X_val_rr_norm, y_val)

        train_loader = DataLoader(train_ds, batch_size=FIXED_HP["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=FIXED_HP["batch_size"], shuffle=False, num_workers=0)

        model = Hybrid1DCNNBiGRU_RR().to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=4)
        optimizer = optim.Adam(model.parameters(), lr=FIXED_HP["learning_rate"], weight_decay=FIXED_HP["weight_decay"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        ckpt_path = checkpoints_dir / f"ecg_rr_features_5fold_fold{fold_idx}_best.pth"

        best_score = -999.0
        best_metrics = {}
        best_epoch = 0
        best_conf_mat = None
        epoch_logs = []

        print(f"\n[*] Starting Training Fold {fold_idx}/5 (20 Epochs)...")
        for epoch in range(1, FIXED_HP["epochs"] + 1):
            model.train()
            train_loss, train_total = 0.0, 0

            for b_wave, b_rr, targets in train_loader:
                b_wave = b_wave.to(device)
                b_rr = b_rr.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(b_wave, b_rr)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * targets.size(0)
                train_total += targets.size(0)

            epoch_train_loss = train_loss / train_total

            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for b_wave, b_rr, targets in val_loader:
                    b_wave = b_wave.to(device)
                    b_rr = b_rr.to(device)
                    outputs = model(b_wave, b_rr)
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
                        "rr_mean": rr_mean.tolist(),
                        "rr_std": rr_std.tolist(),
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
            "rr_norm_mean": rr_mean.tolist(),
            "rr_norm_std": rr_std.tolist(),
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
    sveb_recs = [fold_results[f"fold_{k}"]["metrics"]["per_class"][1]["recall"] for k in range(1, 6)]
    sveb_f1s = [fold_results[f"fold_{k}"]["metrics"]["per_class"][1]["f1"] for k in range(1, 6)]
    veb_recs = [fold_results[f"fold_{k}"]["metrics"]["per_class"][2]["recall"] for k in range(1, 6)]
    veb_f1s = [fold_results[f"fold_{k}"]["metrics"]["per_class"][2]["f1"] for k in range(1, 6)]
    f_recs = [fold_results[f"fold_{k}"]["metrics"]["per_class"][3]["recall"] for k in range(1, 6)]
    f_f1s = [fold_results[f"fold_{k}"]["metrics"]["per_class"][3]["f1"] for k in range(1, 6)]
    scores = [fold_results[f"fold_{k}"]["best_score"] for k in range(1, 6)]

    aggregate_summary = {
        "mean_active_accuracy": float(np.mean(accs)), "std_active_accuracy": float(np.std(accs)),
        "mean_active_macro_f1": float(np.mean(macro_f1s)), "std_active_macro_f1": float(np.std(macro_f1s)),
        "mean_minority_macro_f1": float(np.mean(minority_f1s)), "std_minority_macro_f1": float(np.std(minority_f1s)),
        "mean_n_recall": float(np.mean(n_recs)), "std_n_recall": float(np.std(n_recs)),
        "mean_n_f1": float(np.mean(n_f1s)), "std_n_f1": float(np.std(n_f1s)),
        "mean_sveb_recall": float(np.mean(sveb_recs)), "std_sveb_recall": float(np.std(sveb_recs)),
        "mean_sveb_f1": float(np.mean(sveb_f1s)), "std_sveb_f1": float(np.std(sveb_f1s)),
        "mean_veb_recall": float(np.mean(veb_recs)), "std_veb_recall": float(np.std(veb_recs)),
        "mean_veb_f1": float(np.mean(veb_f1s)), "std_veb_f1": float(np.std(veb_f1s)),
        "mean_f_recall": float(np.mean(f_recs)), "std_f_recall": float(np.std(f_recs)),
        "mean_f_f1": float(np.mean(f_f1s)), "std_f_f1": float(np.std(f_f1s)),
        "mean_fold_score": float(np.mean(scores)), "std_fold_score": float(np.std(scores)),
        "elapsed_sec": elapsed_time,
        "mem_mb": mem_mb
    }

    print("\n=================================================================")
    print("    AGGREGATE 5-FOLD RESULTS: WAVEFORM + 4 RR FEATURES (CE)     ")
    print("=================================================================")
    print(f"[*] Mean Fold Score:       {aggregate_summary['mean_fold_score']:.4f} ± {aggregate_summary['std_fold_score']:.4f}")
    print(f"[*] Mean Active Accuracy:  {aggregate_summary['mean_active_accuracy']:.2f}% ± {aggregate_summary['std_active_accuracy']:.2f}%")
    print(f"[*] Mean Active Macro F1:  {aggregate_summary['mean_active_macro_f1']:.2f}% ± {aggregate_summary['std_active_macro_f1']:.2f}%")
    print(f"[*] Mean Minority Macro F1:{aggregate_summary['mean_minority_macro_f1']:.2f}% ± {aggregate_summary['std_minority_macro_f1']:.2f}%")
    print(f"[*] Mean Normal (N) Recall:{aggregate_summary['mean_n_recall']:.2f}% ± {aggregate_summary['std_n_recall']:.2f}%")
    print(f"[*] Mean Normal (N) F1:    {aggregate_summary['mean_n_f1']:.2f}% ± {aggregate_summary['std_n_f1']:.2f}%")
    print(f"[*] Mean SVEB/A Recall:    {aggregate_summary['mean_sveb_recall']:.2f}% ± {aggregate_summary['std_sveb_recall']:.2f}%")
    print(f"[*] Mean SVEB/A F1:        {aggregate_summary['mean_sveb_f1']:.2f}% ± {aggregate_summary['std_sveb_f1']:.2f}%")
    print(f"[*] Mean VEB/PVC Recall:   {aggregate_summary['mean_veb_recall']:.2f}% ± {aggregate_summary['std_veb_recall']:.2f}%")
    print(f"[*] Mean VEB/PVC F1:       {aggregate_summary['mean_veb_f1']:.2f}% ± {aggregate_summary['std_veb_f1']:.2f}%")
    print(f"[*] Mean F/VT Recall:      {aggregate_summary['mean_f_recall']:.2f}% ± {aggregate_summary['std_f_recall']:.2f}%")
    print(f"[*] Mean F/VT F1:          {aggregate_summary['mean_f_f1']:.2f}% ± {aggregate_summary['std_f_f1']:.2f}%")
    print("=================================================================\n")

    save_data = {
        "experiment": "Patient-Level Stratified 5-Fold CV: Waveform + 4 RR Features (CE Loss)",
        "model_architecture": "Hybrid1DCNNBiGRU_RR (259,669 parameters)",
        "hyperparameters": FIXED_HP,
        "record_statistics": record_stats,
        "aggregate_summary": aggregate_summary,
        "fold_results": fold_results
    }

    results_json = PROJECT_ROOT / "experiments/results_rr_features_5fold_stratified.json"
    with open(results_json, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"[✓] Successfully saved RR feature 5-fold evaluation results JSON to '{results_json}'")
    return save_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5-Fold Patient-Level Stratified Waveform + RR Features ECG Evaluation")
    parser.add_argument("--audit-only", action="store_true", help="Run fold construction audit and RR feature sanity checks without training")
    args = parser.parse_args()

    run_rr_features_5fold_experiment(audit_only=args.audit_only)
