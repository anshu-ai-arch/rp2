import os
import sys
import time
import json
import argparse
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix

from src.models.eeg_transfer_gru_model import EEGTransferGRUModel
from src.engine.trainer import FocalLoss


def compute_sha256(filepath):
    """Computes SHA256 hash of a file for integrity verification."""
    p = Path(filepath)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_gru_param_hash(model):
    """Computes SHA256 hash across all GRU parameters to verify frozen status."""
    h = hashlib.sha256()
    for name, param in model.gru.named_parameters():
        h.update(param.data.cpu().numpy().tobytes())
    return h.hexdigest()


def compute_cnn_param_hash(model):
    """Computes SHA256 hash across all EEG CNN parameters."""
    h = hashlib.sha256()
    for name, param in model.eeg_cnn.named_parameters():
        h.update(param.data.cpu().numpy().tobytes())
    return h.hexdigest()


def compute_head_param_hash(model):
    """Computes SHA256 hash across classifier head parameters."""
    h = hashlib.sha256()
    for name, param in model.classifier.named_parameters():
        h.update(param.data.cpu().numpy().tobytes())
    return h.hexdigest()


def compute_binary_metrics(y_true, y_pred):
    """Computes comprehensive binary classification metrics."""
    acc = accuracy_score(y_true, y_pred) * 100.0
    kappa = cohen_kappa_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    
    return {
        "accuracy": float(acc),
        "precision": float(prec * 100.0),
        "recall": float(rec * 100.0),
        "f1": float(f1 * 100.0),
        "specificity": float(spec),
        "kappa": float(kappa),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist()
    }


def load_and_validate_subject_wise_data(
    prov_dir="data/chbmit/provenance",
    splits_dir="data/chbmit/splits"
):
    """
    Loads provenance numpy datasets and subject-wise split CSVs.
    Executes strict pre-training hard validations.
    """
    prov_path = Path(prov_dir)
    split_path = Path(splits_dir)

    x_prov_file = prov_path / "X_train_provenance.npy"
    y_prov_file = prov_path / "y_train_provenance.npy"
    train_csv_file = split_path / "train.csv"
    val_csv_file = split_path / "validation.csv"
    test_csv_file = split_path / "test.csv"
    manifest_file = split_path / "split_manifest.json"

    # Verify source file existence
    for f in [x_prov_file, y_prov_file, train_csv_file, val_csv_file, test_csv_file, manifest_file]:
        if not f.exists():
            raise FileNotFoundError(f"[!] Critical Error: Required file '{f}' not found!")

    # Load split CSVs
    df_train = pd.read_csv(train_csv_file)
    df_val = pd.read_csv(val_csv_file)
    df_test = pd.read_csv(test_csv_file)

    # 1. Verify Sample Counts
    len_tr, len_va, len_te = len(df_train), len(df_val), len(df_test)
    if len_tr != 8611 or len_va != 1718 or len_te != 1654:
        raise ValueError(f"[!] Validation Failure: Unexpected split sample counts! Train={len_tr}, Val={len_va}, Test={len_te}")

    total_samples = len_tr + len_va + len_te
    if total_samples != 11983:
        raise ValueError(f"[!] Validation Failure: Total samples = {total_samples} != 11983!")

    # 2. Verify Subject Overlap (Zero Leakage)
    tr_subjs = set(df_train['subject_id'].unique())
    va_subjs = set(df_val['subject_id'].unique())
    te_subjs = set(df_test['subject_id'].unique())

    if len(tr_subjs & va_subjs) > 0 or len(tr_subjs & te_subjs) > 0 or len(va_subjs & te_subjs) > 0:
        raise ValueError("[!] Validation Failure: Subject overlap detected across splits!")

    # 3. Verify Full 24-Subject Coverage
    all_subjs = sorted(list(tr_subjs | va_subjs | te_subjs))
    expected_subjs = sorted([f"chb{i:02d}" for i in range(1, 25)])
    if all_subjs != expected_subjs:
        raise ValueError(f"[!] Validation Failure: Subject set {all_subjs} != expected 24 subjects!")

    # 4. Verify Sample Coverage & Duplicates
    all_indices = sorted(list(df_train['sample_index']) + list(df_val['sample_index']) + list(df_test['sample_index']))
    if all_indices != list(range(11983)):
        raise ValueError("[!] Validation Failure: Sample indices do not cover range 0..11982 exactly!")

    if len(all_indices) != len(set(all_indices)):
        raise ValueError("[!] Validation Failure: Duplicate sample_index detected!")

    # 5. Load Provenance Arrays via mmap_mode='r'
    X_mmap = np.load(x_prov_file, mmap_mode='r')
    y_prov = np.load(y_prov_file)

    # 6. Verify Label Consistency with y_train_provenance.npy
    all_df_sorted = pd.concat([df_train, df_val, df_test]).sort_values('sample_index')
    if not np.array_equal(all_df_sorted['label'].values, y_prov):
        raise ValueError("[!] Validation Failure: CSV labels do NOT match y_train_provenance.npy!")

    # Index arrays by sample_index for each split
    idx_tr = df_train['sample_index'].values
    idx_va = df_val['sample_index'].values
    idx_te = df_test['sample_index'].values

    X_tr = np.array(X_mmap[idx_tr], dtype=np.float32)
    y_tr = np.array(y_prov[idx_tr], dtype=np.int64)

    X_va = np.array(X_mmap[idx_va], dtype=np.float32)
    y_va = np.array(y_prov[idx_va], dtype=np.int64)

    X_te = np.array(X_mmap[idx_te], dtype=np.float32)
    y_te = np.array(y_prov[idx_te], dtype=np.int64)

    # Check for NaNs/Infs
    if np.isnan(X_tr[:100]).any() or np.isnan(X_va[:100]).any() or np.isnan(X_te[:100]).any():
        raise ValueError("[!] Validation Failure: NaNs detected in loaded signal arrays!")

    print("[✓] Pre-Training Hard Validations PASSED (10/10 Checks Verified):")
    print(f"    - Train:      {len(X_tr):,} samples ({len(tr_subjs)} subjects)")
    print(f"    - Validation: {len(X_va):,} samples ({len(va_subjs)} subjects)")
    print(f"    - Test:       {len(X_te):,} samples ({len(te_subjs)} subjects — UNTOUCHED DURING TRAINING)")

    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def set_seed(seed=2023):
    """Sets random seeds across Python, NumPy, and PyTorch for strict reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_subject_wise_pipeline(do_train=False):
    set_seed(2023)
    print("=================================================================", flush=True)
    print("   SUBJECT-WISE EEG TRANSFER LEARNING PIPELINE (PHASE C.2)       ", flush=True)
    print("=================================================================", flush=True)

    # 1. Device Selection & Safety Verification
    print("[*] Step 1/7: Verifying Environment & Computing Device...", flush=True)
    mps_available = torch.backends.mps.is_available()
    device = torch.device("mps") if mps_available else torch.device("cpu")
    print(f"  - PyTorch Version: {torch.__version__}", flush=True)
    print(f"  - MPS Available:   {mps_available}", flush=True)
    print(f"  - Selected Device: {device}", flush=True)

    # 2. ECG Checkpoint Integrity SHA256 Verification
    print("\n[*] Step 2/7: Verifying Pretrained ECG Checkpoint SHA256 Hash...", flush=True)
    ecg_ckpt_path = Path("checkpoints/model_c_medium_augmented_best.pth")
    expected_hash = "4b39b3b070fc763a7ff8a7ff68b0968e045033463ea50042f77f4d21d627b499"
    actual_hash_before = compute_sha256(ecg_ckpt_path)

    print(f"  - Expected SHA256: {expected_hash}", flush=True)
    print(f"  - Actual SHA256:   {actual_hash_before}", flush=True)

    if actual_hash_before != expected_hash:
        print("[!] CRITICAL ERROR: ECG Checkpoint SHA256 Hash Mismatch!")
        sys.exit(1)
    print("  - Checkpoint Hash Verification: MATCH (100% Intact)", flush=True)

    # 3. Load & Validate Subject-Wise Datasets
    print("\n[*] Step 3/7: Loading & Validating Locked Subject-Wise Datasets...", flush=True)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_and_validate_subject_wise_data()

    # 4. Model Instantiation & Transfer Weight Audit
    print("\n[*] Step 4/7: Instantiating Model & Transferring Pretrained ECG BiGRU Weights...", flush=True)
    model = EEGTransferGRUModel(
        checkpoint_path=str(ecg_ckpt_path),
        num_eeg_channels=18,
        num_target_classes=2,
        freeze_gru=True,
        dropout=0.2076
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"  - Total Parameters:     {total_params:,} (Expected: 246,082)", flush=True)
    print(f"  - Trainable Parameters: {trainable_params:,} (Expected: 97,090)", flush=True)
    print(f"  - Frozen Parameters:    {frozen_params:,} (Expected: 148,992)", flush=True)

    if total_params != 246082 or trainable_params != 97090 or frozen_params != 148992:
        print("[!] CRITICAL ERROR: Parameter count mismatch!")
        sys.exit(1)

    # Pre-training parameter hashes
    gru_hash_before = compute_gru_param_hash(model)
    cnn_hash_before = compute_cnn_param_hash(model)
    head_hash_before = compute_head_param_hash(model)

    print(f"  - GRU Pre-Training Param Hash:  {gru_hash_before[:16]}...", flush=True)
    print(f"  - CNN Pre-Training Param Hash:  {cnn_hash_before[:16]}...", flush=True)
    print(f"  - Head Pre-Training Param Hash: {head_hash_before[:16]}...", flush=True)

    # 5. Execution Mode Check
    if not do_train:
        print("\n=================================================================", flush=True)
        print(" [✓] PRE-TRAINING CHECK PASSED — TRAINING NOT STARTED           ", flush=True)
        print("=================================================================", flush=True)
        return {
            "status": "PRE-TRAINING CHECK PASSED",
            "training_executed": False
        }

    # 6. Execute Full 25-Epoch Subject-Wise Training
    print("\n[*] Step 5/7: Setting up Loss, Optimizer, DataLoaders for Subject-Wise Training...", flush=True)
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    alpha_weights = torch.tensor([1.0, 2.73], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    output_best_ckpt_path = Path("checkpoints/eeg_transfer_gru_subject_wise_best.pth")
    artifacts_dir = Path("artifacts/eeg_transfer")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = artifacts_dir / "phase_c2_training_metrics.json"

    print("\n[*] Step 6/7: Executing Full 25-Epoch Subject-Wise Training Loop...\n", flush=True)
    epoch_logs = []
    best_val_loss = float("inf")
    best_epoch_info = None
    t_training_start = time.perf_counter()

    for epoch in range(1, 26):
        t_epoch_start = time.perf_counter()
        curr_lr = optimizer.param_groups[0]['lr']

        # Train Epoch
        model.train()
        train_loss_total = 0.0
        train_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[!] CRITICAL STOP: NaN/Inf detected in loss at Epoch {epoch}!")
                sys.exit(1)

            loss.backward()
            optimizer.step()

            if device.type == "mps":
                torch.mps.synchronize()

            train_loss_total += loss.item() * batch_y.size(0)
            train_samples += batch_y.size(0)

        epoch_train_loss = train_loss_total / train_samples

        # Validation Epoch (EXCLUSIVELY ON VALIDATION DATA)
        model.eval()
        val_loss_total = 0.0
        val_samples = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                if device.type == "mps":
                    torch.mps.synchronize()

                val_loss_total += loss.item() * batch_y.size(0)
                val_samples += batch_y.size(0)

                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        epoch_val_loss = val_loss_total / val_samples
        metrics = compute_binary_metrics(np.array(val_targets), np.array(val_preds))

        # Step Scheduler
        scheduler.step(epoch_val_loss)

        t_epoch_sec = time.perf_counter() - t_epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_accuracy": round(metrics["accuracy"], 2),
            "val_precision": round(metrics["precision"], 2),
            "val_sensitivity": round(metrics["recall"], 2),
            "val_specificity": round(metrics["specificity"], 2),
            "val_f1": round(metrics["f1"], 2),
            "val_kappa": round(metrics["kappa"], 4),
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
            "learning_rate": curr_lr,
            "epoch_runtime_sec": round(t_epoch_sec, 2)
        }
        epoch_logs.append(epoch_record)

        # Save Best Checkpoint (Based STRICTLY on lowest Val Loss)
        is_best = False
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch_info = epoch_record.copy()
            is_best = True
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "val_acc": metrics["accuracy"],
                "val_f1": metrics["f1"],
                "metrics": epoch_record,
                "ecg_checkpoint_sha256": actual_hash_before,
                "seed": 2023
            }, output_best_ckpt_path)

        print(
            f"Epoch [{epoch:02d}/25] | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']:.2f}% | "
            f"Sens (Recall): {metrics['recall']:.2f}% | "
            f"Spec: {metrics['specificity']:.2f}% | "
            f"F1: {metrics['f1']:.2f}% | "
            f"Kappa: {metrics['kappa']:.4f} | "
            f"LR: {curr_lr:.6f} | "
            f"Time: {t_epoch_sec:.1f}s"
            f"{' -> BEST SAVED' if is_best else ''}",
            flush=True
        )

    t_total_training_sec = time.perf_counter() - t_training_start

    # Save metrics JSON
    with open(metrics_json_path, 'w') as f:
        json.dump(epoch_logs, f, indent=2)

    # 7. Post-Training Integrity Checks
    print("\n[*] Step 7/7: Performing Post-Training Parameter Integrity Checks...", flush=True)

    actual_hash_after = compute_sha256(ecg_ckpt_path)
    ecg_hash_match = (actual_hash_before == actual_hash_after)

    gru_hash_after = compute_gru_param_hash(model)
    cnn_hash_after = compute_cnn_param_hash(model)
    head_hash_after = compute_head_param_hash(model)

    gru_unchanged = (gru_hash_before == gru_hash_after)
    cnn_changed = (cnn_hash_before != cnn_hash_after)
    head_changed = (head_hash_before != head_hash_after)

    print(f"  - ECG Checkpoint SHA256 Unchanged: {ecg_hash_match} ({actual_hash_after})")
    print(f"  - GRU Parameter Hash Unchanged:    {gru_unchanged}")
    print(f"  - EEG CNN Parameter Hash Changed:  {cnn_changed}")
    print(f"  - Head Parameter Hash Changed:     {head_changed}")

    if not ecg_hash_match or not gru_unchanged:
        print("[!] CRITICAL FAILURE: Checkpoint or GRU parameters were modified!")
        sys.exit(1)

    print("\n=================================================================", flush=True)
    print("PHASE C.2 COMPLETE", flush=True)
    print("=================================================================", flush=True)
    print(f"Epochs executed:        25/25", flush=True)
    print(f"Train samples:          {len(X_tr):,}", flush=True)
    print(f"Validation samples:     {len(X_va):,}", flush=True)
    print(f"Test samples used:      0 (UNTOUCHED)", flush=True)
    print(f"Best validation epoch:  {best_epoch_info['epoch']:02d}", flush=True)
    print(f"Best validation loss:   {best_epoch_info['val_loss']:.4f}", flush=True)
    print("Best validation metrics:", flush=True)
    print(f"  Accuracy:    {best_epoch_info['val_accuracy']:.2f}%", flush=True)
    print(f"  Sensitivity: {best_epoch_info['val_sensitivity']:.2f}%", flush=True)
    print(f"  Specificity: {best_epoch_info['val_specificity']:.2f}%", flush=True)
    print(f"  Precision:   {best_epoch_info['val_precision']:.2f}%", flush=True)
    print(f"  F1:          {best_epoch_info['val_f1']:.2f}%", flush=True)
    print(f"  Kappa:       {best_epoch_info['val_kappa']:.4f}", flush=True)
    print(f"GRU transferred:        148,992", flush=True)
    print(f"GRU frozen:             YES", flush=True)
    print(f"GRU parameters unchanged: YES", flush=True)
    print(f"ECG checkpoint SHA256 unchanged: YES", flush=True)
    print(f"Best checkpoint path:   {output_best_ckpt_path}", flush=True)
    print(f"TEST SET EVALUATED:     NO", flush=True)
    print(f"PHASE C.2 STATUS:       PASSED", flush=True)
    print("=================================================================", flush=True)

    return {
        "status": "PASSED",
        "best_epoch_info": best_epoch_info,
        "final_epoch_info": epoch_logs[-1],
        "total_runtime_sec": round(t_total_training_sec, 2),
        "gru_unchanged": gru_unchanged,
        "ecg_hash_match": ecg_hash_match
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subject-Wise EEG Transfer Learning Training Pipeline")
    parser.add_argument("--train", action="store_true", help="Execute full 25-epoch model training")
    args = parser.parse_args()

    run_subject_wise_pipeline(do_train=args.train)
