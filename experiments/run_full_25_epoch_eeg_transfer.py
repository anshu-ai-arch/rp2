import os
import sys
import time
import json
import hashlib
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix

from src.models.eeg_transfer_gru_model import EEGTransferGRUModel
from src.engine.trainer import FocalLoss


def compute_sha256(filepath):
    """Computes SHA256 hash of a file for safety verification."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
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
        "accuracy": acc,
        "precision": prec * 100.0,
        "recall": rec * 100.0,
        "f1": f1 * 100.0,
        "specificity": spec,
        "kappa": kappa,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist()
    }


def run_full_25_epoch_training():
    print("=================================================================", flush=True)
    print("   FULL 25-EPOCH EEG TRANSFER LEARNING (FROZEN ECG BiGRU TRACK)  ", flush=True)
    print("=================================================================", flush=True)

    # 1. Device Selection & Safety Check
    print("[*] Step 1/7: Verifying PyTorch Environment & Computing Device...", flush=True)
    pytorch_version = torch.__version__
    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    
    if mps_available:
        device = torch.device("mps")
    else:
        print("[!] Warning: MPS is NOT available. Falling back to CPU.", flush=True)
        device = torch.device("cpu")
        
    print(f"  - PyTorch Version:   {pytorch_version}", flush=True)
    print(f"  - MPS Available:     {mps_available}", flush=True)
    print(f"  - Selected Device:   {device}", flush=True)

    # 2. Checkpoint SHA256 Verification (BEFORE Training)
    print("\n[*] Step 2/7: Verifying Pre-Training ECG Checkpoint SHA256 Integrity...", flush=True)
    ecg_ckpt_path = Path("checkpoints/model_c_medium_augmented_best.pth")
    if not ecg_ckpt_path.exists():
        print(f"[!] CRITICAL ERROR: ECG Checkpoint '{ecg_ckpt_path}' missing!")
        sys.exit(1)
        
    sha256_before = compute_sha256(ecg_ckpt_path)
    print(f"  - ECG Checkpoint Path: {ecg_ckpt_path}", flush=True)
    print(f"  - SHA256 (Before):    {sha256_before}", flush=True)

    # 3. Memory-Efficient Dataset Loading & Random 80/20 Window Split
    print("\n[*] Step 3/7: Loading Preprocessed Dataset & Creating 80/20 Random Split...", flush=True)
    x_file = "data/chbmit/processed/X_train.npy"
    y_file = "data/chbmit/processed/y_train.npy"
    
    X_raw = np.load(x_file, mmap_mode='r')
    y_raw = np.load(y_file)
    
    print(f"  - Loaded X shape: {X_raw.shape}, dtype: {X_raw.dtype}", flush=True)
    print(f"  - Loaded y shape: {y_raw.shape}, dtype: {y_raw.dtype}", flush=True)

    # Memory-efficient conversion to float32
    X_mat = np.array(X_raw, dtype=np.float32)
    y_vec = np.array(y_raw, dtype=np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X_mat, y_vec, test_size=0.20, random_state=2023, stratify=y_vec
    )

    print(f"  - Train Set Count: {len(X_train):,} samples", flush=True)
    print(f"  - Val Set Count:   {len(X_val):,} samples", flush=True)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # 4. Model Instantiation & Transfer Weight Check
    print("\n[*] Step 4/7: Instantiating EEGTransferGRUModel & Transferring Pretrained BiGRU Weights...", flush=True)
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
    cnn_params = sum(p.numel() for p in model.eeg_cnn.parameters())
    gru_params = sum(p.numel() for p in model.gru.parameters())
    cls_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"  - Total Parameters:     {total_params:,}", flush=True)
    print(f"  - Trainable Parameters: {trainable_params:,} (CNN: {cnn_params:,}, Head: {cls_params:,})", flush=True)
    print(f"  - Frozen Parameters:    {frozen_params:,} (Pretrained BiGRU: {gru_params:,})", flush=True)

    # 5. Loss, Optimizer, and Scheduler Setup
    print("\n[*] Step 5/7: Setting up Loss, Optimizer, and ReduceLROnPlateau Scheduler...", flush=True)
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

    # 6. Execute Full 25-Epoch Training Loop
    print("\n[*] Step 6/7: Executing Full 25-Epoch Training Loop...\n", flush=True)
    
    output_best_ckpt_path = Path("checkpoints/eeg_transfer_gru_frozen_best.pth")
    artifacts_dir = Path("artifacts/eeg_transfer")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = artifacts_dir / "full_25_epoch_metrics.json"

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
                print(f"[!] STOP CONDITION: NaN/Inf detected in loss at Epoch {epoch}!")
                sys.exit(1)
                
            loss.backward()
            optimizer.step()
            
            if device.type == "mps":
                torch.mps.synchronize()
                
            train_loss_total += loss.item() * batch_y.size(0)
            train_samples += batch_y.size(0)
            
        epoch_train_loss = train_loss_total / train_samples
        
        # Validation Epoch
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
        mem_info = psutil.virtual_memory()
        ram_used_gb = (mem_info.total - mem_info.available) / (1024**3)
        
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
            "epoch_runtime_sec": round(t_epoch_sec, 2),
            "ram_used_gb": round(ram_used_gb, 2)
        }
        epoch_logs.append(epoch_record)
        
        # Save Best Checkpoint (Based on lowest Val Loss)
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
                "metrics": epoch_record
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

    # Save complete metric log to JSON artifact
    with open(metrics_json_path, 'w') as f:
        json.dump(epoch_logs, f, indent=2)

    # 7. Post-Training SHA256 Verification & Final Report Setup
    print("\n[*] Step 7/7: Verifying ECG Checkpoint SHA256 Integrity After Training...", flush=True)
    sha256_after = compute_sha256(ecg_ckpt_path)
    sha256_match = (sha256_before == sha256_after)
    
    print(f"  - SHA256 (Before): {sha256_before}")
    print(f"  - SHA256 (After):  {sha256_after}")
    print(f"  - Status:          {'IDENTICAL (100% Intact)' if sha256_match else 'MUTATED (CRITICAL FAILURE)'}")

    return {
        "pytorch_version": pytorch_version,
        "device_used": str(device),
        "dataset_shape": list(X_mat.shape),
        "train_count": len(X_train),
        "val_count": len(X_val),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_runtime_sec": round(t_total_training_sec, 2),
        "best_epoch_info": best_epoch_info,
        "final_epoch_info": epoch_logs[-1],
        "epoch_logs": epoch_logs,
        "sha256_before": sha256_before,
        "sha256_after": sha256_after,
        "sha256_match": sha256_match,
        "best_ckpt_path": str(output_best_ckpt_path),
        "metrics_json_path": str(metrics_json_path)
    }


if __name__ == "__main__":
    run_full_25_epoch_training()
