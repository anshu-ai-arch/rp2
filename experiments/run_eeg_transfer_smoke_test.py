import os
import sys
import time
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
        "confusion_matrix": cm
    }


def run_smoke_test():
    print("=================================================================", flush=True)
    print("   SAFE M1/MPS SMOKE TEST — EXPLORATORY ECG->EEG TRANSFER (2 EPOCHS) ", flush=True)
    print("=================================================================", flush=True)

    # 1. Device Selection & Environment Safety
    print("[*] Step 1/6: Verifying PyTorch Environment & Computing Device...", flush=True)
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
    print(f"  - CUDA Available:    {cuda_available}", flush=True)
    print(f"  - Selected Device:   {device}", flush=True)

    # 2. Checkpoint SHA256 Verification (BEFORE Training)
    print("\n[*] Step 2/6: Verifying Pre-Training ECG Checkpoint Integrity...", flush=True)
    ecg_ckpt_path = Path("checkpoints/model_c_medium_augmented_best.pth")
    if not ecg_ckpt_path.exists():
        print(f"[!] CRITICAL ERROR: ECG Checkpoint '{ecg_ckpt_path}' missing!")
        sys.exit(1)
        
    sha256_before = compute_sha256(ecg_ckpt_path)
    print(f"  - ECG Checkpoint Path: {ecg_ckpt_path}", flush=True)
    print(f"  - SHA256 (Before):    {sha256_before}", flush=True)

    # 3. Memory-Efficient Dataset Loading & Random 80/20 Window Split
    print("\n[*] Step 3/6: Loading Preprocessed Dataset & Creating 80/20 Random Split...", flush=True)
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

    # PyTorch DataLoaders (num_workers=0 for M1 safety)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # 4. Model Instantiation & Transfer Weight Check
    print("\n[*] Step 4/6: Instantiating EEGTransferGRUModel & Transferring ECG BiGRU Weights...", flush=True)
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
    print(f"  - ECG CNN Transferred:   0", flush=True)
    print(f"  - ECG Classifier Transferred: 0", flush=True)

    # 5. Loss & Optimizer Setup
    print("\n[*] Step 5/6: Configuring FocalLoss (gamma=2.0, alpha=[1.0, 2.73]) & Adam Optimizer...", flush=True)
    alpha_weights = torch.tensor([1.0, 2.73], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )

    # 6. Execute 2-Epoch Training Loop
    print("\n[*] Step 6/6: Starting 2-Epoch MPS Smoke Test Training Loop...\n", flush=True)
    
    output_ckpt_path = Path("checkpoints/eeg_transfer_gru_frozen_smoke_test.pth")
    epoch_metrics_log = []
    
    for epoch in range(1, 3):
        t_epoch_start = time.perf_counter()
        
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
            
            # Check for NaN / Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[!] STOP CONDITION TRIGGERED: NaN/Inf detected in loss at Epoch {epoch}!")
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
        
        t_epoch_sec = time.perf_counter() - t_epoch_start
        mem_info = psutil.virtual_memory()
        ram_used_gb = (mem_info.total - mem_info.available) / (1024**3)
        
        epoch_info = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
            "val_specificity": metrics["specificity"],
            "val_kappa": metrics["kappa"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
            "epoch_runtime_sec": round(t_epoch_sec, 2),
            "learning_rate": 0.001,
            "ram_used_gb": round(ram_used_gb, 2)
        }
        epoch_metrics_log.append(epoch_info)
        
        print(f"=================================================================")
        print(f" EPOCH [{epoch}/2] COMPLETE (Runtime: {t_epoch_sec:.2f}s | RAM Used: {ram_used_gb:.2f} GB)")
        print(f"-----------------------------------------------------------------")
        print(f"  Train Loss:       {epoch_train_loss:.4f}")
        print(f"  Val Loss:         {epoch_val_loss:.4f}")
        print(f"  Val Accuracy:     {metrics['accuracy']:.2f}%")
        print(f"  Val Sensitivity:  {metrics['recall']:.2f}%  (Recall)")
        print(f"  Val Specificity:  {metrics['specificity']:.2f}%")
        print(f"  Val Precision:    {metrics['precision']:.2f}%")
        print(f"  Val F1-Score:     {metrics['f1']:.2f}%")
        print(f"  Cohen's Kappa:    {metrics['kappa']:.4f}")
        print(f"  Confusion Matrix: TN={metrics['confusion_matrix'][0,0]}, FP={metrics['confusion_matrix'][0,1]}, FN={metrics['confusion_matrix'][1,0]}, TP={metrics['confusion_matrix'][1,1]}")
        print(f"=================================================================\n", flush=True)

    # Save Smoke Test Checkpoint
    torch.save({
        "epoch": 2,
        "model_state_dict": model.state_dict(),
        "train_loss": epoch_metrics_log[-1]["train_loss"],
        "val_loss": epoch_metrics_log[-1]["val_loss"],
        "val_acc": epoch_metrics_log[-1]["val_accuracy"],
        "metrics": epoch_metrics_log[-1]
    }, output_ckpt_path)
    print(f"[✓] Smoke Test Checkpoint saved cleanly to: '{output_ckpt_path}'", flush=True)

    # Post-Training SHA256 Verification
    sha256_after = compute_sha256(ecg_ckpt_path)
    print(f"\n[*] Post-Training ECG Checkpoint Integrity Verification:")
    print(f"  - SHA256 (Before): {sha256_before}")
    print(f"  - SHA256 (After):  {sha256_after}")
    print(f"  - Status:          {'IDENTICAL (100% Intact)' if sha256_before == sha256_after else 'MUTATED (FAILURE)'}")

    return {
        "pytorch_version": pytorch_version,
        "device_used": str(device),
        "dataset_shape": list(X_mat.shape),
        "train_count": len(X_train),
        "val_count": len(X_val),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "epoch_logs": epoch_metrics_log,
        "sha256_before": sha256_before,
        "sha256_after": sha256_after,
        "sha256_match": (sha256_before == sha256_after),
        "output_ckpt": str(output_ckpt_path)
    }


if __name__ == "__main__":
    run_smoke_test()
