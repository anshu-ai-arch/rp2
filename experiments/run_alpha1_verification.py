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
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Tuple, List, Dict

sys.path.append(os.getcwd())

from experiments.run_model_c_medium_kfold_minority import (
    load_ds1_record_data,
    FOLD_PATIENTS,
    DS1_ALL_PATIENTS,
    ArrhythmiaDatasetRaw,
    compute_active_metrics
)
from train_augmented_model_c import AugmentedECGDataset
from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU
from experiments.evaluate_patient_wise import get_m1_device


def run_alpha1_fold1_verification():
    device = get_m1_device()
    print("\n=================================================================")
    print("      ECG VERIFICATION RUN: FOLD 1 WITH SAMPLING ALPHA = 1.0     ")
    print("=================================================================")
    print(f"[*] Target Device: {device}")

    # Load DS1 data
    print("[*] Loading DS1 patient record data...")
    record_data = load_ds1_record_data("config.yaml")

    val_records = FOLD_PATIENTS[1]
    train_records = [r for r in DS1_ALL_PATIENTS if r not in val_records]

    print(f" -> Validation Patient Records (Fold 1): {val_records}")
    print(f" -> Training Patient Records ({len(train_records)} records): {train_records}")

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

    # 1. Training class counts
    counts_train = {c: int(np.sum(y_train == c)) for c in range(5)}
    print(f"\n[*] Raw Training Class Counts (Fold 1):")
    for c in range(5):
        print(f"    Class {c}: {counts_train[c]:>6} beats ({counts_train[c]/len(y_train)*100:.2f}%)")

    # 2. Per-class sampling weights for alpha = 1.0
    alpha = 1.0
    class_weights = np.zeros(5, dtype=np.float64)
    for c in range(4):
        if counts_train[c] > 0:
            class_weights[c] = (1.0 / counts_train[c]) ** alpha
        else:
            class_weights[c] = 0.0
    class_weights[4] = 0.0  # Q class weight explicitly ZERO

    print(f"\n[*] Per-Class Sampling Weights (alpha = 1.0, w_4 = 0.0):")
    for c in range(5):
        print(f"    Class {c}: {class_weights[c]:.8f}")

    # 3. Expected theoretical sampling proportions for active classes (classes 0..3)
    # With alpha = 1.0, N_c * w_c = 1.0 for each active class with count > 0!
    # So each of the 4 active classes gets exactly 1.0 / 4.0 = 25.0% probability!
    total_active_weight = sum(counts_train[c] * class_weights[c] for c in range(4))
    theo_props = {c: (counts_train[c] * class_weights[c]) / total_active_weight for c in range(5)}

    print(f"\n[*] Theoretical Expected Sampling Proportions (alpha = 1.0):")
    for c in range(5):
        print(f"    Class {c}: {theo_props[c]*100:.2f}%")

    # Sample weights and WeightedRandomSampler
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    # 4. Empirical sampling simulation for 1 epoch
    np.random.seed(42)
    torch.manual_seed(42)
    sim_indices = list(sampler)
    sim_labels = y_train[sim_indices]
    sim_counts = {c: int(np.sum(sim_labels == c)) for c in range(5)}

    print(f"\n[*] Empirical Sampled Class Counts (1 Epoch Simulation, {len(sim_labels):,} beats):")
    for c in range(5):
        pct = (sim_counts[c] / len(sim_labels)) * 100.0
        print(f"    Class {c}: {sim_counts[c]:>6} beats ({pct:.2f}% empirical vs {theo_props[c]*100:.2f}% theoretical)")

    # Prepare DataLoaders
    train_ds_raw = ArrhythmiaDatasetRaw(X_train, y_train)
    val_ds_raw = ArrhythmiaDatasetRaw(X_val, y_val)

    augmented_train_ds = AugmentedECGDataset(train_ds_raw, is_train=True)

    train_loader = DataLoader(
        augmented_train_ds,
        batch_size=256,
        sampler=sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds_raw,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )

    # Initialize fresh model
    model = GenericHybrid1DBiCNNGRU(
        in_channels=1,
        cnn_channels=[64, 128, 128],
        kernel_sizes=[5, 5, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.2076,
        num_classes=5
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[*] Model Initialized (Fresh Scratch Initialization): {trainable_params:,} parameters")

    criterion = nn.CrossEntropyLoss(ignore_index=4)
    optimizer = optim.Adam(model.parameters(), lr=0.002018, weight_decay=2.35e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "ecg_kfold_alpha1_fold1_20ep_verification_best.pth"

    best_final_score = -999.0
    best_metrics = {}
    best_epoch = 0
    epoch_logs = []

    start_time = time.time()
    process = psutil.Process(os.getpid())

    print("\n[*] Starting 20 Epochs Training...")
    for epoch in range(1, 21):
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

        # Evaluate on validation
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

        # Compute active confusion matrix
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

        print(f"Epoch [{epoch:02d}/20] - Train Loss: {epoch_train_loss:.4f} | Val Score: {metrics['final_score']:.4f} | Val Acc: {metrics['active_accuracy']:.2f}% | Macro F1: {metrics['active_macro_f1']:.2f}% | Min F1: {metrics['minority_macro_f1']:.2f}% | N Rec: {metrics['normal_recall']:.2f}% | SVEB Rec: {metrics['sveb_recall']:.2f}% | VEB Rec: {metrics['veb_recall']:.2f}% | F Rec: {metrics['f_recall']:.2f}%")

        if metrics["final_score"] > best_final_score:
            best_final_score = metrics["final_score"]
            best_metrics = metrics
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "hyperparameters": {
                        "alpha": 1.0,
                        "lr": 0.002018,
                        "weight_decay": 2.35e-5,
                        "batch_size": 256,
                        "fold": 1
                    }
                },
                checkpoint_path
            )

    elapsed_time = time.time() - start_time
    mem_mb = process.memory_info().rss / (1024.0 * 1024.0)

    print("\n=================================================================")
    print("            VERIFICATION RUN COMPLETED SUCCESSFULLY              ")
    print("=================================================================")
    print(f"[*] Best Validation Epoch: Epoch {best_epoch} (Final Score: {best_final_score:.4f})")
    print(f"[*] Checkpoint Saved To: '{checkpoint_path}'")
    print(f"[*] Total Elapsed Time: {elapsed_time:.2f} seconds ({elapsed_time/60.0:.2f} minutes)")
    print(f"[*] Peak RAM Memory Footprint: {mem_mb:.2f} MB")

    # Load best model metrics & confusion matrix
    best_log = [l for l in epoch_logs if l["epoch"] == best_epoch][0]
    best_conf_mat = np.array(best_log["confusion_matrix"])

    print("\n[*] Best Epoch Validation Metrics Breakdown:")
    print(f" -> Active Accuracy: {best_metrics['active_accuracy']:.2f}%")
    print(f" -> Active Macro F1: {best_metrics['active_macro_f1']:.2f}%")
    print(f" -> Minority Macro F1: {best_metrics['minority_macro_f1']:.2f}%")
    print(f" -> Normal (N) Recall: {best_metrics['normal_recall']:.2f}% (Prec: {best_metrics['per_class'][0]['precision']:.2f}%, F1: {best_metrics['per_class'][0]['f1']:.2f}%)")
    print(f" -> SVEB/A Recall: {best_metrics['sveb_recall']:.2f}% (Prec: {best_metrics['per_class'][1]['precision']:.2f}%, F1: {best_metrics['per_class'][1]['f1']:.2f}%)")
    print(f" -> VEB/PVC Recall: {best_metrics['veb_recall']:.2f}% (Prec: {best_metrics['per_class'][2]['precision']:.2f}%, F1: {best_metrics['per_class'][2]['f1']:.2f}%)")
    print(f" -> F/VT Recall: {best_metrics['f_recall']:.2f}% (Prec: {best_metrics['per_class'][3]['precision']:.2f}%, F1: {best_metrics['per_class'][3]['f1']:.2f}%)")
    print(f" -> Recall Floor Penalty Applied: {best_metrics['recall_penalty']:.4f}")

    print("\n[*] Active Confusion Matrix (Classes 0=N, 1=SVEB, 2=VEB, 3=F):")
    print("            Pred N   Pred SVEB  Pred VEB   Pred F")
    for i, c_name in enumerate(["True N   ", "True SVEB", "True VEB ", "True F   "]):
        print(f"{c_name}: {best_conf_mat[i, 0]:>8} {best_conf_mat[i, 1]:>10} {best_conf_mat[i, 2]:>10} {best_conf_mat[i, 3]:>8}")

    # Save results JSON
    results_json = Path("experiments/results_ecg_kfold_alpha1_fold1_20ep_verification.json")
    save_summary = {
        "experiment": "Fold 1 Verification (alpha=1.0, 20 epochs)",
        "hyperparameters": {
            "alpha": 1.0,
            "lr": 0.002018,
            "weight_decay": 2.35e-5,
            "batch_size": 256,
            "epochs": 20,
            "fold": 1
        },
        "train_records": train_records,
        "val_records": val_records,
        "counts_train": counts_train,
        "class_weights": class_weights.tolist(),
        "theo_props": {k: float(v) for k, v in theo_props.items()},
        "sim_counts": sim_counts,
        "elapsed_sec": elapsed_time,
        "mem_mb": mem_mb,
        "best_epoch": best_epoch,
        "best_final_score": best_final_score,
        "best_metrics": best_metrics,
        "best_confusion_matrix": best_conf_mat.tolist(),
        "epoch_logs": epoch_logs
    }

    with open(results_json, "w") as f:
        json.dump(save_summary, f, indent=2)

    print(f"\n[✓] Results JSON successfully saved to '{results_json}'")
    return save_summary


if __name__ == "__main__":
    run_alpha1_fold1_verification()
