import os
import sys
import gc
import time
import json
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Tuple, List, Dict

sys.path.append(os.getcwd())

from experiments.run_model_c_medium_kfold_minority import (
    load_ds1_record_data,
    DS1_ALL_PATIENTS,
    ArrhythmiaDatasetRaw,
    compute_active_metrics
)
from train_augmented_model_c import AugmentedECGDataset
from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU
from experiments.evaluate_patient_wise import get_m1_device

FIXED_3FOLDS = {
    1: ['207', '118', '106', '112'],
    2: ['208', '209', '114', '115'],
    3: ['223', '201', '109', '122']
}

FIXED_HP = {
    "sampling_alpha": 0.7320,
    "learning_rate": 0.001184,
    "weight_decay": 7.114476e-4,
    "batch_size": 128,
    "epochs": 20
}


def run_fixed_alpha0732_3fold_experiment():
    device = get_m1_device()
    print("\n=================================================================")
    print("   FIXED-HYPERPARAMETER 3-FOLD PATIENT-LEVEL ECG EVALUATION    ")
    print("=================================================================")
    print(f"[*] Target Acceleration Device: {device}")
    print(f"[*] Fixed Hyperparameters:")
    print(f"    - sampling_alpha: {FIXED_HP['sampling_alpha']}")
    print(f"    - learning_rate:  {FIXED_HP['learning_rate']}")
    print(f"    - weight_decay:   {FIXED_HP['weight_decay']}")
    print(f"    - batch_size:     {FIXED_HP['batch_size']}")
    print(f"    - epochs:         {FIXED_HP['epochs']}")

    project_root = Path(__file__).resolve().parents[1]

    print("\n[*] Loading DS1 patient record data...")
    record_data = load_ds1_record_data(str(project_root / "config.yaml"))

    checkpoints_dir = project_root / "checkpoints/ecg_fixed_alpha0732_3fold"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    fold_results = {}
    start_time = time.time()
    process = psutil.Process(os.getpid())

    for fold_idx in range(1, 4):
        val_records = FIXED_3FOLDS[fold_idx]
        train_records = [r for r in DS1_ALL_PATIENTS if r not in val_records]

        print(f"\n=================================================================")
        print(f"                  EXECUTING FOLD {fold_idx}/3                    ")
        print(f"=================================================================")
        print(f" -> Validation Patient Records: {val_records}")
        print(f" -> Training Patient Records ({len(train_records)} records): {train_records}")

        train_1d_list = [record_data[r][0] for r in train_records]
        train_y_list = [record_data[r][1] for r in train_records]
        X_train = np.concatenate(train_1d_list, axis=0)
        y_train = np.concatenate(train_y_list, axis=0)

        val_1d_list = [record_data[r][0] for r in val_records]
        val_y_list = [record_data[r][1] for r in val_records]
        X_val = np.concatenate(val_1d_list, axis=0)
        y_val = np.concatenate(val_y_list, axis=0)

        # 1. Training class counts
        counts_train = {c: int(np.sum(y_train == c)) for c in range(5)}
        counts_val = {c: int(np.sum(y_val == c)) for c in range(5)}

        print(f"\n[*] Raw Training Class Counts (Fold {fold_idx}):")
        for c in range(5):
            print(f"    Class {c}: {counts_train[c]:>6} beats ({counts_train[c]/len(y_train)*100:.2f}%)")

        # 2. Class weights for alpha = 0.7320
        class_weights = np.zeros(5, dtype=np.float64)
        for c in range(4):
            if counts_train[c] > 0:
                class_weights[c] = (1.0 / counts_train[c]) ** FIXED_HP["sampling_alpha"]
            else:
                class_weights[c] = 0.0
        class_weights[4] = 0.0  # Q class weight ZERO

        print(f"\n[*] Per-Class Sampling Weights (alpha = 0.7320):")
        for c in range(5):
            print(f"    Class {c}: {class_weights[c]:.8f}")

        # 3. Theoretical & Empirical Sampling Proportions
        total_active_weight = sum(counts_train[c] * class_weights[c] for c in range(4))
        theo_props = {c: (counts_train[c] * class_weights[c]) / total_active_weight for c in range(5)}

        print(f"\n[*] Theoretical Expected Sampling Proportions:")
        for c in range(5):
            print(f"    Class {c}: {theo_props[c]*100:.2f}%")

        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True
        )

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
            batch_size=FIXED_HP["batch_size"],
            sampler=sampler,
            num_workers=0
        )
        val_loader = DataLoader(
            val_ds_raw,
            batch_size=FIXED_HP["batch_size"],
            shuffle=False,
            num_workers=0
        )

        # Initialize FRESH model for each fold
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
        optimizer = optim.Adam(model.parameters(), lr=FIXED_HP["learning_rate"], weight_decay=FIXED_HP["weight_decay"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        ckpt_path = checkpoints_dir / f"ecg_fixed_alpha0732_fold{fold_idx}_best.pth"

        best_score = -999.0
        best_metrics = {}
        best_epoch = 0
        best_conf_mat = None
        epoch_logs = []

        print(f"\n[*] Starting Training Fold {fold_idx}/3 (20 Epochs)...")
        for epoch in range(1, FIXED_HP["epochs"] + 1):
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

            # Evaluation
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
                        "metrics": metrics,
                        "confusion_matrix": best_conf_mat,
                        "hyperparameters": FIXED_HP
                    },
                    ckpt_path
                )

        fold_results[f"fold_{fold_idx}"] = {
            "fold_idx": fold_idx,
            "train_records": train_records,
            "val_records": val_records,
            "counts_train": counts_train,
            "counts_val": counts_val,
            "class_weights": class_weights.tolist(),
            "theo_props": theo_props,
            "sim_counts": sim_counts,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "metrics": best_metrics,
            "confusion_matrix": best_conf_mat,
            "epoch_logs": epoch_logs
        }

        print(f"\n[✓] Fold {fold_idx}/3 Completed! Best Score: {best_score:.4f} at Epoch {best_epoch}")

        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    mem_mb = process.memory_info().rss / (1024.0 * 1024.0)

    # Calculate Aggregate 3-Fold Metrics
    accs = [fold_results[f"fold_{k}"]["metrics"]["active_accuracy"] for k in range(1, 4)]
    macro_f1s = [fold_results[f"fold_{k}"]["metrics"]["active_macro_f1"] for k in range(1, 4)]
    minority_f1s = [fold_results[f"fold_{k}"]["metrics"]["minority_macro_f1"] for k in range(1, 4)]
    n_recs = [fold_results[f"fold_{k}"]["metrics"]["normal_recall"] for k in range(1, 4)]
    n_f1s = [fold_results[f"fold_{k}"]["metrics"]["per_class"][0]["f1"] for k in range(1, 4)]
    sveb_f1s = [fold_results[f"fold_{k}"]["metrics"]["per_class"][1]["f1"] for k in range(1, 4)]
    veb_f1s = [fold_results[f"fold_{k}"]["metrics"]["per_class"][2]["f1"] for k in range(1, 4)]
    f_f1s = [fold_results[f"fold_{k}"]["metrics"]["per_class"][3]["f1"] for k in range(1, 4)]
    scores = [fold_results[f"fold_{k}"]["best_score"] for k in range(1, 4)]

    aggregate_summary = {
        "mean_active_accuracy": float(np.mean(accs)), "std_active_accuracy": float(np.std(accs)),
        "mean_active_macro_f1": float(np.mean(macro_f1s)), "std_active_macro_f1": float(np.std(macro_f1s)),
        "mean_minority_macro_f1": float(np.mean(minority_f1s)), "std_minority_macro_f1": float(np.std(minority_f1s)),
        "mean_n_recall": float(np.mean(n_recs)), "std_n_recall": float(np.std(n_recs)),
        "mean_n_f1": float(np.mean(n_f1s)), "std_n_f1": float(np.std(n_f1s)),
        "mean_sveb_f1": float(np.mean(sveb_f1s)), "std_sveb_f1": float(np.std(sveb_f1s)),
        "mean_veb_f1": float(np.mean(veb_f1s)), "std_veb_f1": float(np.std(veb_f1s)),
        "mean_f_f1": float(np.mean(f_f1s)), "std_f_f1": float(np.std(f_f1s)),
        "mean_fold_score": float(np.mean(scores)), "std_fold_score": float(np.std(scores)),
        "elapsed_sec": elapsed_time,
        "mem_mb": mem_mb
    }

    print("\n=================================================================")
    print("        AGGREGATE 3-FOLD PATIENT-LEVEL CV RESULTS REPORT        ")
    print("=================================================================")
    print(f"[*] Mean Fold Score: {aggregate_summary['mean_fold_score']:.4f} ± {aggregate_summary['std_fold_score']:.4f}")
    print(f"[*] Mean Active Accuracy: {aggregate_summary['mean_active_accuracy']:.2f}% ± {aggregate_summary['std_active_accuracy']:.2f}%")
    print(f"[*] Mean Active Macro F1: {aggregate_summary['mean_active_macro_f1']:.2f}% ± {aggregate_summary['std_active_macro_f1']:.2f}%")
    print(f"[*] Mean Minority Macro F1: {aggregate_summary['mean_minority_macro_f1']:.2f}% ± {aggregate_summary['std_minority_macro_f1']:.2f}%")
    print(f"[*] Mean Normal (N) Recall: {aggregate_summary['mean_n_recall']:.2f}% ± {aggregate_summary['std_n_recall']:.2f}%")
    print(f"[*] Mean Normal (N) F1: {aggregate_summary['mean_n_f1']:.2f}% ± {aggregate_summary['std_n_f1']:.2f}%")
    print(f"[*] Mean SVEB/A F1: {aggregate_summary['mean_sveb_f1']:.2f}% ± {aggregate_summary['std_sveb_f1']:.2f}%")
    print(f"[*] Mean VEB/PVC F1: {aggregate_summary['mean_veb_f1']:.2f}% ± {aggregate_summary['std_veb_f1']:.2f}%")
    print(f"[*] Mean F/VT F1: {aggregate_summary['mean_f_f1']:.2f}% ± {aggregate_summary['std_f_f1']:.2f}%")
    print("=================================================================\n")

    # Save full results JSON
    save_data = {
        "experiment": "Fixed-Hyperparameter 3-Fold Patient-Level CV (alpha=0.7320, lr=0.001184, wd=7.114476e-4, bs=128)",
        "hyperparameters": FIXED_HP,
        "aggregate_summary": aggregate_summary,
        "fold_results": fold_results
    }

    results_json = project_root / "experiments/results_ecg_fixed_alpha0732_3fold.json"
    with open(results_json, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"[✓] Successfully saved fixed evaluation results JSON to '{results_json}'")
    return save_data


if __name__ == "__main__":
    run_fixed_alpha0732_3fold_experiment()
