import os
import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

from src.data.preprocessor import ECGPreprocessor
from src.data.beat_extractor import ECGBeatExtractor


from src.utils.device import get_device


def get_m1_device() -> torch.device:
    return get_device()


@torch.no_grad()
def evaluate_model_patient_wise(
    model: torch.nn.Module,
    config_path: str = "config.yaml"
) -> Tuple[Dict, Dict]:
    """
    Evaluates a PyTorch model on DS2 test set record-by-record (patient-by-patient).
    Computes global DS2 metrics as well as patient-wise accuracy statistics.
    """
    device = get_m1_device()
    model.to(device)
    model.eval()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data"]["data_dir"])
    ds2_recs = config["split"]["ds2_test_recordings"]

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

    patient_results = {}
    all_preds = []
    all_targets = []

    for rec_id in ds2_recs:
        record_path = data_dir / rec_id
        if not record_path.with_suffix(".dat").exists() or not record_path.with_suffix(".atr").exists():
            continue

        import wfdb
        record = wfdb.rdrecord(str(record_path))
        ecg_signal = record.p_signal[:, 0]

        if config["data"].get("use_filtering", True):
            processed_signal = preprocessor.bandpass_filter(ecg_signal)
        else:
            processed_signal = ecg_signal

        resampled_signal = preprocessor.resample_signal(processed_signal)

        b_1d, b_2d, b_rr, b_y = beat_extractor.extract_beats_from_record(
            record_path, resampled_signal, raw_fs=config["data"]["raw_sampling_rate"], target_fs=config["data"]["target_sampling_rate"]
        )

        if len(b_y) == 0:
            continue

        inputs = torch.tensor(b_1d, dtype=torch.float32).unsqueeze(1).to(device)
        targets = torch.tensor(b_y, dtype=torch.long)

        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu()

        rec_acc = accuracy_score(targets.numpy(), preds.numpy()) * 100.0
        patient_results[rec_id] = {
            "accuracy": rec_acc,
            "count": len(targets),
            "y_true": targets.numpy(),
            "y_pred": preds.numpy()
        }

        all_preds.extend(preds.numpy())
        all_targets.extend(targets.numpy())

    y_true_all = np.array(all_targets)
    y_pred_all = np.array(all_preds)

    global_acc = accuracy_score(y_true_all, y_pred_all) * 100.0
    global_macro_f1 = f1_score(y_true_all, y_pred_all, average="macro") * 100.0
    global_weighted_f1 = f1_score(y_true_all, y_pred_all, average="weighted") * 100.0

    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=list(range(5)), zero_division=0
    )

    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(5)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    patient_accs = [p["accuracy"] for p in patient_results.values()]
    patient_stats = {
        "mean_acc": np.mean(patient_accs),
        "std_acc": np.std(patient_accs),
        "min_acc": np.min(patient_accs),
        "max_acc": np.max(patient_accs),
        "patient_breakdown": {k: v["accuracy"] for k, v in patient_results.items()}
    }

    global_summary = {
        "accuracy": global_acc,
        "weighted_f1": global_weighted_f1,
        "macro_f1": global_macro_f1,
        "precision": precision_per_class,
        "recall": recall_per_class,
        "f1": f1_per_class,
        "support": support_per_class,
        "confusion_matrix": cm,
        "confusion_matrix_norm": cm_norm,
        "patient_stats": patient_stats
    }

    return global_summary, patient_results
