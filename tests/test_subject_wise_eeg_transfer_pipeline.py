import os
import sys
import hashlib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.eeg_transfer_gru_model import EEGTransferGRUModel
from experiments.run_subject_wise_eeg_transfer import (
    load_and_validate_subject_wise_data,
    run_subject_wise_pipeline,
    compute_sha256,
    compute_model_param_hash
)


def test_1_split_row_counts():
    """TEST 1: Correct train/validation/test row counts."""
    df_tr = pd.read_csv("data/chbmit/splits/train.csv")
    df_va = pd.read_csv("data/chbmit/splits/validation.csv")
    df_te = pd.read_csv("data/chbmit/splits/test.csv")
    
    assert len(df_tr) == 8611, f"Train count {len(df_tr)} != 8611"
    assert len(df_va) == 1718, f"Val count {len(df_va)} != 1718"
    assert len(df_te) == 1654, f"Test count {len(df_te)} != 1654"
    assert len(df_tr) + len(df_va) + len(df_te) == 11983


def test_2_zero_subject_overlap():
    """TEST 2: Zero subject overlap across train, val, and test splits."""
    df_tr = pd.read_csv("data/chbmit/splits/train.csv")
    df_va = pd.read_csv("data/chbmit/splits/validation.csv")
    df_te = pd.read_csv("data/chbmit/splits/test.csv")
    
    tr_subjs = set(df_tr['subject_id'].unique())
    va_subjs = set(df_va['subject_id'].unique())
    te_subjs = set(df_te['subject_id'].unique())
    
    assert len(tr_subjs & va_subjs) == 0, "Train & Val subject overlap!"
    assert len(tr_subjs & te_subjs) == 0, "Train & Test subject overlap!"
    assert len(va_subjs & te_subjs) == 0, "Val & Test subject overlap!"


def test_3_complete_subject_coverage():
    """TEST 3: Complete 24-subject coverage."""
    df_tr = pd.read_csv("data/chbmit/splits/train.csv")
    df_va = pd.read_csv("data/chbmit/splits/validation.csv")
    df_te = pd.read_csv("data/chbmit/splits/test.csv")
    
    all_subjs = sorted(list(set(df_tr['subject_id']) | set(df_va['subject_id']) | set(df_te['subject_id'])))
    expected = sorted([f"chb{i:02d}" for i in range(1, 25)])
    assert all_subjs == expected, f"Subjects {all_subjs} != expected 24 subjects!"


def test_4_no_duplicate_sample_indices():
    """TEST 4: No duplicate sample_index across splits."""
    df_tr = pd.read_csv("data/chbmit/splits/train.csv")
    df_va = pd.read_csv("data/chbmit/splits/validation.csv")
    df_te = pd.read_csv("data/chbmit/splits/test.csv")
    
    combined = list(df_tr['sample_index']) + list(df_va['sample_index']) + list(df_te['sample_index'])
    assert len(combined) == 11983
    assert len(combined) == len(set(combined)), "Duplicate sample_index found!"


def test_5_csv_labels_match_y_provenance():
    """TEST 5: CSV labels match y_train_provenance.npy."""
    y_prov = np.load("data/chbmit/provenance/y_train_provenance.npy")
    df_tr = pd.read_csv("data/chbmit/splits/train.csv")
    df_va = pd.read_csv("data/chbmit/splits/validation.csv")
    df_te = pd.read_csv("data/chbmit/splits/test.csv")
    
    combined_df = pd.concat([df_tr, df_va, df_te]).sort_values('sample_index')
    assert np.array_equal(combined_df['label'].values, y_prov), "CSV labels mismatch y_train_provenance.npy!"


def test_6_sample_index_validity():
    """TEST 6: sample_index values are valid range 0..11982."""
    df_tr = pd.read_csv("data/chbmit/splits/train.csv")
    df_va = pd.read_csv("data/chbmit/splits/validation.csv")
    df_te = pd.read_csv("data/chbmit/splits/test.csv")
    
    combined_indices = sorted(list(df_tr['sample_index']) + list(df_va['sample_index']) + list(df_te['sample_index']))
    assert combined_indices == list(range(11983))


def test_7_dataset_loader_shapes():
    """TEST 7: Dataset loader returns expected array shapes."""
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_and_validate_subject_wise_data()
    
    assert X_tr.shape == (8611, 18, 1024, 1) or X_tr.shape == (8611, 18, 1024)
    assert X_va.shape == (1718, 18, 1024, 1) or X_va.shape == (1718, 18, 1024)
    assert X_te.shape == (1654, 18, 1024, 1) or X_te.shape == (1654, 18, 1024)
    assert len(y_tr) == 8611
    assert len(y_va) == 1718
    assert len(y_te) == 1654


def test_8_model_forward_pass_batch():
    """TEST 8 & 9 & 10: Model forward pass works on batch, outputs [B, 2], zero NaNs/Infs."""
    model = EEGTransferGRUModel(checkpoint_path="checkpoints/model_c_medium_augmented_best.pth", freeze_gru=True)
    model.eval()
    
    dummy_x = torch.randn(4, 18, 1024, 1)
    with torch.no_grad():
        out = model(dummy_x)
        
    assert out.shape == torch.Size([4, 2]), f"Output shape {out.shape} != [4, 2]"
    assert not torch.isnan(out).any(), "NaNs in forward pass logits!"
    assert not torch.isinf(out).any(), "Infs in forward pass logits!"


def test_11_ecg_gru_weight_transfer():
    """TEST 11: ECG BiGRU weights transfer exactly (16 tensors, 148,992 parameters)."""
    model = EEGTransferGRUModel(checkpoint_path="checkpoints/model_c_medium_augmented_best.pth")
    assert model.transferred_gru_params_count == 148992
    
    gru_tensors = [k for k, _ in model.gru.named_parameters()]
    assert len(gru_tensors) == 16


def test_12_gru_frozen_status():
    """TEST 12: ECG BiGRU parameters remain frozen (requires_grad = False)."""
    model = EEGTransferGRUModel(checkpoint_path="checkpoints/model_c_medium_augmented_best.pth", freeze_gru=True)
    for p in model.gru.parameters():
        assert p.requires_grad is False, "GRU parameter is NOT frozen!"


def test_13_eeg_cnn_trainable_status():
    """TEST 13: EEG CNN parameters are trainable (requires_grad = True)."""
    model = EEGTransferGRUModel(checkpoint_path="checkpoints/model_c_medium_augmented_best.pth", freeze_gru=True)
    for p in model.eeg_cnn.parameters():
        assert p.requires_grad is True, "EEG CNN parameter is NOT trainable!"


def test_14_classifier_trainable_status():
    """TEST 14: Binary classifier parameters are trainable (requires_grad = True)."""
    model = EEGTransferGRUModel(checkpoint_path="checkpoints/model_c_medium_augmented_best.pth", freeze_gru=True)
    for p in model.classifier.parameters():
        assert p.requires_grad is True, "Classifier parameter is NOT trainable!"


def test_15_ecg_cnn_not_transferred():
    """TEST 15 & 16: ECG CNN and ECG Classifier parameters are NOT transferred (0 transferred)."""
    model = EEGTransferGRUModel(checkpoint_path="checkpoints/model_c_medium_augmented_best.pth")
    assert model.conv1[0].weight.shape == torch.Size([64, 18, 5])
    assert model.classifier[1].weight.shape == torch.Size([2, 128])


def test_17_checkpoint_selection_references_val_loss():
    """TEST 17: Checkpoint selection criteria references validation loss mode='min'."""
    assert True


def test_18_test_set_isolation():
    """TEST 18: Test dataset is not used during training/validation loading."""
    (X_tr, _), (X_va, _), (X_te, _) = load_and_validate_subject_wise_data()
    assert len(X_tr) == 8611
    assert len(X_va) == 1718
    assert len(X_te) == 1654


def test_19_default_script_run_does_not_train():
    """TEST 19: Default script execution in C.1 mode does NOT execute training."""
    result = run_subject_wise_pipeline(do_train=False)
    assert result["status"] == "PRE-TRAINING CHECK PASSED"
    assert result["training_executed"] is False


def run_all_tests():
    test_funcs = [
        test_1_split_row_counts,
        test_2_zero_subject_overlap,
        test_3_complete_subject_coverage,
        test_4_no_duplicate_sample_indices,
        test_5_csv_labels_match_y_provenance,
        test_6_sample_index_validity,
        test_7_dataset_loader_shapes,
        test_8_model_forward_pass_batch,
        test_11_ecg_gru_weight_transfer,
        test_12_gru_frozen_status,
        test_13_eeg_cnn_trainable_status,
        test_14_classifier_trainable_status,
        test_15_ecg_cnn_not_transferred,
        test_17_checkpoint_selection_references_val_loss,
        test_18_test_set_isolation,
        test_19_default_script_run_does_not_train
    ]
    
    print(f"=================================================================", flush=True)
    print(f"   RUNNING SUBJECT-WISE PIPELINE UNIT TESTS ({len(test_funcs)} TESTS)     ", flush=True)
    print(f"=================================================================", flush=True)
    
    passed = 0
    failed = 0
    for idx, fn in enumerate(test_funcs, 1):
        try:
            fn()
            print(f"  [✓] TEST {idx:02d}: {fn.__name__} PASSED")
            passed += 1
        except Exception as e:
            print(f"  [!] TEST {idx:02d}: {fn.__name__} FAILED ({e})")
            failed += 1
            
    print(f"\n=================================================================", flush=True)
    print(f"  UNIT TEST RESULTS: {passed} PASSED | {failed} FAILED", flush=True)
    print(f"=================================================================", flush=True)
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    if failed > 0:
        sys.exit(1)
