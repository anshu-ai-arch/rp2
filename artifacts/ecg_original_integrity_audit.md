# Forensic Read-Only Audit Report: Original ECG Research Model Integrity

**Date of Audit:** September 22, 2026  
**Audit Scope:** Read-Only Integrity Verification of Original ECG 1D-CNN-BiGRU Architecture, Checkpoints, Hashes, EEG Transfer Isolation, and Git Protection.  
**Auditor:** Antigravity AI Forensic Auditor  

---

## 1. Executive Summary & Final Verdict

> [!IMPORTANT]
> ### FINAL VERDICT: **PASS**
> 
> The forensic audit confirms that the **original ECG 1D-CNN-BiGRU model**, its **original weights**, **checkpoint files**, **backup copies**, **SHA256 hashes**, and **Git protection tags** are 100% intact, byte-identical, verified, and completely isolated from subsequent EEG experiments.
> 
> **It is 100% SAFE to proceed to controlled ECG focal-loss experimentation.**

---

## 2. Identify the Original ECG Model Architecture

The original ECG model implementation was located and verified in:
- **Primary Source File:** [experiments/count_parameters_compression_models.py](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/experiments/count_parameters_compression_models.py#L6-L86) (`GenericHybrid1DBiCNNGRU`)
- **Training Source File:** [experiments/run_model_c_medium.py](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/experiments/run_model_c_medium.py#L39-L47) (`Experiment B — Medium CNN`)
- **Transfer Backbone Source File:** [src/models/transfer_ecg_model.py](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/src/models/transfer_ecg_model.py#L6-L101) (`ECGTransferModel`)

### Verified Architecture Specifications

- **Input Tensor Shape:** `[Batch, 1, 256]` (1D continuous ECG beat waveform)
- **Stage 1 CNN:** `nn.Conv1d(1, 64, kernel_size=5, padding=2)` $\rightarrow$ `nn.BatchNorm1d(64)` $\rightarrow$ `nn.ReLU()` $\rightarrow$ `nn.MaxPool1d(kernel_size=2, stride=2)` $\rightarrow$ Output `[B, 64, 128]`
- **Stage 2 CNN:** `nn.Conv1d(64, 128, kernel_size=5, padding=2)` $\rightarrow$ `nn.BatchNorm1d(128)` $\rightarrow$ `nn.ReLU()` $\rightarrow$ `nn.MaxPool1d(kernel_size=2, stride=2)` $\rightarrow$ Output `[B, 128, 64]`
- **Stage 3 CNN:** `nn.Conv1d(128, 128, kernel_size=3, padding=1)` $\rightarrow$ `nn.BatchNorm1d(128)` $\rightarrow$ `nn.ReLU()` $\rightarrow$ `nn.MaxPool1d(kernel_size=2, stride=2)` $\rightarrow$ Output `[B, 128, 32]`
- **Permutation:** Permute to sequence format `[B, 32, 128]` for GRU input.
- **BiGRU Encoder:** `nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)` $\rightarrow$ Output `[B, 32, 128]`
- **Temporal Pooling:** Global temporal mean pooling along sequence dimension $T=32$ $\rightarrow$ Output `[B, 128]`
- **Classifier Head:** `nn.Linear(128, 128)` $\rightarrow$ `nn.ReLU()` $\rightarrow$ `nn.Dropout(p=0.2076)` $\rightarrow$ `nn.Linear(128, 5)` $\rightarrow$ Output `[B, 5]` (RAW LOGITS)

---

## 3. Parameter Count & Tensor Shape Verification

### Exact Parameter Breakdown

| Component | Modules / Layers | Weight & Bias Shapes | Parameter Count | Verification Status |
| :--- | :--- | :--- | :---: | :---: |
| **CNN Stage 1** | `conv1.0` (Conv1d 1$\rightarrow$64, k=5)<br>`conv1.1` (BatchNorm1d) | Weight `[64, 1, 5]`, Bias `[64]`<br>Weight `[64]`, Bias `[64]` | 384<br>128 | VERIFIED |
| **CNN Stage 2** | `conv2.0` (Conv1d 64$\rightarrow$128, k=5)<br>`conv2.1` (BatchNorm1d) | Weight `[128, 64, 5]`, Bias `[128]`<br>Weight `[128]`, Bias `[128]` | 41,088<br>256 | VERIFIED |
| **CNN Stage 3** | `conv3.0` (Conv1d 128$\rightarrow$128, k=3)<br>`conv3.1` (BatchNorm1d) | Weight `[128, 128, 3]`, Bias `[128]`<br>Weight `[128]`, Bias `[128]` | 49,280<br>256 | VERIFIED |
| **CNN Total** | **3-Stage 1D CNN Backbone** | **Subtotal** | **91,392** | **VERIFIED** |
| **BiGRU Layer 0** | `gru.weight_ih_l0`, `weight_hh_l0`<br>`gru.bias_ih_l0`, `bias_hh_l0`<br>`gru.weight_ih_l0_reverse`, ... | `[192, 128]`, `[192, 64]`<br>`[192]`, `[192]`<br>`[192, 128]`, `[192, 64]`, `[192]`, `[192]` | 74,496 | VERIFIED |
| **BiGRU Layer 1** | `gru.weight_ih_l1`, `weight_hh_l1`<br>`gru.bias_ih_l1`, `bias_hh_l1`<br>`gru.weight_ih_l1_reverse`, ... | `[192, 128]`, `[192, 64]`<br>`[192]`, `[192]`<br>`[192, 128]`, `[192, 64]`, `[192]`, `[192]` | 74,496 | VERIFIED |
| **BiGRU Total** | **2-Layer Bidirectional GRU** | **Subtotal (16 Tensors)** | **148,992** | **VERIFIED** |
| **Classifier** | `classifier.0` (Linear 128$\rightarrow$128)<br>`classifier.3` (Linear 128$\rightarrow$5) | Weight `[128, 128]`, Bias `[128]`<br>Weight `[5, 128]`, Bias `[5]` | 16,512<br>645 | VERIFIED |
| **Classifier Total**| **Dense Classifier Head** | **Subtotal** | **17,157** | **VERIFIED** |
| **TOTAL MODEL** | **Original 1D-CNN-BiGRU** | **Grand Total** | **257,541** | **VERIFIED** |

---

## 4. Comprehensive ECG Checkpoint Forensic Inventory & Hashes

Every checkpoint file across the codebase was searched, hashed with SHA256, and inspected.

### Canonical Hash Verification Table

| Relative File Path | File Size | Canonical SHA256 Hash | Current Computed SHA256 Hash | Byte Match | Contents & Metadata | Verification |
| :--- | :---: | :--- | :--- | :---: | :--- | :---: |
| `checkpoints/model_c_medium_augmented_best.pth` | 1,047,154 B | `4b39b3b070fc763a7ff8a7ff68b0968e045033463ea50042f77f4d21d627b499` | `4b39b3b070fc763a7ff8a7ff68b0968e045033463ea50042f77f4d21d627b499` | **MATCH** | Full ECG Model (257.5K params), Epoch 17, Val Acc 99.3003%, `cnn_channels=[64, 128, 128]` | **VERIFIED** |
| `checkpoints/ecg_original/model_c_medium_augmented_best.pth` | 1,047,154 B | `4b39b3b070fc763a7ff8a7ff68b0968e045033463ea50042f77f4d21d627b499` | `4b39b3b070fc763a7ff8a7ff68b0968e045033463ea50042f77f4d21d627b499` | **MATCH** | Full ECG Model Backup (257.5K params), Epoch 17, Val Acc 99.3003% | **VERIFIED** |
| `checkpoints/backbone_1d_cnn_pretrained.pth` | 376,037 B | `2a6431e92ae29498cd5bd1f61e7ace48d72a0d7469b3f9b708bc3f278b49fdec` | `2a6431e92ae29498cd5bd1f61e7ace48d72a0d7469b3f9b708bc3f278b49fdec` | **MATCH** | CNN Backbone Only (91,392 params), `cnn_channels=[64, 128, 128]` | **VERIFIED** |
| `checkpoints/ecg_original/backbone_1d_cnn_pretrained.pth` | 376,037 B | `2a6431e92ae29498cd5bd1f61e7ace48d72a0d7469b3f9b708bc3f278b49fdec` | `2a6431e92ae29498cd5bd1f61e7ace48d72a0d7469b3f9b708bc3f278b49fdec` | **MATCH** | CNN Backbone Backup (91,392 params), `cnn_channels=[64, 128, 128]` | **VERIFIED** |

> [!NOTE]
> All original ECG checkpoint files match their expected canonical SHA256 hashes **byte-for-byte**. No file corruption, repair, modification, or overwrite has occurred.

---

## 5. Full Model Checkpoint Load & Forward Pass Verification

`checkpoints/ecg_original/model_c_medium_augmented_best.pth` was instantiated and loaded into `GenericHybrid1DBiCNNGRU` with `strict=True`:

```python
model = GenericHybrid1DBiCNNGRU(
    in_channels=1,
    cnn_channels=[64, 128, 128],
    kernel_sizes=[5, 5, 3],
    gru_hidden_size=64,
    gru_num_layers=2,
    dropout=0.2076,
    num_classes=5
)
ckpt = torch.load("checkpoints/ecg_original/model_c_medium_augmented_best.pth", map_location="cpu")
load_result = model.load_state_dict(ckpt["model_state_dict"], strict=True)
```

### Results
- **Missing Keys:** `0` (None)
- **Unexpected Keys:** `0` (None)
- **Strict Load Status:** `SUCCESS`
- **Dummy Input Shape:** `[2, 1, 256]`
- **Forward Pass Output Shape:** `[2, 5]`
- **Output Properties:** Raw unnormalized logits (range min `-5.4120`, max `+8.9015`), confirming **no final Softmax** layer is embedded in the forward pass.

---

## 6. EEG Experiment Separation & Isolation Audit

Inspection of [src/models/eeg_transfer_gru_model.py](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/src/models/eeg_transfer_gru_model.py#L77-L125) and [experiments/run_subject_wise_eeg_transfer.py](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/experiments/run_subject_wise_eeg_transfer.py) confirmed:

1. **Read-Only Weight Extraction:** The EEG transfer code opened `checkpoints/ecg_original/model_c_medium_augmented_best.pth` using `torch.load()` in read-only mode, filtering specifically for keys beginning with `gru.`.
2. **Transferred Parameters:**
   - ECG CNN parameters transferred: **0 parameters**
   - ECG Classifier parameters transferred: **0 parameters**
   - ECG BiGRU parameters transferred: **16 tensors / 148,992 parameters**
3. **Gradient Locking:** The transferred GRU parameters had `requires_grad = False` set during EEG training (`freeze_gru = True`).
4. **Checkpoint Preservation:** The output EEG models were saved to independent paths (`checkpoints/eeg_transfer_gru_subject_wise_best.pth`). The original ECG checkpoint was **never modified**.

---

## 7. Git Protection & Repository Status Audit

Git repository inspection results:

- **Current Branch:** `main`
- **Current HEAD Commit:** `242b6796bc71cd654c68cf2e78059c2e9965ea54`
- **Protected Tag:** `ecg-original-preserved-before-eeg-training` $\rightarrow$ Points directly to `242b6796bc71cd654c68cf2e78059c2e9965ea54`
- **Git Blob Hash of `model_c_medium_augmented_best.pth` in commit 242b679:** `109a0f36eefba01d9fc1abf7e61cd21cdae99504`
- **Current Workspace File Hash:** Identical to Git object store.
- **Git Status:** Workspace contains uncommitted untracked files (`artifacts/`, EEG results) but **ZERO uncommitted modifications to original ECG source files or checkpoints**.

---

## 8. Original ECG Training Setup & Configuration Provenance

Extracted from [config.yaml](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/config.yaml), [experiments/run_model_c_medium.py](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/experiments/run_model_c_medium.py), and `train_augmented_model_c.py`:

- **Dataset:** MIT-BIH Arrhythmia Database (`mitdb`), 44 recordings.
- **Protocol:** Standard AAMI Inter-Patient Split (DS1 for training/validation, DS2 for unseen testing).
  - **DS1 Records (22):** `101`, `106`, `108`, `109`, `112`, `114`, `115`, `116`, `118`, `119`, `122`, `124`, `201`, `203`, `205`, `207`, `208`, `209`, `215`, `220`, `223`, `230`.
  - **DS2 Records (22):** `100`, `103`, `105`, `111`, `113`, `117`, `121`, `123`, `200`, `202`, `210`, `212`, `213`, `214`, `219`, `221`, `222`, `228`, `231`, `232`, `233`, `234`.
- **Input Feature Vector:** `[B, 1, 256]` (Bandpass filtered 0.5–50.0 Hz, 128 Hz resampling, 256 sample window).
- **Target Classes (5 AAMI):**
  - `0`: Normal Beat (N)
  - `1`: Supraventricular Ectopic Beat / A-Fib (SVEB/A)
  - `2`: Ventricular Ectopic Beat / PVC (VEB/PVC)
  - `3`: Fusion Beat / Ventricular Tachycardia (F/VT)
  - `4`: Unknown / Paced / Other (Q)
- **Loss Function:** `nn.CrossEntropyLoss()` (Standard unweighted CE for original model_c_medium).
- **Optimizer:** Adam (`lr = 0.002018`, `weight_decay = 2.35e-5`).
- **Batch Size:** 256
- **Epochs:** 20 epochs trained, best epoch saved at **Epoch 17** (Val Acc: 99.3003%).
- **Data Augmentation:** Online ECG waveform scaling `[0.85, 1.15]`, Gaussian noise `N(0, 0.02)`, random shift `[-5, +5]` samples.

---

## 9. Checkpoint Distinction & Separation Matrix

| Category | Checkpoint Path | Architecture Summary | Primary Domain | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **A. Original ECG Checkpoint** | `checkpoints/model_c_medium_augmented_best.pth`<br>`checkpoints/ecg_original/model_c_medium_augmented_best.pth` | 1D-CNN (64$\rightarrow$128$\rightarrow$128) + 2L-BiGRU(64) + 5-class Classifier (257,541 params) | ECG (MIT-BIH) | Canonical baseline model for arrhythmia heartbeat classification. |
| **B. ECG CNN Backbone** | `checkpoints/backbone_1d_cnn_pretrained.pth`<br>`checkpoints/ecg_original/backbone_1d_cnn_pretrained.pth` | 3-Stage 1D CNN feature extractor only (91,392 params) | Pretrained Feature Extractor | Portable CNN backbone for transfer learning. |
| **C. EEG Transfer Model** | `checkpoints/eeg_transfer_gru_subject_wise_best.pth`<br>`checkpoints/eeg_transfer_gru_frozen_best.pth` | 18-channel EEG CNN + Transferred Frozen ECG BiGRU + 2-class Head | EEG (CHB-MIT) | Seizure detection model leveraging pretrained ECG temporal representations. |

---

## 10. Answers to Specific Verdict Questions

1. **Is my original ECG 1D-CNN-BiGRU architecture intact?**  
   $\rightarrow$ **YES.** Verified in `experiments/count_parameters_compression_models.py` (`GenericHybrid1DBiCNNGRU`). Exactly 257,541 parameters.
2. **Is the original full ECG checkpoint intact?**  
   $\rightarrow$ **YES.** Saved at `checkpoints/model_c_medium_augmented_best.pth` and `checkpoints/ecg_original/model_c_medium_augmented_best.pth`. Epoch 17, Val Acc 99.3003%.
3. **Is its SHA256 unchanged?**  
   $\rightarrow$ **YES.** Hash is `4b39b3b070fc763a7ff8a7ff68b0968e045033463ea50042f77f4d21d627b499` (100% match).
4. **Is the protected backup intact?**  
   $\rightarrow$ **YES.** `checkpoints/ecg_original/model_c_medium_augmented_best.pth` is byte-identical.
5. **Is the original backbone checkpoint intact?**  
   $\rightarrow$ **YES.** Hash is `2a6431e92ae29498cd5bd1f61e7ace48d72a0d7469b3f9b708bc3f278b49fdec` (100% match).
6. **Does the original checkpoint still load into the original architecture?**  
   $\rightarrow$ **YES.** Loaded with `strict=True` with 0 missing and 0 unexpected keys. Dummy input `[2, 1, 256]` outputs `[2, 5]` raw logits.
7. **Is the original ECG model clearly separated from the later EEG transfer model?**  
   $\rightarrow$ **YES.** Completely separate codebases, checkpoints, and parameter structures.
8. **Did the EEG experiment modify the original ECG weights?**  
   $\rightarrow$ **NO.** Read-only extraction was used. Original file hashes are unchanged.
9. **Is the Git protection still present?**  
   $\rightarrow$ **YES.** Commit `242b6796bc71cd654c68cf2e78059c2e9965ea54` and tag `ecg-original-preserved-before-eeg-training` exist and point to the protected state.
10. **Is it SAFE to proceed to a controlled ECG focal-loss experiment?**  
   $\rightarrow$ **YES.** All original state is fully verified, preserved, and protected.
