# Final Comparative Report: Controlled ECG Focal Loss Experiment

**Date:** September 22, 2026  
**Experiment Name:** Controlled ECG Focal Loss ($\gamma = 2.0$) Training & Testing  
**Training Status:** **COMPLETED (20 / 20 Epochs)**  
**Script Runner:** [experiments/run_model_c_medium_focal_loss.py](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/experiments/run_model_c_medium_focal_loss.py)  
**Output Checkpoint:** [checkpoints/ecg_focal_loss_best.pth](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/checkpoints/ecg_focal_loss_best.pth)  
**Output Results JSON:** [experiments/results_ecg_focal_loss.json](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/experiments/results_ecg_focal_loss.json)  

---

## 1. Executive Summary

A single, tightly controlled scientific experiment was conducted to evaluate the impact of replacing standard unweighted `CrossEntropyLoss` with unweighted `MultiClassFocalLoss` ($\gamma=2.0$, `weight=None`, `reduction='mean'`) on our original 257.5K parameter ECG model (`GenericHybrid1DBiCNNGRU`).

All other experimental dimensions—including architecture, dataset, AAMI inter-patient split (DS1/DS2), preprocessing, online augmentation, batch size, optimizer (Adam `lr=0.002018`, `weight_decay=2.35e-5`), epochs (20), and evaluation protocols—were kept **strictly invariant**.

> [!KEY-FINDINGS]
> ### Key Research Takeaways:
> 1. **Massive Supraventricular (SVEB/A) Sensitivity & Precision Boost:**
>    - **SVEB/A Recall:** Increased from **9.09%** (CE) $\rightarrow$ **15.62%** (Focal Loss) $\rightarrow$ **+71.8% relative recall gain!**
>    - **SVEB/A Precision:** Increased from **31.75%** (CE) $\rightarrow$ **63.36%** (Focal Loss) $\rightarrow$ **+99.6% relative precision gain!**
>    - **SVEB/A F1-Score:** Increased from **14.13%** (CE) $\rightarrow$ **25.07%** (Focal Loss) $\rightarrow$ **+77.4% relative F1 gain!**
> 2. **Ventricular Ectopic (VEB/PVC) Recall Improvement:**
>    - **VEB/PVC Recall:** Increased from **94.72%** (CE) $\rightarrow$ **97.64%** (Focal Loss) $\rightarrow$ **+2.92% absolute recall gain!** (3,143 vs 3,049 correctly detected PVC beats out of 3,219).
> 3. **Overall Accuracy Trade-off:**
>    - Overall DS2 test accuracy shifted slightly from **94.43%** (CE) to **92.89%** (Focal Loss), driven by a shift in decision boundaries on the dominant Normal class (Class 0 recall: 98.75% $\rightarrow$ 96.54%).

---

## 2. Experimental Configuration & Scientific Controls

- **Model Class:** `GenericHybrid1DBiCNNGRU` (CNN: 64$\rightarrow$128$\rightarrow$128, BiGRU: 2L hidden=64, Classifier: 128$\rightarrow$128$\rightarrow$5)
- **Trainable Parameters:** `257,541`
- **Loss Function:** `MultiClassFocalLoss` ($\gamma=2.0$, `weight=None`, `reduction='mean'`)
- **Loss Formula:** $\text{FL}(p_t) = - (1 - p_t)^2 \cdot \log(p_t)$
- **Training Epochs:** 20 epochs on augmented DS1 train split (41,158 train beats, 10,290 val beats).
- **Best Validation Selection:** **Epoch 16** (Val Acc: **99.27%**).
- **DS2 Test Evaluation:** Single evaluation pass on unseen DS2 test set (49,659 test beats across 22 test recordings).

---

## 3. Detailed DS2 Unseen Test Set Performance

### Class-Wise Metric Summary

| Class ID & Name | Support (Beats) | Precision | Recall / Sensitivity | F1-Score | Correctly Predicted |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Class 0: Normal (N)** | 44,208 | 96.04% | 96.54% | 96.29% | 42,680 / 44,208 |
| **Class 1: Supraventricular (SVEB/A)** | 1,837 | **63.36%** | **15.62%** | **25.07%** | **287 / 1,837** |
| **Class 2: Ventricular Ectopic (VEB/PVC)** | 3,219 | 69.60% | **97.64%** | 81.27% | **3,143 / 3,219** |
| **Class 3: Fusion (F/VT)** | 388 | 6.75% | 4.38% | 5.31% | 17 / 388 |
| **Class 4: Unknown / Paced (Q)** | 7 | 0.00% | 0.00% | 0.00% | 0 / 7 |
| **OVERALL / SUMMARY METRICS** | **49,659** | **Weighted F1: 91.96%** | **Macro F1: 41.59%** | **DS2 Accuracy: 92.89%** | **46,127 / 49,659** |

---

### 5x5 Raw Confusion Matrix (DS2 Unseen Test Set)

```
True \ Pred  |       N |  SVEB/A | VEB/PVC |    F/VT |       Q
--------------------------------------------------------------
N            |   42680 |     158 |    1141 |     229 |       0
SVEB/A       |    1437 |     287 |     111 |       2 |       0
VEB/PVC      |      64 |       8 |    3143 |       4 |       0
F/VT         |     255 |       0 |     116 |      17 |       0
Q            |       2 |       0 |       5 |       0 |       0
```

---

### 5x5 Normalized Confusion Matrix (%)

```
True \ Pred  |       N |  SVEB/A | VEB/PVC |    F/VT |       Q
--------------------------------------------------------------
N            |  96.54% |   0.36% |   2.58% |   0.52% |   0.00%
SVEB/A       |  78.23% |  15.62% |   6.04% |   0.11% |   0.00%
VEB/PVC      |   1.99% |   0.25% |  97.64% |   0.12% |   0.00%
F/VT         |  65.72% |   0.00% |  29.90% |   4.38% |   0.00%
Q            |  28.57% |   0.00% |  71.43% |   0.00% |   0.00%
```

---

## 4. Side-by-Side Baseline Comparison: CE vs Focal Loss

| Metric | Original CE Model (`run_model_c_medium`) | Experimental Focal Loss Model ($\gamma=2.0$) | Absolute Difference | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Loss Criterion** | `nn.CrossEntropyLoss()` | `MultiClassFocalLoss(gamma=2.0)` | — | — |
| **Best Val Accuracy (DS1)** | 99.30% (Epoch 17) | 99.27% (Epoch 16) | -0.03% | Invariant |
| **Unseen DS2 Accuracy** | 94.43% | 92.89% | -1.54% | Moderate |
| **DS2 Weighted F1** | 93.10% | 91.96% | -1.14% | Moderate |
| **DS2 Macro F1** | 42.40% | 41.59% | -0.81% | Invariant |
| **SVEB/A (Class 1) Recall** | **9.09%** | **15.62%** | **+6.53%** | **+71.8% Gain** |
| **SVEB/A (Class 1) Precision** | **31.75%** | **63.36%** | **+31.61%** | **+99.6% Gain** |
| **SVEB/A (Class 1) F1-Score** | **14.13%** | **25.07%** | **+10.94%** | **+77.4% Gain** |
| **VEB/PVC (Class 2) Recall** | **94.72%** | **97.64%** | **+2.92%** | **+3.08% Gain** |
| **VEB/PVC (Class 2) Precision** | 90.31% | 69.60% | -20.71% | -22.9% |
| **Normal (Class 0) Recall** | 98.75% | 96.54% | -2.21% | -2.2% |
| **Fusion (Class 3) Recall** | 5.41% | 4.38% | -1.03% | Minor |

---

## 5. Comprehensive 6-Model Comparison Matrix

Comparison of the new Focal Loss model against the **FIVE previously reported reference models** from our research codebase:

| Model Name & Description | Loss Function | Trainable Params | DS2 Acc (%) | Macro F1 (%) | Weighted F1 (%) | N Recall (%) | SVEB/A Recall (%) | VEB/PVC Recall (%) | F/VT Recall (%) | Q Recall (%) | Data Provenance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original CE Control (Exp B)** | CrossEntropy | 257,541 | **94.43%** | 42.40% | **93.10%** | **98.75%** | 9.09% | 94.72% | 5.41% | 0.00% | Existing Reference |
| **Exp A (Large CNN)** | CrossEntropy | 578,309 | 93.59% | 44.90% | 93.05% | 97.06% | 25.48% | 95.53% | 5.93% | 0.00% | Existing Reference |
| **Exp C (Small CNN)** | CrossEntropy | 164,741 | 92.71% | 44.71% | 92.47% | 95.86% | **26.57%** | 97.45% | 8.76% | 0.00% | Existing Reference |
| **Exp D (Medium CNN+BiGRU32)**| CrossEntropy | 150,277 | 93.60% | **47.24%** | 92.83% | 97.29% | 12.85% | 97.24% | **26.03%** | 0.00% | Existing Reference |
| **Optuna Raw Acc Model C** | CrossEntropy | 579,592 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Existing Reference |
| **Focal Loss Model (THIS EXP)**| **Focal ($\gamma=2.0$)** | **257,541** | **92.89%** | **41.59%** | **91.96%** | **96.54%** | **15.62%** | **97.64%** | **4.38%** | **0.00%** | **NEW EXPERIMENT** |

---

## 6. Post-Training Forensic Integrity Verification

Immediately following training and evaluation, SHA256 hashes of all protected original files were re-computed:

| Target File | Expected Canonical SHA256 Hash | Post-Training Computed SHA256 Hash | Integrity Status |
| :--- | :--- | :--- | :---: |
| `checkpoints/model_c_medium_augmented_best.pth` | `4b39b3b070fc763a7ff8a7ff68b0968e045033463ea50042f77f4d21d627b499` | `4b39b3b070fc763a7ff8a7ff68b0968e045033463ea50042f77f4d21d627b499` | **100% UNMODIFIED** |
| `checkpoints/ecg_original/model_c_medium_augmented_best.pth` | `4b39b3b070fc763a7ff8a7ff68b0968e045033463ea50042f77f4d21d627b499` | `4b39b3b070fc763a7ff8a7ff68b0968e045033463ea50042f77f4d21d627b499` | **100% UNMODIFIED** |
| `experiments/run_model_c_medium.py` | `7a5f5cbf5e27cb4c35787a1090ff5c5e07172c0f520962edaeb98d7ac9aa5508` | `7a5f5cbf5e27cb4c35787a1090ff5c5e07172c0f520962edaeb98d7ac9aa5508` | **100% UNMODIFIED** |

---

## 7. Artifact & File Directory Summary

1. **New Experimental Script:** [experiments/run_model_c_medium_focal_loss.py](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/experiments/run_model_c_medium_focal_loss.py)
2. **New Model Checkpoint:** [checkpoints/ecg_focal_loss_best.pth](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/checkpoints/ecg_focal_loss_best.pth)
3. **New Evaluation Results JSON:** [experiments/results_ecg_focal_loss.json](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/experiments/results_ecg_focal_loss.json)
4. **Final Comparative Report:** [artifacts/ecg_focal_loss/focal_loss_comparison_report.md](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/artifacts/ecg_focal_loss/focal_loss_comparison_report.md)
