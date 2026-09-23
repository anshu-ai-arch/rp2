# Forensic Read-Only Audit Report: Original ECG Dataset Integrity

**Date of Audit:** September 22, 2026  
**Audit Scope:** Forensic Read-Only Validation of MIT-BIH ECG Dataset, File Inventory, Signal Readability, AAMI Split Protocol, Beat Class Breakdown, Preprocessing Configuration, and File Integrity.  
**Auditor:** Antigravity AI Forensic Auditor  

---

## 1. Executive Summary & Final Verdict

> [!IMPORTANT]
> ### FINAL VERDICT: **PASS**
> 
> The forensic dataset audit confirms that the **original MIT-BIH Arrhythmia Dataset (`mitdb`)**, its **44 AAMI record files**, **annotations**, **signal channels**, **AAMI inter-patient DS1/DS2 splits**, **preprocessing parameters**, and **protected original scripts/checkpoints** are 100% intact, readable, uncorrupted, and completely consistent with the original ECG experiment.
> 
> **It is SAFE TO START FOCAL-LOSS TRAINING.**

---

## 2. Dataset Location & File Inventory

- **Canonical Dataset Location:** [data/raw](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/data/raw)
- **Directory Accessibility:** Confirmed readable with full read permissions (`os.R_OK = True`).
- **Record Inventory:**
  - **DS1 Training Records Expected:** 22 records (`101`, `106`, `108`, `109`, `112`, `114`, `115`, `116`, `118`, `119`, `122`, `124`, `201`, `203`, `205`, `207`, `208`, `209`, `215`, `220`, `223`, `230`) $\rightarrow$ **0 Missing**
  - **DS2 Testing Records Expected:** 22 records (`100`, `103`, `105`, `111`, `113`, `117`, `121`, `123`, `200`, `202`, `210`, `212`, `213`, `214`, `219`, `221`, `222`, `228`, `231`, `232`, `233`, `234`) $\rightarrow$ **0 Missing**
- **File Completeness:** All 44 records possess valid `.dat` (raw signal), `.hea` (header info), and `.atr` (beat annotations) files (132 total files checked).
- **Corrupted / Zero-Byte Files:** **0 files**.

---

## 3. Signal & Annotation Readability Verification

Read-only inspection of representative records via `wfdb.rdrecord` and `wfdb.rdann`:

| Record ID | Split Assignment | Signal Dimensions | Sampling Frequency ($f_s$) | Annotation Beat Count | Read Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **101** | DS1 (Train) | `(650,000, 2)` | 360 Hz | 1,874 beats | **SUCCESS** |
| **106** | DS1 (Train) | `(650,000, 2)` | 360 Hz | 2,098 beats | **SUCCESS** |
| **208** | DS1 (Train) | `(650,000, 2)` | 360 Hz | 3,040 beats | **SUCCESS** |
| **100** | DS2 (Test) | `(650,000, 2)` | 360 Hz | 2,274 beats | **SUCCESS** |
| **105** | DS2 (Test) | `(650,000, 2)` | 360 Hz | 2,691 beats | **SUCCESS** |
| **200** | DS2 (Test) | `(650,000, 2)` | 360 Hz | 2,792 beats | **SUCCESS** |

---

## 4. AAMI Split Integrity (DS1 vs DS2)

- **DS1 (Train/Val Set):** 22 records
- **DS2 (Unseen Test Set):** 22 records
- **Intersection / Overlap:** **0 records** (Strictly disjoint AAMI inter-patient split).
- **Data Leakage Risk:** **Zero**.

---

## 5. Ground-Truth Beat Class Distribution (5 AAMI Classes)

All 101,217 annotated heartbeats across the 44 records were parsed and mapped to the standard 5 AAMI classes:

| Class ID & Name | AAMI Beat Symbols | DS1 Train Beats | DS2 Test Beats | Total Database Beats | Class Proportion |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Class 0: Normal (N)** | `N`, `L`, `R`, `e`, `j` | 45,866 | 44,259 | 90,125 | 89.04% |
| **Class 1: Supraventricular (SVEB/A)** | `A`, `a`, `J`, `S` | 944 | 1,837 | 2,781 | 2.75% |
| **Class 2: Ventricular Ectopic (VEB/PVC)** | `V`, `E`, `r` | 3,788 | 3,221 | 7,009 | 6.92% |
| **Class 3: Fusion (F/VT)** | `F`, `[`, `]`, `!` | 899 | 388 | 1,287 | 1.27% |
| **Class 4: Unknown / Paced (Q)** | `/`, `f`, `Q`, `?` | 8 | 7 | 15 | 0.01% |
| **TOTAL MAPPED BEATS** | **All 5 AAMI Categories** | **51,505** | **49,712** | **101,217** | **100.00%** |

---

## 6. Preprocessing Configuration Integrity

Confirmed parameters in [config.yaml](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/config.yaml) and [src/data/preprocessor.py](file:///Users/anshutiwari123/Library/Mobile%20Documents/com~apple~CloudDocs/resercharrhythmia/src/data/preprocessor.py):

- **Target Signal Channel:** Lead II / MLII (Primary channel 0)
- **Raw Sampling Frequency:** 360 Hz
- **Resampled Target Frequency:** 128 Hz
- **Bandpass Filter:** 0.5 Hz – 50.0 Hz (4th-order Butterworth filter)
- **Beat Window Size:** 256 samples (R-peak centered: 128 samples pre-R, 128 samples post-R)
- **Beat Normalization:** Per-beat Z-score normalization ($\mu=0, \sigma=1$)

---

## 7. Protected Originals & Audit Safety

- **Original Checkpoint Hash (`model_c_medium_augmented_best.pth`):** `4b39b3b070fc763a7ff8a7ff68b0968e045033463ea50042f77f4d21d627b499` (**INTACT**)
- **Original Training Script Hash (`run_model_c_medium.py`):** `7a5f5cbf5e27cb4c35787a1090ff5c5e07172c0f520962edaeb98d7ac9aa5508` (**INTACT**)
- **Focal Loss Script (`run_model_c_medium_focal_loss.py`):** **UNTOUCHED**
- **Dataset Files Modified:** **0 files** (Read-only audit).

---

## 8. Final Verdict

### **VERDICT: PASS**

**Is it SAFE TO START FOCAL-LOSS TRAINING? YES.**
