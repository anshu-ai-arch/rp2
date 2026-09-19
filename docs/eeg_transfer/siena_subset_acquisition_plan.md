# Siena Scalp EEG Database — Small-Subset Acquisition Plan (Phase 2)

> **RESEARCH CLARIFICATION & TRACEABILITY NOTICE**:
> This document defines a **Phase 2 Acquisition Plan** for a storage-safe subset of the Siena Scalp EEG Database (v1.0.0). 
> This proposed subset is an **initial acquisition boundary for transfer-learning feasibility**, NOT a final scientific inclusion/exclusion criterion. 0 bytes have been downloaded, and zero ECG files or models were modified.

---

## Section A — Facts Obtained Directly from PhysioNet Siena Metadata

Official database specifications extracted directly from PhysioNet `subject_info.csv` and `RECORDS` manifest:

- **Total Database Scope**: 14 subjects (`PN00`, `PN01`, `PN03`, `PN05`, `PN06`, `PN07`, `PN09`, `PN10`, `PN11`, `PN12`, `PN13`, `PN14`, `PN16`, `PN17`).
- **Total EDF Recordings**: 41 `.edf` recording files.
- **Total Documented Seizures**: 47 seizure events.
- **Total Recording Time**: 7,704.0 minutes (128.40 hours).
- **Sampling Frequency**: 512 Hz (29 channels for 13 subjects, 20 channels for subject `PN10`).

### Complete PhysioNet Subject Summary Table

| Subject ID | Age | Sex | Seizure Type | Channels | Duration (min) | Duration (hrs) | EDF Files | Seizure Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `PN00` | 55 | Male | IAS | 29 | 198.0 | 3.30 | 5 | 5 |
| `PN01` | 46 | Male | IAS | 29 | 809.0 | 13.48 | 1 | 2 |
| `PN03` | 54 | Male | IAS | 29 | 752.0 | 12.53 | 2 | 2 |
| `PN05` | 51 | Female | IAS | 29 | 359.0 | 5.98 | 3 | 3 |
| `PN06` | 36 | Male | IAS | 29 | 722.0 | 12.03 | 5 | 5 |
| `PN07` | 20 | Female | IAS | 29 | 523.0 | 8.72 | 1 | 1 |
| `PN09` | 27 | Female | IAS | 29 | 410.0 | 6.83 | 3 | 3 |
| `PN10` | 25 | Male | FBTC | 20 | 1002.0 | 16.70 | 6 | 10 |
| `PN11` | 58 | Female | IAS | 29 | 145.0 | 2.42 | 1 | 1 |
| `PN12` | 71 | Male | IAS | 29 | 246.0 | 4.10 | 3 | 4 |
| `PN13` | 34 | Female | IAS | 29 | 519.0 | 8.65 | 3 | 3 |
| `PN14` | 49 | Male | WIAS | 29 | 1408.0 | 23.47 | 4 | 4 |
| `PN16` | 41 | Female | IAS | 29 | 303.0 | 5.05 | 2 | 2 |
| `PN17` | 42 | Male | IAS | 29 | 308.0 | 5.13 | 2 | 2 |
| **TOTALS** | — | — | — | — | **7,704.0** | **128.40** | **41** | **47** |

---

## Section B — Storage Calculations & Local Capacity Audit

- **EDF Signal Size Formula**: 
  $\text{Size (bytes/sec)} = \text{Sampling Rate (512 Hz)} \times \text{Channels (29)} \times \text{Bytes per sample (2)} = 29,696 \text{ B/s} \approx 1.78 \text{ MB/min} \approx 0.107 \text{ GB/hour}$.
- **Full Siena Database Total Size**: **~13.39 GB** across all 14 subjects (41 EDF files).
- **Available Local Mac Free Space**: **36.98 GB** free space.
- **Feasibility Assessment**: The entire Siena database (~13.39 GB) fits within local disk capacity while leaving > 23 GB free space.

---

## Section C — Research-Design Trade-Off Analysis & Candidate Subsets

To guarantee maximum safety and local disk efficiency, three patient-based candidate acquisition subsets were evaluated:

| Candidate Option | Target GB | Subjects Included | Patient Count | EDF Files | Seizure Events | Duration (hrs) | Estimated Size (GB) | Seizure Coverage (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate 1 (~5 GB)** | 5.0 GB | `PN00, PN05, PN09, PN10, PN12, PN16, PN17` | 7 | 24 | 30 | 42.10 | **4.39 GB** | 63.8% |
| **Candidate 2 (~8 GB)** | 8.0 GB | `PN00, PN01, PN05, PN06, PN09, PN10, PN12, PN16, PN17` | 9 | 30 | 37 | 67.62 | **7.05 GB** | 78.7% |
| **Candidate 3 (~10 GB Recommended)** | 10.0 GB | `PN00, PN01, PN03, PN05, PN06, PN07, PN09, PN10, PN12, PN13, PN16, PN17` | **12** | **36** | **42** | **102.52** | **10.69 GB** | **89.4%** |

---

## Proposed Acquisition Recommendation

### **PROPOSED PLAN: Candidate 3 (~10.69 GB Subset)**
- **Subjects Included (12 of 14)**: `PN00`, `PN01`, `PN03`, `PN05`, `PN06`, `PN07`, `PN09`, `PN10`, `PN12`, `PN13`, `PN16`, `PN17`.
- **Excluded Subjects (2 of 14)**:
  - `PN11` (145 min, 1 seizure, 0.25 GB)
  - `PN14` (1,408 min, 4 seizures, 2.45 GB)
- **Key Metrics**:
  - **Total Size**: **~10.69 GB** (Leaves ~26.3 GB free disk space on local Mac).
  - **Patient Count**: **12 subjects** (85.7% of full database).
  - **EDF Recordings**: **36 files** (87.8% of full database).
  - **Seizure Events Captured**: **42 events** (**89.4% of all documented seizures**).
  - **Patient Diversity**: 6 Males, 6 Females; Age range 20 to 71 years; Focal (Temporal & Frontal) seizure localizations.

---

## Constraint & Safety Verification

- [x] **Zero Downloads Executed**: No network download was initiated.
- [x] **ECG Research Intact**: All ECG models, code, checkpoints, configurations, and datasets remain **100% untouched**.
- [x] **CHB-MIT Code Intact**: Existing CHB-MIT code and logs remain untouched.
- [x] **Zero Architecture / Preprocessing Decisions**: No EEG model architecture was designed, and no signal filtering, DWT, channel selection, or dataset splits were performed.
