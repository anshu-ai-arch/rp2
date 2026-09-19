# Storage-Safe CHB-MIT Dataset Acquisition Strategy — Step 5 Report

## 1. Local Storage Environment Inspection

| Parameter | Observed Measurement | Notes |
| :--- | :--- | :--- |
| **Current Working Path** | `/Users/anshutiwari123/Library/Mobile Documents/com~apple~CloudDocs/resercharrhythmia` | Resides within iCloud Drive storage |
| **Available Local Free Space** | **40.09 GB** | Measured via `shutil.disk_usage('.')` |
| **Total Project Size** | **530.20 MB** (0.53 GB) | Lightweight repository footprint |
| **ECG Dataset (`data/raw/`) Size**| **91.90 MB** | MIT-BIH Arrhythmia raw data |
| **Model Checkpoints (`checkpoints/`) Size** | **14.19 MB** | Pre-trained PyTorch weight checkpoints |
| **Mounted External Volumes** | **None** (`/Volumes/` is empty) | No external USB/Thunderbolt drive detected |
| **Cloud Storage Integration** | **iCloud Drive Active** | No Google Drive in `CloudStorage/` |
| **Local Cache Directories** | `~/.cache` exists | `~/.wfdb` does not exist |

---

## 2. Dataset Size Analysis

| Target Dataset | Estimated Size | File Count / Scope | Feasibility on Local Disk (40.09 GB Free) |
| :--- | :--- | :--- | :--- |
| **Full CHB-MIT Database (v1.0.0)** | **~48.5 – 50.0 GB** | 686 `.edf` files across 24 subjects | **NOT FEASIBLE** (Deficit of ~10 GB; risk of disk exhaustion) |
| **Subject `chb01` Subset** | **~1.2 GB** | 43 `.edf` files + 1 summary text file | **FEASIBLE** (Uses < 3% of available local free space) |

---

## 3. Placement Requirements for Subject `chb01`

If `chb01` is selected for initial verification:
- **Target Folder**: `data/chbmit/chb01/`
- **Required Files**: `chb01_01.edf` through `chb01_46.edf` + `chb01-summary.txt`
- **Required Disk Space**: `~1.2 GB`
- **Isolation**: Stored under `data/chbmit/`, strictly outside `data/raw/`.

---

## 4. Storage Risks & Constraints

1. **Local Capacity Deficit**: Attempting to download all 24 subjects (~50 GB) directly to local storage will fail with `No space left on device`.
2. **iCloud Synchronization Overhead**: The project directory is located in iCloud Drive (`com~apple~CloudDocs`). Adding 50 GB would trigger heavy cloud sync network overhead.

---

## 5. Storage & Acquisition Options Available to the Researcher

1. **Option 1 — Single Subject Subset (`chb01` ~1.2 GB)**: Place `chb01` into `data/chbmit/chb01/` to establish and test the EEG pipeline safely.
2. **Option 2 — Multi-Subject Subset (e.g. `chb01` to `chb10` ~20 GB)**: Place a 10-subject subset into `data/chbmit/` (< 20 GB), remaining well within the 40.09 GB limit.
3. **Option 3 — Mount External High-Capacity Volume**: Connect an external drive with > 50 GB free space and download or link the full dataset to `/Volumes/<ExternalDrive>/chbmit/`.
4. **Option 4 — Free Up Local Volume Space**: Clear ~15 GB of disk space on `/Users/anshutiwari123` to achieve > 55 GB free space prior to full dataset download.

---

## 6. Zero-Modification & Safety Verification

- [x] **No Dataset Download Occurred**: 0 bytes were downloaded or placed during Step 5 inspection.
- [x] **ECG Research Project Intact**: `src/models/`, `src/engine/`, `src/data/`, `checkpoints/`, `config.yaml`, and ECG raw data remain **100% untouched**.
- [x] **Zero Preprocessing Executed**: No filtering, resampling, DWT, channel selection, segmentation, or data splitting was performed.
