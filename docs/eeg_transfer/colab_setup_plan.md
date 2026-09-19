# Google Colab Setup & Architecture Plan — EEG Transfer Learning

## 1. Intended Colab + Google Drive Architecture

The migration of the EEG transfer-learning research track to Google Colab uses a hybrid persistent/ephemeral storage architecture:
- **Codebase Execution**: Code runs in Colab's high-performance Linux runtime environment (`/content/resercharrhythmia`).
- **Persistent Storage Root**: All heavy persistent assets (raw EEG recordings, fine-tuned checkpoints, experiment logs, artifacts) reside in a dedicated Google Drive folder:
  ```text
  /content/drive/MyDrive/eeg_transfer_research/
  ```

---

## 2. Persistent vs. Temporary Storage Allocation Strategy

| Asset Type | Storage Location | Lifetime / Persistence | Rationale |
| :--- | :--- | :--- | :--- |
| **Source Code & Scripts** | `/content/resercharrhythmia/` | Ephemeral (Session) | Cloned/mounted for high-speed execution |
| **Raw CHB-MIT Dataset** | `/content/drive/MyDrive/eeg_transfer_research/data/chbmit` | Persistent | Prevents re-downloading ~50 GB across Colab session restarts |
| **EEG Model Checkpoints** | `/content/drive/MyDrive/eeg_transfer_research/checkpoints` | Persistent | Guards against data loss during runtime disconnects |
| **EEG Research Artifacts** | `/content/drive/MyDrive/eeg_transfer_research/artifacts` | Persistent | Stores JSON/MD reports permanently |
| **EEG Experiment Logs** | `/content/drive/MyDrive/eeg_transfer_research/logs` | Persistent | Preserves execution logs across sessions |

---

## 3. Storage Rationale for CHB-MIT Dataset

- **Bandwidth & Quota Conservation**: Downloading the ~50 GB CHB-MIT dataset on every Colab session causes unnecessary network bandwidth usage and PhysioNet server overhead. Storing the dataset under `/content/drive/MyDrive/eeg_transfer_research/data/chbmit` ensures it is downloaded once and accessible across all future sessions.
- **Drive Capacity Safety**: Google Drive offers ample cloud storage capacity, resolving local Mac disk space constraints (which were capped at 40.09 GB free space).

---

## 4. ECG and EEG Research Isolation

- **Separate Configuration Files**: `config.yaml` remains the master configuration for the ECG project. All EEG transfer settings use `config/eeg_config.yaml`.
- **Separate Checkpoint Directories**: ECG checkpoints reside in `./checkpoints/`, while EEG checkpoints will be saved to `/content/drive/MyDrive/eeg_transfer_research/checkpoints/`.
- **Separate Data Paths**: ECG raw data is stored in `data/raw/` (MIT-BIH), while EEG raw data is stored in `data/chbmit/`.
- **Zero Risk of Overwriting**: Strict path separation ensures 0 risk of modifying existing ECG research models or experiment artifacts.

---

## 5. Directory Structure Established by Step 2

The Colab setup script initializes the following persistent directory tree under Google Drive:
```text
/content/drive/MyDrive/eeg_transfer_research/
├── data/
│   └── chbmit/          <-- Raw CHB-MIT EDF recordings & summary files
├── artifacts/           <-- JSON & MD research inventory artifacts
├── checkpoints/         <-- Fine-tuned PyTorch model weights (.pth)
├── experiments/         <-- Experiment output metrics & evaluation figures
└── logs/                <-- Execution logs & audit logs
```

---

## 6. Intentionally Unconstrained Parameters (Not Decided Yet)

Per strict research rules, the following decisions are **intentionally deferred** and have **NOT** been set in `config/eeg_config.yaml`:
- **Signal Preprocessing**: Bandpass filtering cutoffs, notch filtering, downsampling rates.
- **Wavelet Transformation**: Discrete Wavelet Transform (DWT) decomposition levels or wavelet families (e.g. `db4`, `sym5`).
- **Channel Selection**: Specific EEG montage selection (e.g. 18-channel bipolar vs. 23-channel standard).
- **Segmentation**: Window length (seconds) and overlap percentage.
- **Data Splitting**: Patient-wise inter-subject split ratios (train/val/test).
- **Model Architecture & Hyperparameters**: Transfer head dimensions, frozen vs. fine-tuned backbone layers, learning rate, batch size, epoch count.
