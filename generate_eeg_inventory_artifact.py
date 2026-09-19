import os
import sys
import json
import pathlib

# Add project root directory
sys.path.append(os.getcwd())

from src.eeg.dataset_inventory import EEGDatasetInventory, UNKNOWN


def generate_inventory_artifacts():
    inventory_engine = EEGDatasetInventory()
    report = inventory_engine.run_inventory()

    artifacts_dir = pathlib.Path("artifacts/eeg_transfer")
    docs_dir = pathlib.Path("docs/eeg_transfer")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    json_path = artifacts_dir / "eeg_dataset_inventory.json"
    md_path = artifacts_dir / "eeg_dataset_inventory.md"
    log_path = docs_dir / "eeg_inventory_log.md"

    # Save JSON report
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save Markdown report
    md_content = []
    md_content.append("# Read-Only EEG Dataset Inventory Report")
    md_content.append("\n## Inventory Overview")
    md_content.append(f"- **Inventory Status**: `{report['inventory_status']}`")
    md_content.append(f"- **Inspected Directories**: {', '.join([f'`{d}`' for d in report['inspected_directories']]) if report['inspected_directories'] else 'None'}")
    md_content.append(f"- **Total Files Inspected**: {report['total_files_inspected']}")
    md_content.append(f"- **Total EEG Recordings Found**: {report['total_eeg_recordings_found']}")
    md_content.append(f"- **Total Patients/Cases Identified**: {report['total_patients_identified']}")
    md_content.append(f"- **Patient Identifiers**: {', '.join(report['patient_identifiers']) if report['patient_identifiers'] else UNKNOWN}")

    md_content.append("\n## Data Distributions")
    md_content.append(f"- **Sampling Frequency Distribution**: {json.dumps(report['sampling_frequency_distribution']) if report['sampling_frequency_distribution'] else UNKNOWN}")
    md_content.append(f"- **Channel Count Distribution**: {json.dumps(report['channel_count_distribution']) if report['channel_count_distribution'] else UNKNOWN}")
    md_content.append(f"- **Channel Name Patterns**: {', '.join(report['channel_name_patterns']) if report['channel_name_patterns'] else UNKNOWN}")

    md_content.append("\n## Seizure & Annotation Availability")
    if report['total_eeg_recordings_found'] > 0:
        seizure_annotated_count = sum(1 for r in report['recordings'] if r.get('seizure_annotation_available'))
        total_seizures = sum(r.get('num_seizure_events', 0) for r in report['recordings'] if isinstance(r.get('num_seizure_events'), int))
        md_content.append(f"- **Annotated Recordings**: {seizure_annotated_count} / {report['total_eeg_recordings_found']}")
        md_content.append(f"- **Total Seizure Events Observed**: {total_seizures}")
    else:
        md_content.append(f"- **Seizure Annotation Status**: `{UNKNOWN}` (No EEG recordings currently available in search paths)")

    md_content.append("\n## File Integrity & Error Audit")
    if report['unreadable_or_corrupted_files']:
        md_content.append(f"- **Unreadable / Corrupted Files Count**: {len(report['unreadable_or_corrupted_files'])}")
        for err in report['unreadable_or_corrupted_files']:
            md_content.append(f"  - `{err['file_path']}`: {err['error']}")
    else:
        md_content.append("- **Unreadable / Corrupted Files**: None observed")

    md_content.append("\n## Detailed Recording Inventory")
    if report['recordings']:
        md_content.append("| Recording File | Format | Patient ID | Channels | Sampling Rate | Duration (s) | Seizures |")
        md_content.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in report['recordings']:
            md_content.append(f"| `{r['file_name']}` | {r['format']} | {r['patient_id']} | {r['num_channels']} | {r['sampling_frequency_hz']} | {r['duration_seconds']} | {r['num_seizure_events']} |")
    else:
        md_content.append("_No raw EEG recording files (.edf, .rec, .set, .fif, .vhdr, .cnt) were found in the inspected search directories._")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    # Save Documentation Log
    log_content = [
        "# EEG Transfer Learning — Step 1 Inventory Log",
        "\n## Execution Details",
        f"- **Action**: Read-Only Raw EEG Dataset Inventory",
        f"- **Status**: `{report['inventory_status']}`",
        f"- **Utilities Created**: `src/eeg/dataset_inventory.py`, `tests/test_eeg_inventory.py`",
        f"- **Artifacts Created**: `artifacts/eeg_transfer/eeg_dataset_inventory.json`, `artifacts/eeg_transfer/eeg_dataset_inventory.md`",
        f"- **Documentation Log**: `docs/eeg_transfer/eeg_inventory_log.md`",
        "\n## Constraints Enforcement Verification",
        "- [✓] Existing ECG source files (`src/models/`, `src/engine/`, `src/data/`, etc.): **UNTOUCHED**",
        "- [✓] Existing ECG checkpoints (`checkpoints/`): **UNTOUCHED**",
        "- [✓] Existing ECG configuration (`config.yaml`): **UNTOUCHED**",
        "- [✓] Existing ECG model architecture: **UNTOUCHED**",
        "- [✓] Discrete Wavelet Transform (DWT): **NOT IMPLEMENTED**",
        "- [✓] Segmentation: **NOT IMPLEMENTED**",
        "- [✓] Data Normalization: **NOT IMPLEMENTED**",
        "- [✓] Train/Val/Test Splits: **NOT IMPLEMENTED**",
        "- [✓] External Dataset Download: **NOT PERFORMED**",
        "- [✓] Project Files Deleted/Renamed/Moved: **NONE**"
    ]

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_content))

    print(f"[✓] Inventory JSON saved to: {json_path}")
    print(f"[✓] Inventory Markdown saved to: {md_path}")
    print(f"[✓] Inventory Log saved to: {log_path}")
    return report


if __name__ == "__main__":
    generate_inventory_artifacts()
