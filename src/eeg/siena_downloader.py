import os
import sys
import time
import subprocess
import pathlib
from typing import List, Dict, Any

from src.eeg.siena_selection import build_siena_subset_candidates, SIENA_FULL_SUBJECT_METADATA

PHYSIONET_BASE_URL = "https://physionet.org/files/siena-scalp-eeg/1.0.0/"


def get_approved_candidate_3_files() -> Dict[str, Any]:
    candidates = build_siena_subset_candidates()
    c10 = candidates["candidate_10gb"]
    return c10


def download_file_with_curl(url: str, dest_path: pathlib.Path) -> bool:
    """
    Downloads a file using system curl with automatic HTTP resume (-C -),
    retries, and browser user-agent.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "curl",
        "-C", "-",
        "--retry", "5",
        "--retry-delay", "2",
        "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "--fail",
        "-o", str(dest_path),
        url
    ]

    print(f"[*] Downloading {dest_path.name}...")
    try:
        res = subprocess.run(cmd, check=True)
        if dest_path.exists() and dest_path.stat().st_size > 0:
            print(f"  [✓] Complete: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
            return True
        else:
            print(f"  [!] Failed: File {dest_path.name} is missing or 0 bytes.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"  [!] Curl error downloading {dest_path.name}: {e}")
        return False


def execute_siena_acquisition(target_dir: str = "data/siena") -> Dict[str, Any]:
    c10 = get_approved_candidate_3_files()
    approved_subjects = c10["subjects"]

    target_path = pathlib.Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("  PHASE 3: SIENA SCALP EEG DATASET SUBSET ACQUISITION (Candidate 3)")
    print("=================================================================")
    print(f"  Target Destination Directory: {target_path.resolve()}")
    print(f"  Approved Subjects (12):      {', '.join(approved_subjects)}")
    print(f"  Excluded Subjects (2):       PN11, PN14")
    print(f"  Approved EDF Count:          {c10['total_edf_files']} files")
    print(f"  Expected Size:               {c10['est_storage_size_gb']:.2f} GB")
    print("=================================================================\n")

    # 1. Download subject_info.csv
    info_url = PHYSIONET_BASE_URL + "subject_info.csv"
    download_file_with_curl(info_url, target_path / "subject_info.csv")

    # 2. Download files for approved subjects ONLY
    downloaded_summary = []
    failed_summary = []

    for subj_id in approved_subjects:
        meta = SIENA_FULL_SUBJECT_METADATA[subj_id]
        subj_dir = target_path / subj_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Download per-subject seizure list if present
        sz_file_name = f"Seizures-list-{subj_id}.txt"
        sz_url = PHYSIONET_BASE_URL + f"{subj_id}/{sz_file_name}"
        download_file_with_curl(sz_url, subj_dir / sz_file_name)

        # Download EDF files
        for edf_file in meta["edf_files"]:
            edf_url = PHYSIONET_BASE_URL + f"{subj_id}/{edf_file}"
            edf_dest = subj_dir / edf_file
            ok = download_file_with_curl(edf_url, edf_dest)
            if ok:
                downloaded_summary.append(str(edf_dest))
            else:
                failed_summary.append(str(edf_dest))

    print("\n=================================================================")
    print("                ACQUISITION SUMMARY COMPLETED                    ")
    print("=================================================================")
    print(f"  Total Files Downloaded/Verified: {len(downloaded_summary)}")
    print(f"  Total Failed Downloads:         {len(failed_summary)}")
    print("=================================================================\n")

    return {
        "status": "COMPLETED" if not failed_summary else "PARTIAL_FAILURE",
        "downloaded_files_count": len(downloaded_summary),
        "failed_files_count": len(failed_summary),
        "downloaded_files": downloaded_summary,
        "failed_files": failed_summary
    }


if __name__ == "__main__":
    execute_siena_acquisition()
