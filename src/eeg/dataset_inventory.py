import os
import re
import json
import struct
import pathlib
from typing import Dict, List, Any, Optional, Tuple


UNKNOWN = "UNKNOWN"


def _clean_str(b: bytes) -> str:
    """Helper to decode ASCII bytes, stripping null characters and whitespace."""
    return b.decode("ascii", errors="ignore").replace("\x00", "").strip()


class EEGDatasetInventory:
    """
    Read-only dataset inventory utility for raw EEG recordings.
    Inspects EEG dataset directories, parses metadata directly from raw files
    (EDF, EDF+, MAT, CSV, summary text annotations) without performing any
    preprocessing, filtering, normalization, segmentation, or data mutation.
    """

    def __init__(self, search_directories: Optional[List[str]] = None):
        if search_directories is None:
            self.search_directories = [
                "data/eeg",
                "data/raw_eeg",
                "data/chbmit",
                "data/raw"
            ]
        else:
            self.search_directories = search_directories

    @staticmethod
    def parse_edf_header(file_path: pathlib.Path) -> Dict[str, Any]:
        """
        Parses metadata directly from standard EDF / EDF+ header (ANSI/IEEE EDF spec).
        Returns observed header metadata or error details.
        """
        try:
            with open(file_path, "rb") as f:
                header_256 = f.read(256)
                if len(header_256) < 256:
                    return {
                        "status": "CORRUPTED",
                        "error": "File size smaller than 256-byte EDF header"
                    }

                version = _clean_str(header_256[0:8])
                patient_id = _clean_str(header_256[8:88])
                recording_id = _clean_str(header_256[88:168])
                start_date = _clean_str(header_256[168:176])
                start_time = _clean_str(header_256[176:184])

                try:
                    header_bytes = int(_clean_str(header_256[184:192]))
                except ValueError:
                    header_bytes = UNKNOWN

                reserved = _clean_str(header_256[192:236])

                try:
                    num_records = int(_clean_str(header_256[236:244]))
                except ValueError:
                    num_records = UNKNOWN

                try:
                    record_duration = float(_clean_str(header_256[244:252]))
                except ValueError:
                    record_duration = UNKNOWN

                try:
                    num_channels = int(_clean_str(header_256[252:256]))
                except ValueError:
                    num_channels = UNKNOWN

                if num_channels == UNKNOWN or num_channels <= 0:
                    return {
                        "status": "CORRUPTED",
                        "error": "Invalid channel count in EDF header"
                    }

                # Read signals header (num_channels * 256 bytes)
                signals_header_bytes = num_channels * 256
                signals_header = f.read(signals_header_bytes)
                if len(signals_header) < signals_header_bytes:
                    return {
                        "status": "CORRUPTED",
                        "error": f"Truncated channel header (expected {signals_header_bytes} bytes, got {len(signals_header)})"
                    }

                # Parse channel labels (num_channels * 16 bytes)
                labels = []
                for c in range(num_channels):
                    lbl = _clean_str(signals_header[c * 16:(c + 1) * 16])
                    labels.append(lbl)

                # Offset after labels:
                # transducers: 80 * num_channels
                # dimensions: 8 * num_channels
                # physical_min: 8 * num_channels
                # physical_max: 8 * num_channels
                # digital_min: 8 * num_channels
                # digital_max: 8 * num_channels
                # prefiltering: 80 * num_channels
                offset_samples_per_rec = num_channels * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)

                samples_per_record = []
                for c in range(num_channels):
                    start_idx = offset_samples_per_rec + (c * 8)
                    sample_str = _clean_str(signals_header[start_idx:start_idx + 8])
                    try:
                        samples_per_record.append(int(sample_str))
                    except ValueError:
                        samples_per_record.append(UNKNOWN)

                # Compute sampling rate and duration
                sampling_freqs = []
                for spr in samples_per_record:
                    if spr != UNKNOWN and record_duration != UNKNOWN and record_duration > 0:
                        sampling_freqs.append(spr / record_duration)
                    else:
                        sampling_freqs.append(UNKNOWN)

                valid_freqs = [sf for sf in sampling_freqs if sf != UNKNOWN]
                primary_sf = valid_freqs[0] if valid_freqs else UNKNOWN

                if num_records != UNKNOWN and record_duration != UNKNOWN and num_records >= 0 and record_duration >= 0:
                    total_duration_sec = num_records * record_duration
                else:
                    total_duration_sec = UNKNOWN

                if valid_freqs and total_duration_sec != UNKNOWN:
                    total_samples_primary = int(primary_sf * total_duration_sec)
                else:
                    total_samples_primary = UNKNOWN

                return {
                    "status": "VALID",
                    "format": "EDF+C" if "EDF+C" in reserved else ("EDF+D" if "EDF+D" in reserved else "EDF"),
                    "patient_id_header": patient_id if patient_id else UNKNOWN,
                    "recording_id_header": recording_id if recording_id else UNKNOWN,
                    "start_date": start_date if start_date else UNKNOWN,
                    "start_time": start_time if start_time else UNKNOWN,
                    "num_channels": num_channels,
                    "channel_names": labels,
                    "sampling_frequency_hz": primary_sf,
                    "num_data_records": num_records,
                    "record_duration_sec": record_duration,
                    "total_duration_sec": total_duration_sec,
                    "total_samples": total_samples_primary
                }
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e)
            }

    @staticmethod
    def parse_chbmit_summary_file(summary_path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
        """
        Parses CHB-MIT summary text annotation file (e.g. chb01-summary.txt).
        Extracts seizure start/end times and count per EDF file.
        """
        file_annotations = {}
        if not summary_path.exists():
            return file_annotations

        try:
            with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            file_blocks = re.split(r"File Name:\s*", content)
            for block in file_blocks[1:]:
                lines = block.strip().split("\n")
                if not lines:
                    continue
                filename = lines[0].strip()

                num_seizures = UNKNOWN
                seizure_events = []

                for line in lines:
                    line_clean = line.strip()
                    if "Number of Seizures in File:" in line_clean:
                        try:
                            num_seizures = int(line_clean.split(":")[-1].strip())
                        except ValueError:
                            num_seizures = UNKNOWN
                    elif "Seizure" in line_clean and "Start Time:" in line_clean:
                        try:
                            start_t = int(line_clean.split(":")[-1].replace("seconds", "").strip())
                            seizure_events.append({"start_sec": start_t})
                        except ValueError:
                            pass
                    elif "Seizure" in line_clean and "End Time:" in line_clean:
                        try:
                            end_t = int(line_clean.split(":")[-1].replace("seconds", "").strip())
                            if seizure_events and "end_sec" not in seizure_events[-1]:
                                seizure_events[-1]["end_sec"] = end_t
                        except ValueError:
                            pass

                file_annotations[filename] = {
                    "seizure_annotation_available": True,
                    "num_seizures": num_seizures,
                    "seizure_events": seizure_events
                }
        except Exception as e:
            pass

        return file_annotations

    def run_inventory(self) -> Dict[str, Any]:
        """
        Executes complete read-only inventory scan over target search directories.
        Returns deterministic JSON-serializable inventory report.
        """
        inspected_directories = []
        recordings = []
        files_inspected_count = 0
        total_patients = set()

        sf_distribution = {}
        channel_count_dist = {}
        channel_name_patterns = set()

        unreadable_corrupted_files = []

        # Find all summary annotation files first
        summary_annotations = {}
        for dir_str in self.search_directories:
            p_dir = pathlib.Path(dir_str)
            if p_dir.exists() and p_dir.is_dir():
                inspected_directories.append(str(p_dir))
                for summary_file in p_dir.glob("*-summary.txt"):
                    parsed_ann = self.parse_chbmit_summary_file(summary_file)
                    summary_annotations.update(parsed_ann)

        # Scan for EEG recording files (.edf, .rec, .set, .fif, .vhdr, .cnt)
        eeg_extensions = {".edf", ".rec", ".set", ".fif", ".vhdr", ".cnt"}
        
        found_eeg_files = []
        for dir_str in self.search_directories:
            p_dir = pathlib.Path(dir_str)
            if p_dir.exists() and p_dir.is_dir():
                for entry in p_dir.rglob("*"):
                    files_inspected_count += 1
                    if entry.is_file() and entry.suffix.lower() in eeg_extensions:
                        found_eeg_files.append(entry)

        for edf_path in sorted(found_eeg_files):
            file_name = edf_path.name
            parent_dir_name = edf_path.parent.name
            
            # Extract patient identifier if present (e.g. chb01_01.edf -> chb01 or parent dir)
            pat_match = re.match(r"^(chb\d+|subject\d+|patient\d+|p\d+|sub-\w+)", file_name, re.IGNORECASE)
            if pat_match:
                patient_id = pat_match.group(1).lower()
            elif parent_dir_name.startswith("chb") or parent_dir_name.startswith("sub"):
                patient_id = parent_dir_name.lower()
            else:
                patient_id = UNKNOWN

            if patient_id != UNKNOWN:
                total_patients.add(patient_id)

            parsed_header = self.parse_edf_header(edf_path)

            if parsed_header.get("status") != "VALID":
                unreadable_corrupted_files.append({
                    "file_path": str(edf_path),
                    "error": parsed_header.get("error", "Unknown error")
                })
                continue

            # Annotation metadata
            ann_info = summary_annotations.get(file_name, {})
            seizure_avail = ann_info.get("seizure_annotation_available", False)
            num_seizures = ann_info.get("num_seizures", UNKNOWN if not seizure_avail else 0)
            seizure_events = ann_info.get("seizure_events", [])

            # Update distributions
            sf = parsed_header["sampling_frequency_hz"]
            if sf != UNKNOWN:
                sf_str = f"{sf:.1f} Hz"
                sf_distribution[sf_str] = sf_distribution.get(sf_str, 0) + 1

            cc = parsed_header["num_channels"]
            if cc != UNKNOWN:
                cc_str = f"{cc} channels"
                channel_count_dist[cc_str] = channel_count_dist.get(cc_str, 0) + 1

            for ch_name in parsed_header.get("channel_names", []):
                channel_name_patterns.add(ch_name)

            record_entry = {
                "file_path": str(edf_path),
                "file_name": file_name,
                "file_size_bytes": edf_path.stat().st_size,
                "patient_id": patient_id,
                "recording_id": file_name,
                "format": parsed_header["format"],
                "sampling_frequency_hz": sf,
                "num_channels": cc,
                "channel_names": parsed_header.get("channel_names", []),
                "num_samples": parsed_header.get("total_samples", UNKNOWN),
                "duration_seconds": parsed_header.get("total_duration_sec", UNKNOWN),
                "seizure_annotation_available": seizure_avail,
                "num_seizure_events": num_seizures,
                "seizure_events": seizure_events,
                "header_observed_metadata": {
                    "patient_id_header": parsed_header.get("patient_id_header", UNKNOWN),
                    "recording_id_header": parsed_header.get("recording_id_header", UNKNOWN),
                    "start_date": parsed_header.get("start_date", UNKNOWN),
                    "start_time": parsed_header.get("start_time", UNKNOWN)
                }
            }
            recordings.append(record_entry)

        has_eeg_data = len(recordings) > 0

        inventory_report = {
            "inventory_status": "EEG_DATASET_PRESENT" if has_eeg_data else "NO_EEG_DATASET_FOUND",
            "inspected_directories": inspected_directories,
            "total_files_inspected": files_inspected_count,
            "total_eeg_recordings_found": len(recordings),
            "total_patients_identified": len(total_patients) if total_patients else UNKNOWN,
            "patient_identifiers": sorted(list(total_patients)) if total_patients else [],
            "sampling_frequency_distribution": sf_distribution,
            "channel_count_distribution": channel_count_dist,
            "channel_name_patterns": sorted(list(channel_name_patterns)),
            "unreadable_or_corrupted_files": unreadable_corrupted_files,
            "recordings": recordings
        }

        return inventory_report


if __name__ == "__main__":
    inventory_engine = EEGDatasetInventory()
    report = inventory_engine.run_inventory()
    print(json.dumps(report, indent=2))
