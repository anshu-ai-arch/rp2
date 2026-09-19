import os
import sys
import json
import tempfile
import unittest
import pathlib

# Add project root directory
sys.path.append(os.getcwd())

from src.eeg.dataset_inventory import EEGDatasetInventory, UNKNOWN


class TestEEGDatasetInventory(unittest.TestCase):

    def test_empty_directory_inventory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inventory = EEGDatasetInventory(search_directories=[temp_dir])
            report = inventory.run_inventory()
            
            self.assertEqual(report["inventory_status"], "NO_EEG_DATASET_FOUND")
            self.assertEqual(report["total_eeg_recordings_found"], 0)
            self.assertEqual(report["total_patients_identified"], UNKNOWN)
            self.assertEqual(report["recordings"], [])

    def test_mock_edf_header_parser(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            edf_file = pathlib.Path(temp_dir) / "chb01_01.edf"
            
            # Construct a valid fixed-width 256-byte EDF header + 256-byte signal header for 1 channel
            header_str = (
                f"{'0':<8}"
                f"{'Patient_01':<80}"
                f"{'Recording_01':<80}"
                f"{'01.01.21':<8}"
                f"{'00.00.00':<8}"
                f"{'512':<8}"
                f"{'EDF+C':<44}"
                f"{'10':<8}"
                f"{'1':<8}"
                f"{'1':<4}"
            )
            header_256 = header_str.encode("ascii")
            self.assertEqual(len(header_256), 256)

            # Signals header (1 channel * 256 bytes)
            sig_str = (
                f"{'EEG FP1-F7':<16}"
                f"{'AgAgCl electrode':<80}"
                f"{'uV':<8}"
                f"{'-800':<8}"
                f"{'800':<8}"
                f"{'-2048':<8}"
                f"{'2048':<8}"
                f"{'HP:0.5Hz LP:50Hz':<80}"
                f"{'256':<8}"
                f"{'':<32}"
            )
            sig_header = sig_str.encode("ascii")
            self.assertEqual(len(sig_header), 256)

            with open(edf_file, "wb") as f:
                f.write(header_256)
                f.write(sig_header)
                # Write dummy signal data (10 records * 256 samples * 2 bytes = 5120 bytes)
                f.write(bytes(5120))

            inventory = EEGDatasetInventory(search_directories=[temp_dir])
            report = inventory.run_inventory()

            self.assertEqual(report["inventory_status"], "EEG_DATASET_PRESENT")
            self.assertEqual(report["total_eeg_recordings_found"], 1)
            self.assertEqual(report["total_patients_identified"], 1)
            self.assertIn("chb01", report["patient_identifiers"])
            
            rec = report["recordings"][0]
            self.assertEqual(rec["file_name"], "chb01_01.edf")
            self.assertEqual(rec["format"], "EDF+C")
            self.assertEqual(rec["num_channels"], 1)
            self.assertEqual(rec["sampling_frequency_hz"], 256.0)
            self.assertEqual(rec["duration_seconds"], 10.0)
            self.assertEqual(rec["num_samples"], 2560)
            self.assertEqual(rec["channel_names"], ["EEG FP1-F7"])


if __name__ == "__main__":
    unittest.main()
