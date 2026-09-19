import os
import shutil
import tempfile
import unittest
import pathlib

from src.eeg.siena_inventory import SienaRecordInventory, generate_siena_inventory_artifacts


class TestSienaRecordInventory(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_nonexistent_directory(self):
        inv = SienaRecordInventory(dataset_root=os.path.join(self.test_dir, "nonexistent"))
        res = inv.run_inventory()
        self.assertEqual(res["inventory_status"], "NO_SIENA_DATASET_FOUND")
        self.assertEqual(res["total_subjects_found"], 0)
        self.assertEqual(res["total_edf_recordings"], 0)

    def test_empty_directory(self):
        inv = SienaRecordInventory(dataset_root=self.test_dir)
        res = inv.run_inventory()
        self.assertEqual(res["inventory_status"], "NO_SIENA_DATASET_FOUND")
        self.assertEqual(res["total_subjects_found"], 0)
        self.assertEqual(res["total_edf_recordings"], 0)

    def _create_mock_edf(self, file_path: pathlib.Path, num_channels: int = 2, sf: int = 512, duration_sec: int = 10):
        # Build 256-byte header
        header = bytearray(256)
        header[0:8] = b"0       "
        header[8:88] = b"PN00                                                                            "
        header[88:168] = b"Rec1                                                                            "
        header[168:176] = b"01.01.20"
        header[176:184] = b"00.00.00"
        header_bytes = 256 + (num_channels * 256)
        header[184:192] = f"{header_bytes:<8}".encode("ascii")
        header[192:236] = b" " * 44
        num_records = duration_sec
        header[236:244] = f"{num_records:<8}".encode("ascii")
        header[244:252] = "1.0     ".encode("ascii")
        header[252:256] = f"{num_channels:<4}".encode("ascii")

        # Signals header: 256 bytes per channel
        sig_header = bytearray(num_channels * 256)
        for c in range(num_channels):
            lbl = f"EEG Ch{c+1}".ljust(16).encode("ascii")
            sig_header[c * 16:(c + 1) * 16] = lbl

            spr_offset = num_channels * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80) + (c * 8)
            spr_val = f"{sf:<8}".encode("ascii")
            sig_header[spr_offset:spr_offset + 8] = spr_val

        with open(file_path, "wb") as f:
            f.write(header)
            f.write(sig_header)
            f.write(b"\x00" * (num_records * num_channels * sf * 2))

    def test_mock_siena_subject(self):
        subj_dir = pathlib.Path(self.test_dir) / "PN00"
        subj_dir.mkdir(parents=True, exist_ok=True)

        edf_file = subj_dir / "PN00-1.edf"
        self._create_mock_edf(edf_file, num_channels=2, sf=512, duration_sec=5)

        # Mock subject_info.csv
        csv_file = pathlib.Path(self.test_dir) / "subject_info.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("subject,age,sex,seizures\nPN00,45,M,2\n")

        # Mock Seizures-list-PN00.txt
        sz_file = subj_dir / "Seizures-list-PN00.txt"
        with open(sz_file, "w", encoding="utf-8") as f:
            f.write("File name: PN00-1.edf\nSeizure 1: 10 to 30\n")

        inv = SienaRecordInventory(dataset_root=self.test_dir)
        res = inv.run_inventory()

        self.assertEqual(res["inventory_status"], "SIENA_DATASET_PRESENT")
        self.assertEqual(res["total_subjects_found"], 1)
        self.assertEqual(res["total_edf_recordings"], 1)
        self.assertIn("PN00", res["subjects"])

        subj_data = res["subjects"]["PN00"]
        self.assertEqual(subj_data["age"], "45")
        self.assertEqual(subj_data["sex"], "M")
        self.assertTrue(subj_data["has_seizure_file"])
        self.assertEqual(len(subj_data["recordings"]), 1)

        rec = subj_data["recordings"][0]
        self.assertEqual(rec["file_name"], "PN00-1.edf")
        self.assertEqual(rec["num_channels"], 2)
        self.assertEqual(rec["sampling_frequency_hz"], 512.0)
        self.assertEqual(rec["duration_seconds"], 5.0)
        self.assertTrue(rec["is_seizure_recording"])
        self.assertEqual(rec["num_seizure_events"], 1)
        self.assertEqual(rec["seizure_events"][0]["start_sec"], 10)
        self.assertEqual(rec["seizure_events"][0]["end_sec"], 30)


if __name__ == "__main__":
    unittest.main()
