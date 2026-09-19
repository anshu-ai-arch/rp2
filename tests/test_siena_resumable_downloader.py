import os
import shutil
import tempfile
import unittest
import pathlib

from src.eeg.siena_resumable_downloader import SienaResumableDownloader


class TestSienaResumableDownloader(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.downloader = SienaResumableDownloader(target_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_complete_file_skip(self):
        mock_file = pathlib.Path(self.test_dir) / "PN00" / "PN00-1.edf"
        mock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(mock_file, "wb") as f:
            f.write(b"\x00" * 1000)

        res = self.downloader.download_file_resumable(
            url="https://example.com/PN00-1.edf",
            dest_path=mock_file,
            expected_size=1000
        )

        self.assertEqual(res["status"], "COMPLETE")
        self.assertEqual(res["local_size_bytes"], 1000)
        self.assertEqual(res["remaining_bytes"], 0)

    def test_anomaly_file_preservation(self):
        mock_file = pathlib.Path(self.test_dir) / "PN00" / "PN00-2.edf"
        mock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(mock_file, "wb") as f:
            f.write(b"\x00" * 1500)

        res = self.downloader.download_file_resumable(
            url="https://example.com/PN00-2.edf",
            dest_path=mock_file,
            expected_size=1000
        )

        self.assertEqual(res["status"], "ANOMALY_LOCAL_LARGER")
        self.assertEqual(res["local_size_bytes"], 1500)


if __name__ == "__main__":
    unittest.main()
