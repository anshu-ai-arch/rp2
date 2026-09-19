import unittest
from src.eeg.siena_selection import build_siena_subset_candidates, SIENA_FULL_SUBJECT_METADATA


class TestSienaSelection(unittest.TestCase):

    def test_full_metadata_integrity(self):
        self.assertEqual(len(SIENA_FULL_SUBJECT_METADATA), 14)
        total_seizures = sum(s["number_seizures"] for s in SIENA_FULL_SUBJECT_METADATA.values())
        total_files = sum(len(s["edf_files"]) for s in SIENA_FULL_SUBJECT_METADATA.values())
        self.assertEqual(total_seizures, 47)
        self.assertEqual(total_files, 41)

    def test_candidate_evaluations(self):
        candidates = build_siena_subset_candidates()
        self.assertIn("candidate_5gb", candidates)
        self.assertIn("candidate_8gb", candidates)
        self.assertIn("candidate_10gb", candidates)

        c10 = candidates["candidate_10gb"]
        self.assertEqual(c10["subject_count"], 12)
        self.assertEqual(c10["total_seizures"], 42)
        self.assertLessEqual(c10["est_storage_size_gb"], 11.0)


if __name__ == "__main__":
    unittest.main()
