import json
import pathlib
from typing import Dict, List, Any


SIENA_FULL_SUBJECT_METADATA = {
    "PN00": {"age": 55, "gender": "Male", "seizure_type": "IAS", "localization": "T", "lateralization": "R", "channels": 29, "number_seizures": 5, "rec_time_minutes": 198.0, "edf_files": ["PN00-1.edf", "PN00-2.edf", "PN00-3.edf", "PN00-4.edf", "PN00-5.edf"], "est_size_gb": 0.344},
    "PN01": {"age": 46, "gender": "Male", "seizure_type": "IAS", "localization": "T", "lateralization": "L", "channels": 29, "number_seizures": 2, "rec_time_minutes": 809.0, "edf_files": ["PN01-1.edf"], "est_size_gb": 1.406},
    "PN03": {"age": 54, "gender": "Male", "seizure_type": "IAS", "localization": "T", "lateralization": "R", "channels": 29, "number_seizures": 2, "rec_time_minutes": 752.0, "edf_files": ["PN03-1.edf", "PN03-2.edf"], "est_size_gb": 1.307},
    "PN05": {"age": 51, "gender": "Female", "seizure_type": "IAS", "localization": "T", "lateralization": "L", "channels": 29, "number_seizures": 3, "rec_time_minutes": 359.0, "edf_files": ["PN05-2.edf", "PN05-3.edf", "PN05-4.edf"], "est_size_gb": 0.624},
    "PN06": {"age": 36, "gender": "Male", "seizure_type": "IAS", "localization": "T", "lateralization": "L", "channels": 29, "number_seizures": 5, "rec_time_minutes": 722.0, "edf_files": ["PN06-1.edf", "PN06-2.edf", "PN06-3.edf", "PN06-4.edf", "PN06-5.edf"], "est_size_gb": 1.255},
    "PN07": {"age": 20, "gender": "Female", "seizure_type": "IAS", "localization": "T", "lateralization": "L", "channels": 29, "number_seizures": 1, "rec_time_minutes": 523.0, "edf_files": ["PN07-1.edf"], "est_size_gb": 0.909},
    "PN09": {"age": 27, "gender": "Female", "seizure_type": "IAS", "localization": "T", "lateralization": "L", "channels": 29, "number_seizures": 3, "rec_time_minutes": 410.0, "edf_files": ["PN09-1.edf", "PN09-2.edf", "PN09-3.edf"], "est_size_gb": 0.713},
    "PN10": {"age": 25, "gender": "Male", "seizure_type": "FBTC", "localization": "F", "lateralization": "Bilateral", "channels": 20, "number_seizures": 10, "rec_time_minutes": 1002.0, "edf_files": ["PN10-1.edf", "PN10-2.edf", "PN10-3.edf", "PN10-4.edf", "PN10-5.edf", "PN10-6.edf"], "est_size_gb": 1.742},
    "PN11": {"age": 58, "gender": "Female", "seizure_type": "IAS", "localization": "T", "lateralization": "R", "channels": 29, "number_seizures": 1, "rec_time_minutes": 145.0, "edf_files": ["PN11-1.edf"], "est_size_gb": 0.252},
    "PN12": {"age": 71, "gender": "Male", "seizure_type": "IAS", "localization": "T", "lateralization": "L", "channels": 29, "number_seizures": 4, "rec_time_minutes": 246.0, "edf_files": ["PN12-1.edf", "PN12-2.edf", "PN12-3.edf"], "est_size_gb": 0.428},
    "PN13": {"age": 34, "gender": "Female", "seizure_type": "IAS", "localization": "T", "lateralization": "L", "channels": 29, "number_seizures": 3, "rec_time_minutes": 519.0, "edf_files": ["PN13-1.edf", "PN13-2.edf", "PN13-3.edf"], "est_size_gb": 0.902},
    "PN14": {"age": 49, "gender": "Male", "seizure_type": "WIAS", "localization": "T", "lateralization": "L", "channels": 29, "number_seizures": 4, "rec_time_minutes": 1408.0, "edf_files": ["PN14-1.edf", "PN14-2.edf", "PN14-3.edf", "PN14-4.edf"], "est_size_gb": 2.448},
    "PN16": {"age": 41, "gender": "Female", "seizure_type": "IAS", "localization": "T", "lateralization": "L", "channels": 29, "number_seizures": 2, "rec_time_minutes": 303.0, "edf_files": ["PN16-1.edf", "PN16-2.edf"], "est_size_gb": 0.527},
    "PN17": {"age": 42, "gender": "Male", "seizure_type": "IAS", "localization": "T", "lateralization": "R", "channels": 29, "number_seizures": 2, "rec_time_minutes": 308.0, "edf_files": ["PN17-1.edf", "PN17-2.edf"], "est_size_gb": 0.535}
}


def build_siena_subset_candidates() -> Dict[str, Any]:
    """
    Constructs deterministic candidate subsets for initial Siena Scalp EEG transfer learning acquisition.
    Returns metadata dict containing candidate metrics.
    """
    candidates = {
        "candidate_5gb": {
            "name": "Candidate 1 (~5 GB Subset)",
            "target_gb_limit": 5.0,
            "subjects": ["PN00", "PN05", "PN09", "PN10", "PN12", "PN16", "PN17"]
        },
        "candidate_8gb": {
            "name": "Candidate 2 (~8 GB Subset)",
            "target_gb_limit": 8.0,
            "subjects": ["PN00", "PN01", "PN05", "PN06", "PN09", "PN10", "PN12", "PN16", "PN17"]
        },
        "candidate_10gb": {
            "name": "Candidate 3 (~10 GB Recommended Subset)",
            "target_gb_limit": 10.0,
            "subjects": ["PN00", "PN01", "PN03", "PN05", "PN06", "PN07", "PN09", "PN10", "PN12", "PN13", "PN16", "PN17"]
        }
    }

    evaluated = {}
    for c_id, c_data in candidates.items():
        subjs = c_data["subjects"]
        tot_files = 0
        tot_seizures = 0
        tot_minutes = 0.0
        tot_gb = 0.0
        genders = set()
        ages = []
        edf_list = []

        for s_id in subjs:
            meta = SIENA_FULL_SUBJECT_METADATA[s_id]
            tot_files += len(meta["edf_files"])
            tot_seizures += meta["number_seizures"]
            tot_minutes += meta["rec_time_minutes"]
            tot_gb += meta["est_size_gb"]
            genders.add(meta["gender"])
            ages.append(meta["age"])
            edf_list.extend([f"{s_id}/{f}" for f in meta["edf_files"]])

        evaluated[c_id] = {
            "candidate_id": c_id,
            "name": c_data["name"],
            "target_gb_limit": c_data["target_gb_limit"],
            "subject_count": len(subjs),
            "subjects": subjs,
            "total_edf_files": tot_files,
            "total_seizures": tot_seizures,
            "total_rec_time_minutes": tot_minutes,
            "total_rec_time_hours": tot_minutes / 60.0,
            "est_storage_size_gb": tot_gb,
            "gender_diversity": sorted(list(genders)),
            "age_range": [min(ages), max(ages)],
            "edf_file_list": edf_list
        }

    return evaluated
