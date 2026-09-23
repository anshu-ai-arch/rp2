import os
import re
import sys
import time
import struct
import shutil
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
from concurrent.futures import ThreadPoolExecutor, as_completed


# Approved 18 Bipolar Channels Specification
TARGET_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ"
]


def read_edf(filepath):
    """
    Pure Python EDF file reader.
    Extracts header, channel labels, sampling rates, physical ranges, and raw signal matrix.
    """
    with open(filepath, 'rb') as f:
        header = f.read(256)
        n_channels = int(header[252:256].decode('ascii', errors='ignore').strip())
        ch_headers = f.read(256 * n_channels)

        def get_field_blocks(offset_bytes, field_size):
            res = []
            start = offset_bytes * n_channels
            for i in range(n_channels):
                val = ch_headers[start + i*field_size : start + (i+1)*field_size].decode('ascii', errors='ignore').strip()
                res.append(val)
            return res

        labels = [l.upper() for l in get_field_blocks(0, 16)]
        phys_min = [float(v) for v in get_field_blocks(16+80+8, 8)]
        phys_max = [float(v) for v in get_field_blocks(16+80+16, 8)]
        dig_min = [float(v) for v in get_field_blocks(16+80+24, 8)]
        dig_max = [float(v) for v in get_field_blocks(16+80+32, 8)]
        samples_per_rec = [int(v) for v in get_field_blocks(16+80+40+80, 8)]

        num_records = int(header[236:244].decode('ascii', errors='ignore').strip())
        rec_duration = float(header[244:252].decode('ascii', errors='ignore').strip())

        # Read signal data
        raw_data = f.read()

    fs = [int(spr / rec_duration) for spr in samples_per_rec]
    total_samples_per_rec = sum(samples_per_rec)
    data_int16 = np.frombuffer(raw_data, dtype=np.int16)
    
    if len(data_int16) < total_samples_per_rec * num_records:
        num_records = len(data_int16) // total_samples_per_rec

    signals = {}
    idx = 0
    for i in range(n_channels):
        spr = samples_per_rec[i]
        ch_data = []
        for r in range(num_records):
            rec_start = r * total_samples_per_rec + idx
            ch_data.append(data_int16[rec_start : rec_start + spr])
        idx += spr
        
        raw_sig = np.concatenate(ch_data).astype(np.float64)
        p_min, p_max = phys_min[i], phys_max[i]
        d_min, d_max = dig_min[i], dig_max[i]
        if d_max != d_min:
            sig_phys = (raw_sig - d_min) / (d_max - d_min) * (p_max - p_min) + p_min
        else:
            sig_phys = raw_sig
        signals[labels[i]] = (sig_phys, fs[i])

    return signals, rec_duration * num_records


def match_channel_name(target_name, available_labels):
    """Finds exact or prefixed matching channel label from EDF header."""
    target_clean = target_name.upper().replace('EEG ', '').strip()
    for lab in available_labels:
        lab_clean = lab.upper().replace('EEG ', '').strip()
        if target_clean == lab_clean:
            return lab
    return None


def preprocess_signal(sig, orig_fs=256, target_fs=128):
    """Applies bandpass 0.5-45 Hz, notch 60 Hz filter, and resamples to 128 Hz."""
    b_band, a_band = butter(4, [0.5, 45.0], btype='bandpass', fs=orig_fs)
    sig_filtered = filtfilt(b_band, a_band, sig)
    
    b_notch, a_notch = iirnotch(60.0, 30.0, fs=orig_fs)
    sig_filtered = filtfilt(b_notch, a_notch, sig_filtered)
    
    if orig_fs == target_fs:
        return sig_filtered
    elif orig_fs == 256 and target_fs == 128:
        return resample_poly(sig_filtered, 1, 2)
    else:
        num_target_samples = int(len(sig_filtered) * target_fs / orig_fs)
        from scipy.signal import resample
        return resample(sig_filtered, num_target_samples)


def download_file_with_retry(url, dest_path, retries=5):
    """Downloads a file via HTTP with verification and exponential backoff retry."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True
        
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, 'wb') as out_f:
                shutil.copyfileobj(resp, out_f)
            if dest_path.exists() and dest_path.stat().st_size > 0:
                return True
        except Exception as e:
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)
            time.sleep(1)
    return False


def process_single_rec(rec, base_url, raw_path):
    """Processes a single EDF recording and returns extracted window tensors (float32) and metadata."""
    subj_id = rec["subject_id"]
    fname = rec["filename"]
    seizures = rec["seizure_intervals"]
    
    edf_url = f"{base_url}/{subj_id}/{fname}"
    edf_path = raw_path / subj_id / fname
    
    if not download_file_with_retry(edf_url, edf_path):
        print(f"    [!] Failed to download {fname}, skipping.", flush=True)
        return [], [], []
        
    try:
        signals, total_duration = read_edf(edf_path)
    except Exception as e:
        print(f"    [!] Error reading EDF {fname}: {e}", flush=True)
        if edf_path.exists(): edf_path.unlink()
        return [], [], []

    avail_labels = list(signals.keys())
    ch_signals = []
    for tch in TARGET_CHANNELS:
        mch = match_channel_name(tch, avail_labels)
        if mch is None:
            if edf_path.exists(): edf_path.unlink()
            return [], [], []
        sig_raw, orig_fs = signals[mch]
        sig_proc = preprocess_signal(sig_raw, orig_fs=orig_fs, target_fs=128)
        ch_signals.append(sig_proc)

    eeg_matrix = np.array(ch_signals, dtype=np.float32)  # Shape: (18, T_128hz)
    n_samples_128hz = eeg_matrix.shape[1]
    
    win_len_128hz = 8 * 128   # 1024 samples
    step_128hz = 4 * 128      # 512 samples
    step_orig = 4 * 256       # 1024 samples at 256 Hz

    tensors = []
    labels = []
    metadata = []
    
    curr_step = 0
    while (curr_step * step_128hz + win_len_128hz) <= n_samples_128hz:
        start_idx_128 = curr_step * step_128hz
        end_idx_128 = start_idx_128 + win_len_128hz
        
        start_sec = curr_step * 4.0
        end_sec = start_sec + 8.0
        start_sample_orig = curr_step * step_orig
        
        is_seizure = 0
        for sz_st, sz_en in seizures:
            if not (end_sec <= sz_st or start_sec >= sz_en):
                is_seizure = 1
                break
                
        win_data = eeg_matrix[:, start_idx_128:end_idx_128]
        mean = np.mean(win_data, axis=1, keepdims=True)
        std = np.std(win_data, axis=1, keepdims=True)
        win_norm = (win_data - mean) / (std + 1e-8)
        
        tensors.append(win_norm[:, :, np.newaxis].astype(np.float32))
        labels.append(is_seizure)
        metadata.append({
            "subject_id": subj_id,
            "session_filename": fname,
            "window_start_sample_original_fs": start_sample_orig,
            "window_start_seconds": start_sec,
            "label": is_seizure
        })
        curr_step += 1

    # Cleanup temporary EDF file immediately
    if edf_path.exists():
        edf_path.unlink()
        
    return tensors, labels, metadata


def build_chbmit_provenance_dataset(
    raw_dir="data/chbmit/raw",
    output_dir="data/chbmit/processed_provenance",
    random_seed=2023,
    max_workers=3
):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("=================================================================", flush=True)
    print("  REBUILDING CHB-MIT EEG DATASET WITH EXACT SUBJECT PROVENANCE   ", flush=True)
    print("=================================================================", flush=True)
    print(f"[*] Raw Data Cache Directory:  {raw_path.resolve()}", flush=True)
    print(f"[*] Output Dataset Directory:   {out_path.resolve()}", flush=True)
    print(f"[*] Target Channels:            18 Standard Bipolar Channels", flush=True)
    print(f"[*] Window Length:              8.0s (1,024 samples @ 128 Hz)", flush=True)
    print(f"[*] Window Step:                4.0s (512 samples @ 128 Hz)", flush=True)
    print(f"[*] Parallel Download Workers: {max_workers}", flush=True)
    print(f"[*] Random Seed:                {random_seed}", flush=True)
    print("=================================================================\n", flush=True)

    base_url = "https://physionet.org/files/chbmit/1.0.0"
    
    # 1. Fetch summary text files for all 24 subjects
    all_seizure_records = []
    print("[*] Step 1/4: Fetching CHB-MIT Subject Annotations & Summaries...", flush=True)
    for i in range(1, 25):
        subj_id = f"chb{i:02d}"
        sum_filename = f"{subj_id}-summary.txt"
        sum_url = f"{base_url}/{subj_id}/{sum_filename}"
        sum_path = raw_path / subj_id / sum_filename
        
        if download_file_with_retry(sum_url, sum_path):
            with open(sum_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                
            blocks = text.split("File Name:")
            for block in blocks[1:]:
                lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
                if not lines: continue
                fname = lines[0]
                num_seizures = 0
                for l in lines:
                    if "Number of Seizures in File:" in l:
                        num_seizures = int(l.split(":")[-1].strip())
                        
                seizure_intervals = []
                if num_seizures > 0:
                    starts, ends = [], []
                    for l in lines:
                        if "Seizure" in l and "Start Time:" in l:
                            m = re.search(r'(\d+)\s*seconds', l)
                            if m: starts.append(int(m.group(1)))
                        elif "Seizure" in l and "End Time:" in l:
                            m = re.search(r'(\d+)\s*seconds', l)
                            if m: ends.append(int(m.group(1)))
                    for st, en in zip(starts, ends):
                        seizure_intervals.append((st, en))
                
                all_seizure_records.append({
                    "subject_id": subj_id,
                    "filename": fname,
                    "num_seizures": num_seizures,
                    "seizure_intervals": seizure_intervals
                })

    seiz_recs = [r for r in all_seizure_records if r["num_seizures"] > 0]
    print(f"[✓] Total recordings cataloged: {len(all_seizure_records)} ({len(seiz_recs)} with seizures)", flush=True)

    # 2. Extract EEG windows and provenance metadata in parallel
    all_window_tensors = []
    all_window_labels = []
    all_window_metadata = []
    
    print(f"\n[*] Step 2/4: Processing {len(seiz_recs)} seizure recordings in parallel ({max_workers} threads)...", flush=True)
    
    completed_cnt = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_rec, rec, base_url, raw_path): rec for rec in seiz_recs}
        for future in as_completed(futures):
            rec = futures[future]
            completed_cnt += 1
            tensors, labels, metadata = future.result()
            if tensors:
                all_window_tensors.extend(tensors)
                all_window_labels.extend(labels)
                all_window_metadata.extend(metadata)
            print(f" -> [{completed_cnt:03d}/{len(seiz_recs):03d}] Processed {rec['subject_id']}/{rec['filename']} (Extracted: {len(tensors)} windows | Cumulative: {len(all_window_tensors):,})", flush=True)

    print(f"\n[✓] Total raw EEG windows extracted: {len(all_window_tensors):,}", flush=True)

    # 3. Subsample non-seizure windows with fixed seed 2023
    print("\n[*] Step 3/4: Subsampling Non-Seizure Windows to match reference pool (~11,983 samples)...", flush=True)
    labels_arr = np.array(all_window_labels)
    seiz_indices = np.where(labels_arr == 1)[0]
    non_seiz_indices = np.where(labels_arr == 0)[0]
    
    print(f"    - Total Seizure Windows (Label 1):     {len(seiz_indices):,}", flush=True)
    print(f"    - Total Non-Seizure Windows (Label 0): {len(non_seiz_indices):,}", flush=True)
    
    rng = np.random.default_rng(seed=random_seed)
    target_non_seizure = min(8773, len(non_seiz_indices))
    kept_non_seiz_indices = rng.choice(non_seiz_indices, size=target_non_seizure, replace=False)
    
    selected_indices = np.concatenate([seiz_indices, kept_non_seiz_indices])
    rng.shuffle(selected_indices)
    
    final_X = np.array([all_window_tensors[idx] for idx in selected_indices], dtype=np.float64)
    final_y = np.array([all_window_labels[idx] for idx in selected_indices], dtype=np.float64)
    
    final_metadata = []
    for new_idx, orig_idx in enumerate(selected_indices):
        m = all_window_metadata[orig_idx].copy()
        m["sample_index"] = new_idx
        final_metadata.append(m)
        
    df_meta = pd.DataFrame(final_metadata)
    cols = ["sample_index", "subject_id", "session_filename", "window_start_sample_original_fs", "window_start_seconds", "label"]
    df_meta = df_meta[cols]

    # 4. Save NEW dataset and metadata
    print("\n[*] Step 4/4: Saving NEW Dataset and Provenance Metadata...", flush=True)
    x_path = out_path / "X.npy"
    y_path = out_path / "y.npy"
    meta_path = out_path / "window_metadata.csv"
    
    np.save(x_path, final_X)
    np.save(y_path, final_y)
    df_meta.to_csv(meta_path, index=False)
    
    print(f"  [✓] X.npy saved to:               {x_path.resolve()} (Shape: {final_X.shape}, dtype: {final_X.dtype})", flush=True)
    print(f"  [✓] y.npy saved to:               {y_path.resolve()} (Shape: {final_y.shape}, dtype: {final_y.dtype})", flush=True)
    print(f"  [✓] window_metadata.csv saved to: {meta_path.resolve()} (Rows: {len(df_meta)})", flush=True)
    
    return final_X, final_y, df_meta


if __name__ == "__main__":
    build_chbmit_provenance_dataset()
