"""
Platform-Aware Google Colab & Local Mac Smoke Test Script for ECG Research Project.

Verifies:
1. Python & PyTorch environment imports
2. Device detection (CUDA -> MPS -> CPU)
3. GPU identification (e.g. NVIDIA T4 when run on Colab)
4. Config and dataset path resolution
5. GenericHybrid1DBiCNNGRU model construction
6. 1 forward and 1 backward pass with CrossEntropyLoss(ignore_index=4)
7. Tensor shape & device consistency
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import get_device
from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU


def run_smoke_test():
    print("=================================================================")
    print("      ECG RESEARCH PROJECT — PLATFORM-AWARE SMOKE TEST           ")
    print("=================================================================")

    # 1. Imports Verification
    print("\n[*] Step 1: Verifying Python Package Imports...")
    try:
        import wfdb
        import psutil
        import seaborn
        import sklearn
        print(f"    [✓] wfdb version:       {wfdb.__version__}")
        import pandas as pd
        print(f"    [✓] pandas version:     {pd.__version__}")
        import numpy as np
        print(f"    [✓] numpy version:      {np.__version__}")
        import sklearn
        print(f"    [✓] scikit-learn ver:   {sklearn.__version__}")
        print("    [✓] All required non-PyTorch packages imported successfully.")
    except Exception as e:
        print(f"    [!] Failed importing dependencies: {e}")
        sys.exit(1)

    # 2. PyTorch & Device Detection
    print("\n[*] Step 2: Verifying PyTorch Environment & Device Selection...")
    device = get_device()

    if device.type == "cuda":
        print("    [✓] CUDA Environment Verified (Google Colab / NVIDIA GPU).")
    elif device.type == "mps":
        print("    [✓] MPS Environment Verified (Apple Silicon Mac).")
    else:
        print("    [✓] CPU Environment Verified (Fallback Mode).")

    # 3. Path Resolution & Dataset Path Check
    print("\n[*] Step 3: Verifying Configuration & Dataset Paths...")
    cfg_path = PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        print(f"    [!] config.yaml missing at '{cfg_path}'")
        sys.exit(1)

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    env_data_dir = os.getenv("ECG_DATA_DIR")
    if env_data_dir:
        data_dir = Path(env_data_dir)
    else:
        raw_dir = Path(config["data"]["data_dir"])
        data_dir = raw_dir if raw_dir.is_absolute() else (PROJECT_ROOT / raw_dir)

    print(f"    - Config Path:  {cfg_path}")
    print(f"    - Dataset Path: {data_dir}")

    if data_dir.exists():
        print("    [✓] Dataset path exists.")
    else:
        print(f"    [!] Note: Dataset directory does not exist at '{data_dir}'.")
        print("        (Set 'ECG_DATA_DIR' environment variable if using Google Drive or custom location).")

    # 4. Model Construction
    print("\n[*] Step 4: Instantiating GenericHybrid1DBiCNNGRU Architecture...")
    model = GenericHybrid1DBiCNNGRU(
        in_channels=1,
        cnn_channels=[64, 128, 128],
        kernel_sizes=[5, 5, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.2076,
        num_classes=5
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    [✓] Model initialized cleanly on device '{device}'.")
    print(f"    [✓] Total Trainable Parameters: {total_params:,} (Expected: 257,541)")
    assert total_params == 257541, f"Parameter count mismatch! Expected 257,541, got {total_params}"

    # 5. Dummy Forward & Backward Pass
    print("\n[*] Step 5: Executing 1 Forward and Backward Pass (Batch Size: 128)...")
    batch_size = 128
    seq_len = 256
    dummy_input = torch.randn(batch_size, 1, seq_len, device=device)
    dummy_targets = torch.randint(0, 4, (batch_size,), device=device)

    criterion = nn.CrossEntropyLoss(ignore_index=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001184, weight_decay=7.114476e-4)

    # Forward pass
    model.train()
    optimizer.zero_grad()
    outputs = model(dummy_input)

    assert outputs.shape == (batch_size, 5), f"Output shape mismatch! Expected ({batch_size}, 5), got {outputs.shape}"
    assert outputs.device.type == device.type, f"Output device type mismatch! Expected {device.type}, got {outputs.device.type}"

    loss = criterion(outputs, dummy_targets)
    print(f"    [✓] Forward Pass Output Shape: {list(outputs.shape)}")
    print(f"    [✓] Loss Value:                {loss.item():.4f}")

    # Backward pass
    loss.backward()
    optimizer.step()
    print("    [✓] Backward Pass & Optimizer Step Executed Cleanly.")

    print("\n=================================================================")
    print("      [✓] PLATFORM-AWARE SMOKE TEST PASSED SUCCESSFULLY!          ")
    print("=================================================================")


if __name__ == "__main__":
    run_smoke_test()
