"""
Device selection utility for ECG research project.
Provides platform-aware device selection with priority: CUDA -> MPS -> CPU.
"""

import torch


def get_device() -> torch.device:
    """
    Returns the optimal PyTorch device based on hardware availability:
    1. CUDA (NVIDIA GPU, e.g. Colab T4)
    2. MPS (Apple Silicon GPU)
    3. CPU (Fallback)
    
    Prints device details at runtime.
    """
    print(f"[*] PyTorch Version: {torch.__version__}")
    print(f"[*] CUDA Available:  {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[*] Selected Device: {device}")
        print(f"[*] GPU Name:        {gpu_name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[*] Selected Device: {device} (Apple Silicon Acceleration)")
    else:
        device = torch.device("cpu")
        print(f"[*] Selected Device: {device}")

    return device
