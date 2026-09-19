import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple, Any, Optional


class EEGTransferGRUModel(nn.Module):
    """
    EEG Seizure Detection Model leveraging Pretrained ECG BiGRU Weights.
    
    Architecture:
    1. EEG-Specific CNN (Trained from Scratch):
       Processes 18-channel EEG sequences [B, 18, 1024] -> [B, 128, 128].
    2. Transferred ECG BiGRU (Pretrained Weights):
       2-layer Bidirectional GRU (input_size=128, hidden_size=64, num_layers=2)
       transferred strictly from ECG checkpoint 'model_c_medium_augmented_best.pth'.
    3. Global Temporal Average Pooling:
       Averages along sequence dimension T -> [B, 128].
    4. New EEG Binary Classifier Head (Trained from Scratch):
       Linear(128 -> 2) for binary seizure detection (0: non-seizure, 1: seizure).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = "checkpoints/model_c_medium_augmented_best.pth",
        num_eeg_channels: int = 18,
        num_target_classes: int = 2,
        freeze_gru: bool = False,
        dropout: float = 0.2076
    ):
        super().__init__()

        self.num_eeg_channels = num_eeg_channels
        self.num_target_classes = num_target_classes
        self.freeze_gru = freeze_gru

        # 1. EEG-Specific 3-Stage 1D CNN (Initialized from Scratch)
        # Stage 1: (18 channels -> 64 filters)
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_eeg_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Stage 2: (64 filters -> 128 filters)
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Stage 3: (128 filters -> 128 filters)
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.eeg_cnn = nn.Sequential(self.conv1, self.conv2, self.conv3)

        # 2. Transferred 2-Layer Bidirectional GRU (Matching ECG GRU Architecture)
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # 3. New EEG Binary Classification Head (Initialized from Scratch)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128, num_target_classes)
        )

        # 4. Load Pretrained ECG BiGRU Weights (STRICTLY GRU ONLY)
        self.transferred_gru_params_count = 0
        if checkpoint_path:
            self.transferred_gru_params_count = self.load_pretrained_ecg_gru(checkpoint_path)

        # 5. Configure GRU Gradient Freezing Status
        self.set_gru_freeze_status(freeze_gru)

    def load_pretrained_ecg_gru(self, checkpoint_path: str) -> int:
        """
        Extracts and loads ONLY GRU parameter weights from the ECG model checkpoint.
        Strictly excludes all ECG CNN and ECG Classifier weights.
        """
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            print(f"[!] Warning: ECG checkpoint '{ckpt_file}' not found. GRU initialized randomly.")
            return 0

        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Filter ONLY keys starting with 'gru.'
        gru_state_dict = {}
        for key, tensor in state_dict.items():
            if key.startswith("gru."):
                # Remove 'gru.' prefix for self.gru.load_state_dict
                sub_key = key[4:]
                gru_state_dict[sub_key] = tensor

        if not gru_state_dict:
            print(f"[!] Warning: No 'gru.' keys found in '{ckpt_file}'. GRU initialized randomly.")
            return 0

        self.gru.load_state_dict(gru_state_dict)
        param_count = sum(t.numel() for t in gru_state_dict.values())
        print(f"[✓] EEGTransferGRUModel: Successfully loaded {len(gru_state_dict)} GRU tensors ({param_count:,} parameters) from '{ckpt_file}'.")
        print("    - ECG CNN weights transferred:        0")
        print("    - ECG Classifier weights transferred: 0")
        return param_count

    def set_gru_freeze_status(self, freeze_gru: bool):
        """Sets requires_grad status on GRU parameters."""
        self.freeze_gru = freeze_gru
        for param in self.gru.parameters():
            param.requires_grad = not freeze_gru
        
        status_str = "FROZEN (Locked)" if freeze_gru else "TRAINABLE (Fine-Tuning)"
        print(f"[✓] EEGTransferGRUModel: Transferred GRU status set to: {status_str}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle 4D [B, 18, 1024, 1] or 3D [B, 18, 1024]
        if x.dim() == 4:
            x = x.squeeze(-1)
        elif x.dim() == 2:
            x = x.unsqueeze(0)

        # 1. EEG-Specific CNN Feature Extractor
        cnn_features = self.eeg_cnn(x)  # Shape: [B, 128, 128]

        # 2. Permute for Sequence Input: [B, 128, 128] -> [B, 128 (seq_len), 128 (features)]
        seq_features = cnn_features.permute(0, 2, 1)

        # 3. Transferred BiGRU Temporal Encoder
        gru_out, _ = self.gru(seq_features)  # Shape: [B, 128, 128]

        # 4. Global Temporal Average Pooling
        pooled = torch.mean(gru_out, dim=1)  # Shape: [B, 128]

        # 5. New EEG Binary Classifier
        logits = self.classifier(pooled)  # Shape: [B, num_target_classes]
        return logits


if __name__ == "__main__":
    model = EEGTransferGRUModel()
    dummy_input = torch.randn(2, 18, 1024, 1)
    out = model(dummy_input)
    print("Dummy Forward Pass Output Shape:", out.shape)
