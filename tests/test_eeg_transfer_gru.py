import unittest
import torch
from pathlib import Path
from src.models.eeg_transfer_gru_model import EEGTransferGRUModel


class TestEEGTransferGRUModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = "checkpoints/model_c_medium_augmented_best.pth"
        cls.has_checkpoint = Path(cls.checkpoint_path).exists()

    def test_01_architecture_and_parameter_counts(self):
        """Verify model component shapes and exact parameter counts."""
        model = EEGTransferGRUModel(checkpoint_path=None)
        
        total_params = sum(p.numel() for p in model.parameters())
        cnn_params = sum(p.numel() for p in model.eeg_cnn.parameters())
        gru_params = sum(p.numel() for p in model.gru.parameters())
        classifier_params = sum(p.numel() for p in model.classifier.parameters())
        
        self.assertEqual(gru_params, 148992)
        self.assertEqual(cnn_params, 96832)
        self.assertEqual(classifier_params, 258)
        self.assertEqual(total_params, 246082)

    def test_02_weight_transfer_verification(self):
        """Verify loading pretrained ECG BiGRU weights strictly loads GRU parameters."""
        if not self.has_checkpoint:
            self.skipTest(f"Checkpoint {self.checkpoint_path} not available.")
            
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        ecg_state_dict = ckpt.get("model_state_dict", ckpt)
        
        model = EEGTransferGRUModel(checkpoint_path=self.checkpoint_path)
        
        # Verify exact matching of GRU tensors
        for name, param in model.gru.named_parameters():
            ckpt_key = f"gru.{name}"
            self.assertIn(ckpt_key, ecg_state_dict)
            self.assertTrue(torch.equal(param, ecg_state_dict[ckpt_key]), f"Mismatch in tensor {name}")
            
        # Assert that no CNN or classifier parameters were transferred
        self.assertEqual(model.transferred_gru_params_count, 148992)

    def test_03_forward_pass_and_tensor_integrity(self):
        """Test forward pass shapes, output logits, and absence of NaN/Inf values."""
        model = EEGTransferGRUModel(checkpoint_path=None)
        
        # 4D input [Batch, Channels, Time, Dummy]
        x_4d = torch.randn(4, 18, 1024, 1)
        out_4d = model(x_4d)
        self.assertEqual(out_4d.shape, (4, 2))
        self.assertFalse(torch.isnan(out_4d).any())
        self.assertFalse(torch.isinf(out_4d).any())
        
        # 3D input [Batch, Channels, Time]
        x_3d = torch.randn(4, 18, 1024)
        out_3d = model(x_3d)
        self.assertEqual(out_3d.shape, (4, 2))
        self.assertFalse(torch.isnan(out_3d).any())
        self.assertFalse(torch.isinf(out_3d).any())

    def test_04_gradient_and_freeze_verification(self):
        """Verify GRU parameter freeze status and gradient flow."""
        # Case 1: Frozen GRU
        model_frozen = EEGTransferGRUModel(checkpoint_path=None, freeze_gru=True)
        for param in model_frozen.gru.parameters():
            self.assertFalse(param.requires_grad)
        for param in model_frozen.eeg_cnn.parameters():
            self.assertTrue(param.requires_grad)
        for param in model_frozen.classifier.parameters():
            self.assertTrue(param.requires_grad)

        # Case 2: Unfrozen GRU (Fine-tuning)
        model_trainable = EEGTransferGRUModel(checkpoint_path=None, freeze_gru=False)
        for param in model_trainable.gru.parameters():
            self.assertTrue(param.requires_grad)
        for param in model_trainable.eeg_cnn.parameters():
            self.assertTrue(param.requires_grad)
        for param in model_trainable.classifier.parameters():
            self.assertTrue(param.requires_grad)


if __name__ == "__main__":
    unittest.main()
