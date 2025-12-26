#!/usr/bin/env python
"""Unit tests for VoxMonitorLightningModule."""

import pytest
import torch
from unittest.mock import MagicMock

from deepsuite.lightning_base import BaseModule
from voxmonitor.lightning import VoxMonitorLightningModule


class TestVoxMonitorLightningModule:
    """Test suite for VoxMonitorLightningModule."""

    @pytest.fixture
    def num_classes(self):
        """Test configuration."""
        return {"age": 4, "sex": 2, "valence": 3, "context": 5}

    @pytest.fixture
    def module(self, num_classes):
        """Create a module instance."""
        return VoxMonitorLightningModule(
            num_classes=num_classes,
            lr=1e-3,
            weight_decay=1e-5,
            embed_dim=128,
        )

    def test_initialization(self, module, num_classes):
        """Test module initialization."""
        assert isinstance(module, BaseModule)
        assert module.num_classes == num_classes
        assert module.learning_rate == 1e-3
        assert module.weight_decay == 1e-5
        assert module.label_tasks == ["age", "sex", "valence", "context"]
        print("✅ Initialization test passed")

    def test_model_exists(self, module):
        """Test that model is correctly initialized."""
        assert hasattr(module, "model")
        assert module.model is not None
        print("✅ Model existence test passed")

    def test_forward_shape(self, module, num_classes):
        """Test forward pass output shapes."""
        batch_size = 4
        sample_rate = 16000
        duration_sec = 3
        
        # Create dummy mel-spectrogram input [batch, channels, freq, time]
        mel_spec = torch.randn(batch_size, 1, 64, sample_rate // (16000 // 3))
        
        outputs = module(mel_spec)
        
        # Check that output is a dict
        assert isinstance(outputs, dict), "Output should be a dict"
        
        # Check all tasks are present
        for task, num_task_classes in num_classes.items():
            assert task in outputs, f"Task {task} not in outputs"
            # Shape should be [batch_size, num_classes]
            assert outputs[task].shape == (batch_size, num_task_classes), \
                f"Expected shape ({batch_size}, {num_task_classes}), got {outputs[task].shape}"
        
        print("✅ Forward pass test passed")

    def test_training_step(self, module):
        """Test training step."""
        batch_size = 4
        sample_rate = 16000
        duration_sec = 3
        
        # Create dummy batch
        waveform = torch.randn(batch_size, 1, 64, sample_rate // (16000 // 3))
        labels = {
            "age": torch.randint(0, 4, (batch_size,)),
            "sex": torch.randint(0, 2, (batch_size,)),
            "valence": torch.randint(0, 3, (batch_size,)),
            "context": torch.randint(0, 5, (batch_size,)),
        }
        batch = (waveform, labels)
        
        # Mock logger
        module.log = MagicMock()
        
        loss = module.training_step(batch, batch_idx=0)
        
        # Check that loss is a tensor
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.requires_grad, "Loss should require gradients"
        assert loss.item() > 0, "Loss should be positive"
        
        # Check that logging was called
        assert module.log.called, "log() should have been called"
        
        print("✅ Training step test passed")

    def test_validation_step(self, module):
        """Test validation step."""
        batch_size = 4
        sample_rate = 16000
        duration_sec = 3
        
        # Create dummy batch
        waveform = torch.randn(batch_size, 1, 64, sample_rate // (16000 // 3))
        labels = {
            "age": torch.randint(0, 4, (batch_size,)),
            "sex": torch.randint(0, 2, (batch_size,)),
            "valence": torch.randint(0, 3, (batch_size,)),
            "context": torch.randint(0, 5, (batch_size,)),
        }
        batch = (waveform, labels)
        
        # Mock logger
        module.log = MagicMock()
        
        module.validation_step(batch, batch_idx=0)
        
        # Check that logging was called
        assert module.log.called, "log() should have been called"
        
        # Check that correct metrics are logged
        logged_calls = [call[0][0] for call in module.log.call_args_list]
        
        # Should log loss and accuracy for each task, plus total loss
        assert any("val_loss_age" in call for call in logged_calls)
        assert any("val_acc_age" in call for call in logged_calls)
        assert any("val_loss" in call for call in logged_calls)
        
        print("✅ Validation step test passed")

    def test_configure_optimizers(self, module):
        """Test optimizer configuration."""
        optimizer = module.configure_optimizers()
        
        assert isinstance(optimizer, torch.optim.Adam), \
            f"Expected Adam optimizer, got {type(optimizer)}"
        
        # Check learning rate
        assert optimizer.param_groups[0]["lr"] == 1e-3
        
        # Check weight decay
        assert optimizer.param_groups[0]["weight_decay"] == 1e-5
        
        print("✅ Optimizer configuration test passed")

    def test_hyperparameters_saved(self, module):
        """Test that hyperparameters are saved."""
        assert hasattr(module, "hparams"), "Hyperparameters should be saved"
        print("✅ Hyperparameters saved test passed")

    def test_inheritance_from_basemodule(self, module):
        """Test that module correctly inherits from BaseModule."""
        assert isinstance(module, BaseModule), \
            "VoxMonitorLightningModule should inherit from BaseModule"
        print("✅ BaseModule inheritance test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
