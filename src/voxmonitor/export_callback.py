"""Custom export callbacks for ONNX and TorchScript with PyTorch 2.6 compatibility."""

import logging
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class CustomExportCallback(Callback):
    """Export best model to ONNX and TorchScript after training completes.
    
    Fixes PyTorch 2.6 weights_only issue by using weights_only=False.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        export_formats: list = None,
        input_shape: tuple = (1, 1, 64, 256),  # (batch, channels, n_mels, time_frames)
    ):
        """Initialize export callback.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            export_formats: List of formats to export (onnx, torchscript)
            input_shape: Shape for dummy mel-spectrogram (batch, channels, n_mels, time_frames)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.export_formats = export_formats or ["onnx", "torchscript"]
        self.input_shape = input_shape

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Export model after training completes."""
        # Find best checkpoint
        best_ckpt = self._find_best_checkpoint()
        
        if best_ckpt is None:
            logger.warning("No checkpoint found for export")
            return

        logger.info(f"🔹 Exporting model from checkpoint: {best_ckpt}")

        try:
            # Load model state with weights_only=False to avoid PyTorch 2.6 issues
            state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            logger.info("✅ Successfully loaded checkpoint")
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return

        # Export to requested formats
        if "onnx" in self.export_formats:
            self._export_onnx(pl_module, best_ckpt)

        if "torchscript" in self.export_formats:
            self._export_torchscript(pl_module, best_ckpt)

    def _find_best_checkpoint(self) -> Optional[Path]:
        """Find the best (most recent) checkpoint."""
        ckpt_dir = self.checkpoint_dir / "tensorboard"
        
        # Find all version directories
        versions = sorted(
            [d for d in ckpt_dir.glob("version_*") if d.is_dir()],
            key=lambda x: int(x.name.split("_")[1]),
            reverse=True,
        )

        if not versions:
            return None

        # Check most recent version
        latest_version = versions[0]
        ckpt_files = list((latest_version / "checkpoints").glob("*.ckpt"))

        if not ckpt_files:
            return None

        # Return most recent checkpoint (highest epoch)
        return sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1]

    def _export_onnx(self, pl_module: pl.LightningModule, checkpoint_path: Path) -> None:
        """Export model to ONNX format."""
        try:
            output_path = self.checkpoint_dir / f"model_{checkpoint_path.stem}.onnx"
            
            # Move model to CPU for export (avoid device mismatch)
            pl_module.cpu()
            pl_module.eval()
            
            # Create dummy input for ONNX export on CPU
            dummy_input = torch.randn(self.input_shape, device="cpu")
            
            # Export to ONNX
            torch.onnx.export(
                pl_module,
                dummy_input,
                str(output_path),
                input_names=["audio_input"],
                output_names=["age_output", "sex_output", "valence_output", "context_output"],
                dynamic_axes={
                    "audio_input": {0: "batch_size", 2: "time_steps"},
                    "age_output": {0: "batch_size"},
                    "sex_output": {0: "batch_size"},
                    "valence_output": {0: "batch_size"},
                    "context_output": {0: "batch_size"},
                },
                opset_version=14,
                verbose=False,
            )
            logger.info(f"✅ ONNX export successful: {output_path}")
        except Exception as e:
            logger.error(f"❌ ONNX export failed: {e}")

    def _export_torchscript(self, pl_module: pl.LightningModule, checkpoint_path: Path) -> None:
        """Export model to TorchScript format."""
        try:
            output_path = self.checkpoint_dir / f"model_{checkpoint_path.stem}.pt"
            
            # Move model to CPU for export (avoid device mismatch)
            pl_module.cpu()
            pl_module.eval()
            
            # Create dummy input on CPU
            dummy_input = torch.randn(self.input_shape, device="cpu")
            
            # Trace the model with strict=False to allow dict outputs
            scripted_model = torch.jit.trace(pl_module, dummy_input, strict=False)
            
            # Save TorchScript model
            scripted_model.save(str(output_path))
            logger.info(f"✅ TorchScript export successful: {output_path}")
        except Exception as e:
            logger.error(f"❌ TorchScript export failed: {e}")
