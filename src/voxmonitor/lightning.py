"""Lightning modules for training VoxMonitor models."""

from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# Direct import to avoid __init__ issues
from deepsuite.lightning_base.module import BaseModule
from voxmonitor.model import MultiTaskAudioModule


class VoxMonitorLightningModule(BaseModule):
  """Lightning module for multi-task pig vocalization classification.
  
  Args:
    num_classes: Dict mapping task name to number of classes.
    lr: Learning rate.
    weight_decay: L2 regularization weight.
    embed_dim: Embedding dimension.
  """

  def __init__(
      self,
      num_classes: Dict[str, int],
      lr: float = 1e-3,
      weight_decay: float = 1e-5,
      embed_dim: int = 128,
      **kwargs: Any,
  ) -> None:
    # Use a dummy loss function that will be overridden
    # Use 'mel' from DeepSuite as placeholder (we'll override in training_step)
    super().__init__(
        loss_name='mel',  # Use DeepSuite's mel loss as placeholder
        optimizer=torch.optim.Adam,
        metrics=[],  # Custom per-task metrics
        num_classes=max(num_classes.values()),  # Max classes for metric init
        **kwargs,
    )
    
    # VoxMonitor specific attributes
    self.model = MultiTaskAudioModule(num_classes=num_classes, embed_dim=embed_dim)
    self.num_classes = num_classes
    self.weight_decay = weight_decay
    self.label_tasks = list(num_classes.keys())
    self.learning_rate = lr
    
    # Save hyperparameters
    self.save_hyperparameters()

  def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
    """Override to handle dict labels correctly."""
    x, y = batch
    x = x.to(self.device)
    # y is a dict, move each tensor to device
    if isinstance(y, dict):
      y = {k: v.to(self.device) for k, v in y.items()}
    else:
      y = y.to(self.device)
    return x, y

  def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Forward pass returning per-task logits."""
    return self.model(waveform)

  def configure_optimizers(self) -> Any:
    """Configure optimizer."""
    return torch.optim.Adam(
        self.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay,
    )

  def training_step(
      self, batch: tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int
  ) -> torch.Tensor:
    """Compute loss on training batch."""
    waveform, labels = batch

    outputs = self(waveform)
    losses = {}
    for task in self.label_tasks:
      task_loss = F.cross_entropy(outputs[task], labels[task])
      losses[task] = task_loss
      self.log(f"train_loss_{task}", task_loss, on_step=False, on_epoch=True)

    total_loss = sum(losses.values()) / len(losses)
    self.log("train_loss", total_loss, on_step=False, on_epoch=True)
    return total_loss

  def validation_step(
      self, batch: tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int
  ) -> None:
    """Compute metrics on validation batch."""
    waveform, labels = batch

    outputs = self(waveform)
    losses = {}
    for task in self.label_tasks:
      task_loss = F.cross_entropy(outputs[task], labels[task])
      losses[task] = task_loss
      self.log(f"val_loss_{task}", task_loss, on_step=False, on_epoch=True)

      preds = outputs[task].argmax(dim=1)
      acc = (preds == labels[task]).float().mean()
      self.log(f"val_acc_{task}", acc, on_step=False, on_epoch=True)

    total_loss = sum(losses.values()) / len(losses)
    self.log("val_loss", total_loss, on_step=False, on_epoch=True)
