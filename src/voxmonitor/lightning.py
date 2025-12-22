"""Lightning modules for training VoxMonitor models."""

from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from deepsuite.lightning_base import BaseTrainer
from voxmonitor.model import MultiTaskAudioModule


class VoxMonitorLightningModule(BaseTrainer):
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
  ) -> None:
    super().__init__(
        module=MultiTaskAudioModule(num_classes=num_classes, embed_dim=embed_dim),
        loss_fn=None,
        learning_rate=lr,
    )
    self.num_classes = num_classes
    self.weight_decay = weight_decay
    self.label_tasks = list(num_classes.keys())

  def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Forward pass returning per-task logits."""
    return self.module(waveform)

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
