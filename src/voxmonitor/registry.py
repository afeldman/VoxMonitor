"""Registry integration for VoxMonitor with DeepSuite HeadRegistry.

Provides a factory to build the VoxMonitor Lightning module and
registers it under a stable key for reuse in DeepSuite pipelines.
"""

from __future__ import annotations
from typing import Dict, Optional

import torch

try:
  from deepsuite.registry import HeadRegistry  # type: ignore
except Exception:  # pragma: no cover
  HeadRegistry = None  # fallback when deepsuite isn't available

from voxmonitor.lightning import VoxMonitorLightningModule


def build_voxmonitor_module(
    num_classes: Dict[str, int],
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    embed_dim: int = 128,
) -> VoxMonitorLightningModule:
  """Factory: Create VoxMonitor LightningModule.

  Args:
    num_classes: Mapping of task → number of classes.
    lr: Learning rate.
    weight_decay: L2 regularization weight.
    embed_dim: Backbone embedding dimension.
  """
  return VoxMonitorLightningModule(
      num_classes=num_classes,
      lr=lr,
      weight_decay=weight_decay,
      embed_dim=embed_dim,
  )


def register_voxmonitor_head() -> bool:
  """Register VoxMonitor in the global DeepSuite HeadRegistry.

  Returns:
    True if registration succeeded, else False.
  """
  if HeadRegistry is None:
    return False

  # Stable key for reuse across projects
  key = "voxmonitor.audio.multitask"

  try:
    HeadRegistry.register(key, build_voxmonitor_module)  # type: ignore[attr-defined]
    return True
  except Exception:
    return False


# Attempt auto-registration on import; non-fatal if unavailable.
try:
  register_voxmonitor_head()
except Exception:
  pass
