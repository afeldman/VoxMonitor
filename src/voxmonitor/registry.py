"""Registry integration for VoxMonitor with DeepSuite HeadRegistry.

Provides a factory to build the VoxMonitor Lightning module and
registers it under a stable key for reuse in DeepSuite pipelines.
"""

from __future__ import annotations
from typing import Dict, Any

from deepsuite.registry import HeadRegistry
from voxmonitor.lightning import VoxMonitorLightningModule


def build_voxmonitor_module(
    num_classes: Dict[str, int],
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    embed_dim: int = 128,
    **kwargs: Any,
) -> VoxMonitorLightningModule:
  """Factory: Create VoxMonitor LightningModule.

  Args:
    num_classes: Mapping of task → number of classes.
    lr: Learning rate.
    weight_decay: L2 regularization weight.
    embed_dim: Backbone embedding dimension.
    **kwargs: Additional arguments passed to VoxMonitorLightningModule.
  """
  return VoxMonitorLightningModule(
      num_classes=num_classes,
      lr=lr,
      weight_decay=weight_decay,
      embed_dim=embed_dim,
      **kwargs,
  )


def register_voxmonitor_head() -> bool:
  """Register VoxMonitor in the global DeepSuite HeadRegistry.

  Returns:
    True if registration succeeded, else False.
  """
  # Stable key for reuse across projects
  key = "voxmonitor.audio.multitask"

  try:
    # Register the factory function directly using DeepSuite's register decorator
    if key not in HeadRegistry._registry:
      HeadRegistry._registry[key] = build_voxmonitor_module
    return True
  except Exception as e:
    print(f"Failed to register VoxMonitor: {e}")
    return False


# Attempt auto-registration on import
register_voxmonitor_head()

# Re-export DeepSuite HeadRegistry for convenience
__all__ = ["HeadRegistry", "build_voxmonitor_module", "register_voxmonitor_head"]
