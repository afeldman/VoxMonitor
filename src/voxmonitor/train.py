"""Training script for VoxMonitor models using PyTorch Lightning.

Usage:
  python -m voxmonitor.train_pl --config config/config.yaml
  voxmonitor-train --config config/config.yaml
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from deepsuite.lightning_base.trainer import BaseTrainer

# PyTorch 2.6 weights_only fix: Allow mse_loss for checkpoint loading
try:
    torch.serialization.add_safe_globals([torch.nn.functional.mse_loss])
except Exception:
    pass  # torch version might not have add_safe_globals

from voxmonitor.data import SoundwelDataModule
from voxmonitor.lightning import VoxMonitorLightningModule
from voxmonitor.export_callback import CustomExportCallback


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(cfg_path: str) -> Dict:
  """Load YAML configuration file."""
  with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
  return cfg


def get_device(device_str: str) -> str:
  """Validate and return appropriate device."""
  if device_str.lower() == "auto":
    if torch.cuda.is_available():
      return "cuda"
    if torch.backends.mps.is_available():
      return "mps"
    return "cpu"
  return device_str.lower()


def main(
    config_path: str = "config/config.yaml",
    checkpoint_dir: Optional[str] = None,
    resume_ckpt: Optional[str] = None,
) -> None:
  """Train VoxMonitor model.
  
  Args:
    config_path: Path to YAML config file.
    checkpoint_dir: Override checkpoint directory from config.
    resume_ckpt: Path to checkpoint to resume from.
  """
  cfg = load_config(config_path)

  data_cfg = cfg.get("data", {})
  train_cfg = cfg.get("train", {})
  labels_cfg = cfg.get("labels", {})
  model_cfg = cfg.get("model", {})

  root_dir = data_cfg.get("root_dir", "data/audio")
  key_xlsx = data_cfg.get("key_xlsx", "data/SoundwelDatasetKey.xlsx")
  label_columns = labels_cfg.get("columns", ["age", "sex", "valence", "context"])
  sample_rate = data_cfg.get("sample_rate", 16000)
  val_fraction = data_cfg.get("val_fraction", 0.2)

  batch_size = train_cfg.get("batch_size", 32)
  num_workers = train_cfg.get("num_workers", 4)
  max_epochs = train_cfg.get("max_epochs", 100)
  learning_rate = train_cfg.get("lr", 1e-3)
  weight_decay = train_cfg.get("weight_decay", 1e-5)
  seed = train_cfg.get("seed", 42)
  device = get_device(train_cfg.get("device", "auto"))
  fast_dev_run = train_cfg.get("fast_dev_run", False)

  ckpt_dir = checkpoint_dir or train_cfg.get("checkpoint_dir", "ckpt/voxmonitor")
  os.makedirs(ckpt_dir, exist_ok=True)

  logger.info("Loading configuration...")
  logger.info(f"  Root dir: {root_dir}")
  logger.info(f"  Key xlsx: {key_xlsx}")
  logger.info(f"  Label columns: {label_columns}")
  logger.info(f"  Sample rate: {sample_rate}")
  logger.info(f"  Batch size: {batch_size}, LR: {learning_rate}, Max epochs: {max_epochs}")
  logger.info(f"  Device: {device}")

  pl.seed_everything(seed)

  logger.info("Initializing data module...")
  dm = SoundwelDataModule(
      root_dir=root_dir,
      csv_path=key_xlsx,
      label_columns=label_columns,
      batch_size=batch_size,
      num_workers=num_workers,
      sample_rate=sample_rate,
      val_fraction=val_fraction,
      seed=seed,
  )
  dm.setup("fit")

  logger.info(f"Train dataset size: {len(dm.train_ds)}")
  logger.info(f"Val dataset size: {len(dm.val_ds)}")

  num_classes = dm.train_ds.label_mappings
  logger.info(f"Num classes per task: {num_classes}")

  logger.info("Initializing model...")
  lit = VoxMonitorLightningModule(
      num_classes={k: len(v) for k, v in num_classes.items()},
      lr=learning_rate,
      weight_decay=weight_decay,
      embed_dim=model_cfg.get("embed_dim", 128),
  )

  logger.info("Setting up callbacks...")
  ckpt_cb = ModelCheckpoint(
      dirpath=ckpt_dir,
      filename="epoch-{epoch:02d}-val_loss-{val_loss:.3f}",
      monitor="val_loss",
      mode="min",
      save_top_k=3,
      verbose=True,
  )
  
  early_stop = EarlyStopping(
      monitor="val_loss",
      patience=train_cfg.get("patience", 10),
      mode="min",
      verbose=True,
  )

  # Get export formats from config (optional)
  export_formats = train_cfg.get("export_formats", ["onnx", "torchscript"])
  mlflow_experiment = train_cfg.get("mlflow_experiment", None)

  # Add custom export callback (handles PyTorch 2.6 weights_only issue)
  export_callback = CustomExportCallback(
      checkpoint_dir=ckpt_dir,
      export_formats=export_formats,
  )

  logger.info("Creating DeepSuite BaseTrainer...")
  # Important: Do not pass accelerator/devices here to avoid duplicate kwargs.
  # Disable DeepSuite's built-in export to use our custom callback instead
  trainer = BaseTrainer(
    max_epochs=max_epochs,
    log_dir=ckpt_dir,
    model_output_dir=ckpt_dir,
    early_stopping=early_stop,
    model_checkpoint=ckpt_cb,
    enable_progress_bar=True,
    export_formats=[],  # Disabled - using custom callback instead
    fast_dev_run=fast_dev_run,
    callbacks=[export_callback],  # Add custom export callback
  )

  logger.info("Starting training...")
  trainer.fit(lit, dm, ckpt_path=resume_ckpt)

  logger.info(f"Best model saved in {ckpt_dir}")

  ckpt_files = sorted(Path(ckpt_dir).glob("epoch-*.ckpt"))
  if ckpt_files:
    best_ckpt = ckpt_files[-1]
    logger.info(f"Best checkpoint: {best_ckpt}")

    export_path = Path(ckpt_dir) / "voxmonitor_final.pt"
    logger.info(f"Exporting model to {export_path}...")
    torch.save(lit.model.state_dict(), export_path)
    logger.info("Export complete!")
    
  if export_formats:
    logger.info(f"Models exported in formats: {', '.join(export_formats)}")
    logger.info(f"Export directory: {ckpt_dir}")


if __name__ == "__main__":
  import sys

  cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
  main(config_path=cfg_path)
