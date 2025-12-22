"""VoxMonitor: Multi-task pig vocalization classification using DeepSuite.

This package provides:
- Mel-spectrogram feature extraction (librosa-compatible)
- Multi-task CNN backbone for audio classification
- PyTorch Lightning training pipeline
- DeepSuite integration for model registry and export

Example:
  from voxmonitor.data import SoundwelDataModule
  from voxmonitor.lightning import VoxMonitorLightningModule
  import pytorch_lightning as pl

  dm = SoundwelDataModule(
      root_dir="data/audio",
      key_xlsx="data/SoundwelDatasetKey.xlsx",
      label_columns=["age", "sex", "valence", "context"],
  )
  
  lit = VoxMonitorLightningModule(
      num_classes={"age": 4, "sex": 2, "valence": 3, "context": 5}
  )
  
  trainer = pl.Trainer(max_epochs=100)
  trainer.fit(lit, dm)
"""

__version__ = "0.2.0"

from voxmonitor.model import MultiTaskAudioModule, MelSpectrogramExtractor, AudioCNN
from voxmonitor.data import SoundwelDataset, SoundwelDataModule
from voxmonitor.lightning import VoxMonitorLightningModule

__all__ = [
    "MultiTaskAudioModule",
    "MelSpectrogramExtractor",
    "AudioCNN",
    "SoundwelDataset",
    "SoundwelDataModule",
    "VoxMonitorLightningModule",
]
