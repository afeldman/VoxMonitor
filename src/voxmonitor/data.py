"""Data loading and preprocessing for pig vocalization datasets."""

from typing import Optional, List, Dict, Tuple
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class SoundwelDataset(Dataset):
  """Pig vocalization dataset from Soundwel CSV/audio files.
  
  Args:
    root_dir: Root directory containing audio files.
    key_xlsx: Path to Excel file with metadata (audio_path, age, sex, valence, context).
    label_columns: List of column names to use as classification targets.
    sample_rate: Target sample rate.
    max_length_sec: Maximum audio duration in seconds.
    indices: Optional list of indices to use (for train/val split).
  """

  def __init__(
      self,
      root_dir: str,
      key_xlsx: str,
      label_columns: List[str],
      sample_rate: int = 16000,
      max_length_sec: float = 5.0,
      indices: Optional[List[int]] = None,
  ) -> None:
    super().__init__()
    self.root_dir = Path(root_dir)
    self.sample_rate = sample_rate
    self.max_length = int(max_length_sec * sample_rate)

    df = pd.read_excel(key_xlsx)
    if indices is not None:
      df = df.iloc[indices].reset_index(drop=True)

    self.df = df
    self.label_columns = label_columns
    self.label_mappings = {col: {v: i for i, v in enumerate(df[col].unique())} for col in label_columns}

  def __len__(self) -> int:
    return len(self.df)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load audio and labels.
    
    Returns:
      Tuple of (waveform, label_dict).
    """
    row = self.df.iloc[idx]
    audio_path = self.root_dir / row["audio_path"]

    try:
      waveform, sr = torchaudio.load(str(audio_path))
      if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
      if sr != self.sample_rate:
        waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
      waveform = waveform.squeeze(0)
    except Exception as e:
      print(f"[WARN] Failed to load {audio_path}: {e}. Using silence.")
      waveform = torch.zeros(self.max_length)

    if waveform.shape[0] < self.max_length:
      waveform = F.pad(waveform, (0, self.max_length - waveform.shape[0]))
    elif waveform.shape[0] > self.max_length:
      waveform = waveform[: self.max_length]

    labels = {col: self.label_mappings[col].get(row[col], 0) for col in self.label_columns}

    return waveform, labels


class SoundwelDataModule(pl.LightningDataModule):
  """Lightning DataModule for Soundwel dataset."""

  def __init__(
      self,
      root_dir: str,
      key_xlsx: str,
      label_columns: List[str],
      batch_size: int = 32,
      num_workers: int = 4,
      sample_rate: int = 16000,
      val_fraction: float = 0.2,
      seed: int = 42,
  ) -> None:
    super().__init__()
    self.root_dir = root_dir
    self.key_xlsx = key_xlsx
    self.label_columns = label_columns
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.sample_rate = sample_rate
    self.val_fraction = val_fraction
    self.seed = seed

  def setup(self, stage: str = "fit") -> None:
    """Prepare train/val datasets."""
    df = pd.read_excel(self.key_xlsx)
    n = len(df)
    val_size = int(n * self.val_fraction)
    rng = np.random.RandomState(self.seed)
    indices = rng.permutation(n)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    if stage in ("fit", "train"):
      self.train_ds = SoundwelDataset(
          root_dir=self.root_dir,
          key_xlsx=self.key_xlsx,
          label_columns=self.label_columns,
          sample_rate=self.sample_rate,
          indices=train_idx.tolist(),
      )

    if stage in ("fit", "validate", "val"):
      self.val_ds = SoundwelDataset(
          root_dir=self.root_dir,
          key_xlsx=self.key_xlsx,
          label_columns=self.label_columns,
          sample_rate=self.sample_rate,
          indices=val_idx.tolist(),
      )

  def train_dataloader(self) -> DataLoader:
    """Return training dataloader."""
    return DataLoader(
        self.train_ds,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=self.num_workers,
        persistent_workers=True,
    )

  def val_dataloader(self) -> DataLoader:
    """Return validation dataloader."""
    return DataLoader(
        self.val_ds,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=self.num_workers,
        persistent_workers=True,
    )
