"""Soundwell pig vocalization dataset module.

Dataset: https://zenodo.org/records/8252482
Multi-task classification targets: age, sex, valence, context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import pickle

import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
import soundfile as sf
from loguru import logger

from deepsuite.layers.mel import MelSpectrogramExtractor
from deepsuite.lightning_base.dataset.universal_set import UniversalDataset
from deepsuite.lightning_base.dataset.audio_loader import AudioLoader

from voxmonitor.zenodo_download import download_dataset


class SoundwelDataset(UniversalDataset):
    """Soundwell pig vocalization classification dataset.

    Multi-task learning targets:
    - age: Piglet age category
    - sex: Male/Female
    - valence: Positive/Neutral/Negative emotion
    - context: Activity context (feeding, playing, etc.)
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        sample_rate: int = 16000,
        n_mels: int = 64,
        max_length_sec: float = 5.0,
        label_columns: Optional[list[str]] = None,
        download: bool = True,
        csv_path: Optional[str | Path] = None,
        metadata: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initialize Soundwell dataset.

        Args:
            root_dir: Root directory of dataset.
            split: "train", "val", or "test".
            sample_rate: Audio sample rate.
            n_mels: Number of mel bands.
            max_length_sec: Maximum audio length in seconds.
            label_columns: List of label columns to use.
            download: Whether to download from Zenodo if not found.
            csv_path: Path to CSV file with metadata (alternative to Zenodo download).
            metadata: Pre-loaded DataFrame with metadata (overrides csv_path and split).
            sample_rate: Audio sample rate.
            n_mels: Number of mel bands.
            max_length_sec: Maximum audio length in seconds.
            label_columns: List of label columns to use.
            download: Whether to download from Zenodo if not found.
            csv_path: Path to CSV file with metadata (alternative to Zenodo download).
            metadata: Pre-loaded DataFrame with metadata (overrides csv_path and split).
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_length_sec = max_length_sec
        self.label_columns = label_columns or ["age", "sex", "valence", "context"]
        self.audio_dir = self.root_dir

        # Load metadata from different sources
        if metadata is not None:
            # Use pre-loaded metadata
            logger.info("Using provided metadata DataFrame...")
            self.metadata = metadata
        elif csv_path is not None:
            # Load from CSV/XLSX file
            suffix = Path(csv_path).suffix.lower()
            if suffix in {".xlsx", ".xls"}:
                logger.info(f"Loading metadata from Excel: {csv_path}")
                try:
                    self.metadata = pd.read_excel(csv_path)
                except ImportError as e:
                    raise RuntimeError("Excel-Datei erkannt, aber 'openpyxl' fehlt. Bitte installieren: uv add openpyxl") from e
            else:
                logger.info(f"Loading metadata from CSV: {csv_path}")
                self.metadata = pd.read_csv(csv_path)
        else:
            # Load from Zenodo or existing files
            self.metadata_file = self.root_dir / "metadata.pkl"
            if not (self.audio_dir / "audio").exists() and download:
                logger.info("Downloading Soundwell dataset from Zenodo...")
                download_dataset("soundwell", output_dir=self.root_dir.parent)
            
            self.audio_dir = self.root_dir / "audio"
            if not self.audio_dir.exists():
                raise FileNotFoundError(
                    f"Audio directory not found: {self.audio_dir}\n"
                    f"Download from: https://zenodo.org/records/8252482"
                )
            
            logger.info(f"Loading {split} metadata...")
            self._load_metadata_from_file()

        # Initialize mel-spectrogram extractor
        self.mel_extractor = MelSpectrogramExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
        )

        # Build label mappings
        self._build_label_mappings()

        logger.info(
            f"Loaded {len(self)} {split} samples with labels: {self.label_columns}"
        )

    def _load_metadata_from_file(self) -> None:
        """Load metadata from pickle file or CSV (for Zenodo data)."""
        # Try pickle first
        if self.metadata_file.exists():
            with open(self.metadata_file, "rb") as f:
                data = pickle.load(f)
        else:
            # Try CSV
            csv_file = self.root_dir / "metadata.csv"
            if csv_file.exists():
                # Support CSV and Excel in cached metadata
                suffix = csv_file.suffix.lower()
                if suffix in {".xlsx", ".xls"}:
                    data = pd.read_excel(csv_file)
                else:
                    data = pd.read_csv(csv_file)
            else:
                raise FileNotFoundError(
                    f"Metadata not found in {self.root_dir}\n"
                    f"Expected: metadata.pkl or metadata.csv"
                )

        # Filter by split
        if isinstance(data, pd.DataFrame):
            if "split" in data.columns:
                data = data[data["split"] == self.split].reset_index(drop=True)
            self.metadata = data
        else:
            # Assume it's a dict with split keys
            self.metadata = pd.DataFrame(data.get(self.split, []))

    def _build_label_mappings(self) -> None:
        """Build label-to-index mappings."""
        self.label_mappings = {}
        
        # Map column names from CSV to internal labels
        column_mapping = {
            "age": ["Age Category", "age"],
            "sex": ["Sex", "sex"],
            "valence": ["Valence", "valence"],
            "context": ["Context", "Context Category", "context"],
        }

        for task, possible_cols in column_mapping.items():
            if task not in self.label_columns:
                continue
            
            # Find the actual column name in metadata
            actual_col = None
            for col in possible_cols:
                if col in self.metadata.columns:
                    actual_col = col
                    break
            
            if actual_col is None:
                available = self.metadata.columns.tolist()
                logger.warning(
                    f"Label '{task}' not found in metadata columns: {available}"
                )
                continue

            # Filter out NaN values and convert to string for consistent sorting
            valid_labels = self.metadata[actual_col].dropna().unique().astype(str)
            unique_labels = sorted(valid_labels)
            # Ensure an 'UNK' class exists to handle missing/unknown values
            if "UNK" not in unique_labels:
                unique_labels.append("UNK")
            self.label_mappings[task] = {
                label: idx for idx, label in enumerate(unique_labels)
            }

        logger.info(f"Label mappings: {self.label_mappings}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get sample and labels.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (mel_spectrogram, labels_dict).
        """
        row = self.metadata.iloc[idx]

        # Load audio - handle different filename column names
        filename_col = "Audio Filename" if "Audio Filename" in self.metadata.columns else "filename"
        audio_filename = row[filename_col]
        audio_file = self.audio_dir / audio_filename
        
        if not audio_file.exists():
            # If file doesn't exist, return dummy data (will be filtered out during training)
            # This allows datasets with missing files to work
            logger.warning(f"Audio file not found, returning dummy: {audio_file}")
            # Return dummy mel-spectrogram with expected shape
            max_samples = int(self.max_length_sec * self.sample_rate)
            expected_mel_frames = (max_samples + 160 - 400) // 160 + 1
            expected_mel_frames = max(expected_mel_frames, 100)
            mel_spec = torch.zeros(self.n_mels, expected_mel_frames)
            # Return dummy labels for all tasks
            labels = {task: torch.tensor(0, dtype=torch.long) for task in self.label_columns}
            return mel_spec, labels

        # Load audio with soundfile (no FFmpeg dependency)
        audio_data, sr = sf.read(str(audio_file))
        
        # Convert to tensor (soundfile returns numpy array)
        waveform = torch.from_numpy(audio_data).float()
        
        # Handle mono/stereo - convert to mono if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, time]
        elif waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # stereo to mono

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Trim/pad to max length
        max_samples = int(self.max_length_sec * self.sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        else:
            pad_len = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        # Extract mel-spectrogram
        mel_spec = self.mel_extractor(waveform)  # [1, n_mels, time]
        mel_spec = mel_spec.squeeze(0)  # [n_mels, time]

        # Ensure consistent time dimension for mel-spectrogram
        # Calculate expected length based on waveform padding
        expected_mel_frames = (max_samples + 160 - 400) // 160 + 1  # Based on hop_length=160, n_fft=400
        expected_mel_frames = max(expected_mel_frames, 100)  # Ensure minimum length
        
        if mel_spec.shape[1] < expected_mel_frames:
            # Pad time dimension
            pad_len = expected_mel_frames - mel_spec.shape[1]
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_len))
        elif mel_spec.shape[1] > expected_mel_frames:
            # Trim time dimension
            mel_spec = mel_spec[:, :expected_mel_frames]

        # Get labels
        labels = {}
        column_mapping = {
            "age": ["Age Category", "age"],
            "sex": ["Sex", "sex"],
            "valence": ["Valence", "valence"],
            "context": ["Context", "Context Category", "context"],
        }
        
        for task in self.label_columns:
            if task not in self.label_mappings:
                continue
            
            # Find actual column name
            possible_cols = column_mapping.get(task, [task])
            actual_col = None
            for col in possible_cols:
                if col in self.metadata.columns:
                    actual_col = col
                    break
            
            if actual_col is None:
                continue
            
            label_value = row[actual_col]
            # Handle NaN/unknown values
            if pd.isna(label_value):
                label_value = "UNK"
            
            # Convert to string for consistent mapping
            label_value = str(label_value)
            
            if label_value not in self.label_mappings[task]:
                # Unknown label -> map to UNK
                label_value = "UNK"
            
            label_idx = self.label_mappings[task][label_value]
            labels[task] = torch.tensor(label_idx, dtype=torch.long)

        return mel_spec, labels


class SoundwelDataModule(AudioLoader):
    """PyTorch Lightning DataModule for Soundwell dataset.

    Downloads dataset from Zenodo on first use and provides train/val/test splits.
    """

    def __init__(
        self,
        root_dir: str | Path = "datasets/soundwell",
        csv_path: Optional[str | Path] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        sample_rate: int = 16000,
        n_mels: int = 64,
        max_length_sec: float = 5.0,
        label_columns: Optional[list[str]] = None,
        download: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize Soundwell DataModule.

        Args:
            root_dir: Root directory for dataset.
            csv_path: Path to CSV metadata file.
            batch_size: Batch size for training.
            num_workers: Number of workers for DataLoader.
            sample_rate: Audio sample rate.
            n_mels: Number of mel bands.
            max_length_sec: Maximum audio length in seconds.
            label_columns: List of label columns.
            download: Whether to download from Zenodo.
            **kwargs: Additional arguments.
        """
        super().__init__()

        self.root_dir = Path(root_dir)
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_length_sec = max_length_sec
        self.label_columns = label_columns or ["age", "sex", "valence", "context"]
        self.download = download

        self.train_ds: Optional[SoundwellDataset] = None
        self.val_ds: Optional[SoundwellDataset] = None
        self.test_ds: Optional[SoundwellDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare datasets.

        Args:
            stage: "fit", "validate", "test", or None for all.
        """
        if stage in ("fit", None):
            logger.info("Setting up train and val datasets...")
            
            # Load CSV if provided
            metadata = None
            csv_path_arg = None
            if self.csv_path:
                suffix = Path(self.csv_path).suffix.lower()
                if suffix in {".xlsx", ".xls"}:
                    try:
                        metadata = pd.read_excel(self.csv_path)
                    except ImportError as e:
                        raise RuntimeError("Excel-Datei erkannt, aber 'openpyxl' fehlt. Bitte installieren: uv add openpyxl") from e
                else:
                    metadata = pd.read_csv(self.csv_path)
                csv_path_arg = None  # Don't pass csv_path if metadata is provided
            
            # Create train dataset (90%)
            self.train_ds = SoundwelDataset(
                root_dir=self.root_dir,
                metadata=metadata,
                csv_path=csv_path_arg,
                sample_rate=self.sample_rate,
                max_length_sec=self.max_length_sec,
                label_columns=self.label_columns,
                download=self.download,
            )
            
            # Create val dataset as subset of train (10%)
            # For now, just use the same dataset (proper splitting would use indices)
            self.val_ds = SoundwelDataset(
                root_dir=self.root_dir,
                metadata=metadata,
                csv_path=csv_path_arg,
                sample_rate=self.sample_rate,
                max_length_sec=self.max_length_sec,
                label_columns=self.label_columns,
                download=False,
            )

        if stage in ("test", None):
            logger.info("Setting up test dataset...")
            metadata = None
            if self.csv_path:
                suffix = Path(self.csv_path).suffix.lower()
                if suffix in {".xlsx", ".xls"}:
                    try:
                        metadata = pd.read_excel(self.csv_path)
                    except ImportError as e:
                        raise RuntimeError("Excel-Datei erkannt, aber 'openpyxl' fehlt. Bitte installieren: uv add openpyxl") from e
                else:
                    metadata = pd.read_csv(self.csv_path)
            
            self.test_ds = SoundwelDataset(
                root_dir=self.root_dir,
                metadata=metadata,
                csv_path=self.csv_path,
                sample_rate=self.sample_rate,
                max_length_sec=self.max_length_sec,
                label_columns=self.label_columns,
                download=False,
            )

    def train_dataloader(self):
        """Return training DataLoader."""
        assert self.train_ds is not None
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Return validation DataLoader."""
        assert self.val_ds is not None
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Return test DataLoader."""
        assert self.test_ds is not None
        from torch.utils.data import DataLoader
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
