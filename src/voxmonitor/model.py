"""Audio models for pig vocalization classification and synthesis."""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepsuite.layers.mel import MelSpectrogramExtractor


class ResidualBlock(nn.Module):
  """Simple residual block."""
  
  def __init__(self, in_channels: int, out_channels: int) -> None:
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    
    # Skip connection
    self.skip = (in_channels == out_channels)
    if not self.skip:
      self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x
    x = F.relu(self.bn1(self.conv1(x)))
    x = self.bn2(self.conv2(x))
    if not self.skip:
      residual = self.conv_skip(residual)
    x = x + residual
    return F.relu(x)


class AudioCNN(nn.Module):
  """CNN backbone for mel-spectrogram audio input.
  
  Args:
    in_channels: Number of input channels (1 for mono, n_mels).
    embed_dim: Embedding dimension.
  """

  def __init__(self, in_channels: int = 64, embed_dim: int = 128) -> None:
    super().__init__()
    self.conv1 = nn.Conv2d(1, in_channels // 2, kernel_size=(5, 5), padding=2)
    self.pool1 = nn.MaxPool2d((2, 2))
    self.res1 = ResidualBlock(in_channels // 2, in_channels)
    self.pool2 = nn.MaxPool2d((2, 2))
    self.res2 = ResidualBlock(in_channels, in_channels * 2)
    self.pool3 = nn.MaxPool2d((2, 2))
    self.res3 = ResidualBlock(in_channels * 2, in_channels * 2)
    self.glob_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(in_channels * 2, embed_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass. Input: [B, 1, n_mels, time_frames]."""
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(self.res1(x))
    x = self.pool3(self.res2(x))
    x = self.res3(x)
    x = self.glob_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x


class MultiTaskAudioModule(nn.Module):
  """Multi-task audio classification with shared CNN backbone.
  
  Args:
    num_classes: Dict mapping task name to number of classes.
    embed_dim: Embedding dimension of CNN.
  """

  def __init__(self, num_classes: Dict[str, int], embed_dim: int = 128) -> None:
    super().__init__()
    self.num_classes = num_classes
    self.embed_dim = embed_dim

    # Note: Mel-spectrogram extraction is already done in SoundwelDataset
    # We just use the CNN backbone to process the mel-spectrograms
    self.backbone = AudioCNN(in_channels=64, embed_dim=embed_dim)

    self.heads = nn.ModuleDict()
    for task, n_cls in num_classes.items():
      self.heads[task] = nn.Linear(embed_dim, n_cls)

  def forward(self, mel_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Forward pass.
    
    Args:
      mel_spec: Mel-spectrogram [B, n_mels, time_frames] (already processed).
      
    Returns:
      Dict mapping task name to logits [B, num_classes].
    """
    # Add channel dimension if not present
    if mel_spec.dim() == 3:
      mel_spec = mel_spec.unsqueeze(1)  # [B, 1, n_mels, time]
    
    embedding = self.backbone(mel_spec)

    outputs = {}
    for task, head in self.heads.items():
      outputs[task] = head(embedding)

    return outputs
