"""Audio models for pig vocalization classification and synthesis."""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MelSpectrogramExtractor(nn.Module):
  """Extract Mel-spectrogram features from raw audio.
  
  Args:
    sample_rate: Audio sample rate in Hz.
    n_mels: Number of mel bands.
    n_fft: FFT window size.
    hop_length: Hop length for STFT.
    f_min: Minimum frequency.
    f_max: Maximum frequency.
  """

  def __init__(
      self,
      sample_rate: int = 16000,
      n_mels: int = 64,
      n_fft: int = 400,
      hop_length: int = 160,
      f_min: int = 50,
      f_max: int = 8000,
  ) -> None:
    super().__init__()
    self.sample_rate = sample_rate
    self.n_mels = n_mels
    self.n_fft = n_fft
    self.hop_length = hop_length
    self.f_min = f_min
    self.f_max = f_max

    mel_fb = torch.nn.functional.pad(
        torch.from_numpy(self._mel_scale(sample_rate, n_fft, n_mels, f_min, f_max)).float(),
        (0, 1),
    )
    self.register_buffer("mel_fb", mel_fb)

  @staticmethod
  def _mel_scale(
      sr: int, n_fft: int, n_mels: int, f_min: float, f_max: float
  ) -> "np.ndarray":
    """Compute mel filterbank (librosa-compatible)."""
    import numpy as np

    def hz_to_mel(f: float) -> float:
      return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m: float) -> float:
      return 700 * (10 ** (m / 2595) - 1)

    mel_f = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_f = mel_to_hz(mel_f)
    bin_hz = np.linspace(0, sr / 2, n_fft // 2 + 1)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
      f_m_left = hz_f[m - 1]
      f_m = hz_f[m]
      f_m_right = hz_f[m + 1]

      for k in range(n_fft // 2 + 1):
        if bin_hz[k] < f_m_left or bin_hz[k] > f_m_right:
          continue
        if bin_hz[k] < f_m:
          fb[m - 1, k] = (bin_hz[k] - f_m_left) / (f_m - f_m_left)
        else:
          fb[m - 1, k] = (f_m_right - bin_hz[k]) / (f_m_right - f_m)

    return fb

  def forward(self, waveform: torch.Tensor) -> torch.Tensor:
    """Compute Mel-spectrogram.
    
    Args:
      waveform: Audio tensor [B, time_steps] or [time_steps].
      
    Returns:
      Mel-spectrogram [B, n_mels, time_frames].
    """
    if waveform.dim() == 1:
      waveform = waveform.unsqueeze(0)

    stft = torch.stft(
        waveform,
        n_fft=self.n_fft,
        hop_length=self.hop_length,
        return_complex=False,
    )
    spec = stft[..., 0] ** 2 + stft[..., 1] ** 2
    mel = torch.matmul(self.mel_fb, spec)
    mel = torch.log(torch.clamp(mel, min=1e-9))
    return mel


class ResidualBlock(nn.Module):
  """Residual block for audio CNN."""

  def __init__(self, in_channels: int, out_channels: int) -> None:
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.skip = in_channels == out_channels

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass."""
    res = x if self.skip else x
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    if self.skip:
      out = out + res
    return F.relu(out)


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

    self.mel_extractor = MelSpectrogramExtractor()
    self.backbone = AudioCNN(in_channels=64, embed_dim=embed_dim)

    self.heads = nn.ModuleDict()
    for task, n_cls in num_classes.items():
      self.heads[task] = nn.Linear(embed_dim, n_cls)

  def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Forward pass.
    
    Args:
      waveform: Raw audio [B, time_steps].
      
    Returns:
      Dict mapping task name to logits [B, num_classes].
    """
    mel = self.mel_extractor(waveform).unsqueeze(1)
    embedding = self.backbone(mel)

    outputs = {}
    for task, head in self.heads.items():
      outputs[task] = head(embedding)

    return outputs
