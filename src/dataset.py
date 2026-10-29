"""
PyTorch Dataset and DataLoader with dynamic on-the-fly noise mixing and data augmentation.
"""

import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import config
from src.audio_processing import load_audio, mix_audio, compute_stft


class SpeechDenoisingDataset(Dataset):
    """
    Dynamic Speech Denoising Dataset.
    Pairs clean speech with real background noise (traffic, fan, babble, cafe)
    at dynamically sampled Signal-to-Noise Ratios (SNR).
    """
    def __init__(
        self,
        clean_files: List[Path],
        noise_files: List[Path],
        segment_samples: int = config.SEGMENT_SAMPLES,
        snr_range: Tuple[float, float] = config.TRAIN_SNR_RANGE,
        is_train: bool = True,
    ):
        self.clean_files = [Path(p) for p in clean_files]
        self.noise_files = [Path(p) for p in noise_files]
        self.segment_samples = segment_samples
        self.snr_range = snr_range
        self.is_train = is_train

        if len(self.clean_files) == 0:
            raise ValueError("No clean speech audio files provided.")
        if len(self.noise_files) == 0:
            raise ValueError("No noise audio files provided.")

    def __len__(self) -> int:
        return len(self.clean_files)

    def _extract_segment(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract or pad to fixed sample length."""
        num_samples = wav.shape[-1]
        if num_samples < self.segment_samples:
            # Repeat and pad to reach required duration
            repeats = (self.segment_samples // num_samples) + 1
            wav = wav.repeat(1, repeats)[:, :self.segment_samples]
        elif num_samples > self.segment_samples:
            if self.is_train:
                start = random.randint(0, num_samples - self.segment_samples)
            else:
                start = (num_samples - self.segment_samples) // 2
            wav = wav[:, start:start + self.segment_samples]
        return wav

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1. Load clean speech
        clean_path = self.clean_files[idx]
        clean_wav = load_audio(str(clean_path))
        clean_wav = self._extract_segment(clean_wav)

        # 2. Select noise sample
        noise_path = random.choice(self.noise_files)
        noise_wav = load_audio(str(noise_path))
        noise_wav = self._extract_segment(noise_wav)

        # 3. Dynamic SNR selection
        if self.is_train:
            snr = random.uniform(self.snr_range[0], self.snr_range[1])
        else:
            snr = 5.0  # Standard validation benchmark SNR

        # 4. Synthesize mixture
        noisy_wav, clean_wav, _ = mix_audio(clean_wav, noise_wav, snr_db=snr)

        # 5. Compute STFT Spectrograms
        noisy_mag, noisy_phase, _ = compute_stft(noisy_wav)
        clean_mag, _, _ = compute_stft(clean_wav)

        # Expand channel dimension: (1, F, T)
        noisy_mag = noisy_mag.unsqueeze(1) if noisy_mag.ndim == 2 else noisy_mag
        clean_mag = clean_mag.unsqueeze(1) if clean_mag.ndim == 2 else clean_mag

        return {
            "noisy_mag": noisy_mag.squeeze(0) if noisy_mag.shape[0] == 1 and noisy_mag.ndim == 4 else noisy_mag,
            "clean_mag": clean_mag.squeeze(0) if clean_mag.shape[0] == 1 and clean_mag.ndim == 4 else clean_mag,
            "noisy_phase": noisy_phase.squeeze(0),
            "clean_wav": clean_wav.squeeze(0),
            "noisy_wav": noisy_wav.squeeze(0),
            "snr": torch.tensor(snr, dtype=torch.float32),
        }


def create_dataloaders(
    clean_dir: Path = config.CLEAN_DIR,
    noise_dir: Path = config.NOISE_DIR,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = 0.2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Scans directories and creates train and validation DataLoaders."""
    clean_files = sorted(list(clean_dir.glob("*.wav")) + list(clean_dir.glob("*.mp3")) + list(clean_dir.glob("*.flac")))
    noise_files = sorted(list(noise_dir.glob("*.wav")) + list(noise_dir.glob("*.mp3")) + list(noise_dir.glob("*.flac")))

    if not clean_files:
        raise FileNotFoundError(f"No audio files found in clean speech directory: {clean_dir}")
    if not noise_files:
        raise FileNotFoundError(f"No audio files found in noise directory: {noise_dir}")

    # Split clean files into train and validation
    random.seed(42)
    random.shuffle(clean_files)

    split_idx = max(1, int(len(clean_files) * (1 - val_split)))
    train_clean = clean_files[:split_idx]
    val_clean = clean_files[split_idx:] if split_idx < len(clean_files) else clean_files[:1]

    train_ds = SpeechDenoisingDataset(train_clean, noise_files, is_train=True)
    val_ds = SpeechDenoisingDataset(val_clean, noise_files, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
