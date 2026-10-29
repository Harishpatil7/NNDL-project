"""
Audio signal processing utilities: STFT, iSTFT, dynamic SNR mixing, and audio I/O.
"""

import math
import numpy as np
import torch
import soundfile as sf
from typing import Tuple, Optional
import config


def load_audio(filepath: str, target_sr: int = config.SAMPLE_RATE) -> torch.Tensor:
    """
    Load an audio file, convert to mono, and ensure it matches the target sample rate.
    Returns:
        torch.Tensor of shape (1, num_samples), float32 normalized in [-1.0, 1.0].
    """
    data, sr = sf.read(filepath, dtype="float32")
    if data.ndim > 1:
        # Convert multi-channel to mono
        data = np.mean(data, axis=1)

    tensor = torch.from_numpy(data).unsqueeze(0)  # (1, T)

    # Resample if sample rate does not match target
    if sr != target_sr:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        tensor = resampler(tensor)

    # Normalize amplitude
    max_val = torch.max(torch.abs(tensor))
    if max_val > 1e-6:
        tensor = tensor / max_val * 0.95

    return tensor


def save_audio(filepath: str, waveform: torch.Tensor, sr: int = config.SAMPLE_RATE):
    """
    Save audio waveform tensor to disk as WAV format.
    waveform: torch.Tensor of shape (num_samples,) or (1, num_samples).
    """
    if isinstance(waveform, torch.Tensor):
        audio_np = waveform.detach().cpu().numpy()
    else:
        audio_np = np.asarray(waveform)

    if audio_np.ndim == 2:
        audio_np = audio_np.squeeze(0)

    # Clip to avoid distortion
    audio_np = np.clip(audio_np, -1.0, 1.0)
    sf.write(filepath, audio_np, sr)


def compute_stft(
    waveform: torch.Tensor,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    win_length: int = config.WIN_LENGTH,
    crop_bins: int = config.N_FREQ_BINS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Short-Time Fourier Transform.
    Args:
        waveform: (batch, time) or (batch, 1, time)
    Returns:
        mag_cropped: (batch, crop_bins, frames) -> cropped magnitude spectrogram (256 bins)
        phase: (batch, 257, frames) -> full angle phase
        full_mag: (batch, 257, frames) -> full magnitude spectrogram
    """
    if waveform.ndim == 3:
        waveform = waveform.squeeze(1)

    device = waveform.device
    window = torch.hann_window(win_length, device=device)

    # Complex STFT: shape (batch, freq_bins=257, time_frames)
    stft_res = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    full_mag = torch.abs(stft_res) + 1e-8
    phase = torch.angle(stft_res)

    # Crop to 256 bins for clean U-Net downsampling
    mag_cropped = full_mag[:, :crop_bins, :]

    return mag_cropped, phase, full_mag


def compute_istft(
    mag: torch.Tensor,
    phase: torch.Tensor,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    win_length: int = config.WIN_LENGTH,
    length: Optional[int] = None,
) -> torch.Tensor:
    """
    Inverse STFT to reconstruct time-domain waveform from magnitude and phase.
    Args:
        mag: (batch, 256 or 257, frames)
        phase: (batch, 257, frames)
        length: original waveform sample length for exact truncation
    Returns:
        waveform: (batch, samples)
    """
    device = mag.device
    window = torch.hann_window(win_length, device=device)

    # If mag has 256 bins, pad the highest Nyquist bin back to 257
    if mag.shape[1] == config.N_FREQ_BINS:
        # Pad with 0 or the last bin
        last_bin = mag[:, -1:, :]
        mag_full = torch.cat([mag, last_bin], dim=1)
    else:
        mag_full = mag

    # Complex spectrogram reconstruction
    complex_spec = torch.polar(mag_full, phase)

    # Reconstruct waveform
    waveform = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        length=length,
    )

    return waveform


def mix_audio(
    speech: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, float]:
    """
    Mix clean speech and noise at a specified Signal-to-Noise Ratio (SNR) in dB.
    Args:
        speech: (1, T) clean speech waveform
        noise: (1, T) noise waveform of identical length
        snr_db: target SNR in decibels
    Returns:
        noisy_audio: (1, T) mixed waveform
        scale: noise scaling factor applied
    """
    # Compute active signal power
    p_speech = torch.mean(speech ** 2) + eps
    p_noise = torch.mean(noise ** 2) + eps

    # Target noise power for the desired SNR:
    # SNR = 10 * log10(P_speech / P_noise_target)
    # P_noise_target = P_speech / (10 ** (SNR / 10))
    p_noise_target = p_speech / (10.0 ** (snr_db / 10.0))

    scale = torch.sqrt(p_noise_target / p_noise)
    scaled_noise = noise * scale

    noisy = speech + scaled_noise

    # Prevent clipping distortion while maintaining relative SNR
    max_amp = torch.max(torch.abs(noisy))
    if max_amp > 0.99:
        norm_factor = 0.95 / max_amp
        noisy = noisy * norm_factor
        speech = speech * norm_factor

    return noisy, speech, float(scale)


def calculate_snr(clean: torch.Tensor, noisy_or_denoised: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Calculate conventional Signal-to-Noise Ratio (SNR) in decibels.
    SNR = 10 * log10( sum(clean^2) / sum((clean - estimated)^2) )
    """
    noise = clean - noisy_or_denoised
    signal_power = torch.sum(clean ** 2) + eps
    noise_power = torch.sum(noise ** 2) + eps
    snr = 10.0 * torch.log10(signal_power / noise_power)
    return float(snr.item())
