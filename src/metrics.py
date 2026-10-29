"""
Evaluation metrics for speech enhancement:
- SNR (Signal-to-Noise Ratio) & SNR Improvement (Delta-SNR)
- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
- LSD (Log-Spectral Distance)
"""

import torch
import numpy as np
from typing import Dict


def calculate_snr(clean: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> float:
    """Computes conventional SNR in dB."""
    if clean.ndim > 1:
        clean = clean.squeeze()
    if est.ndim > 1:
        est = est.squeeze()

    min_len = min(clean.shape[-1], est.shape[-1])
    clean = clean[:min_len]
    est = est[:min_len]

    noise = clean - est
    s_p = torch.sum(clean ** 2) + eps
    n_p = torch.sum(noise ** 2) + eps
    return float(10.0 * torch.log10(s_p / n_p).item())


def calculate_si_sdr(clean: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.
    Standard metric for speech enhancement and source separation.
    """
    if clean.ndim > 1:
        clean = clean.squeeze()
    if est.ndim > 1:
        est = est.squeeze()

    min_len = min(clean.shape[-1], est.shape[-1])
    clean = clean[:min_len] - torch.mean(clean[:min_len])
    est = est[:min_len] - torch.mean(est[:min_len])

    # Optimal scaling factor: alpha = <s, s_hat> / ||s||^2
    alpha = torch.dot(est, clean) / (torch.sum(clean ** 2) + eps)
    s_target = alpha * clean
    e_noise = est - s_target

    target_power = torch.sum(s_target ** 2) + eps
    noise_power = torch.sum(e_noise ** 2) + eps
    si_sdr = 10.0 * torch.log10(target_power / noise_power)
    return float(si_sdr.item())


def calculate_lsd(clean_mag: torch.Tensor, est_mag: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Log-Spectral Distance (LSD) in dB.
    Measures the root-mean-square distance between log-power spectrograms. Lower is better.
    """
    p_clean = 10.0 * torch.log10(clean_mag ** 2 + eps)
    p_est = 10.0 * torch.log10(est_mag ** 2 + eps)

    # Frame-wise Euclidean distance averaged over time
    frame_dist = torch.sqrt(torch.mean((p_clean - p_est) ** 2, dim=-2) + eps)
    lsd = torch.mean(frame_dist)
    return float(lsd.item())


def compute_metrics(clean_wav: torch.Tensor, noisy_wav: torch.Tensor, denoised_wav: torch.Tensor) -> Dict[str, float]:
    """Computes all key objective performance metrics for a sample."""
    snr_noisy = calculate_snr(clean_wav, noisy_wav)
    snr_denoised = calculate_snr(clean_wav, denoised_wav)
    delta_snr = snr_denoised - snr_noisy

    si_sdr_noisy = calculate_si_sdr(clean_wav, noisy_wav)
    si_sdr_denoised = calculate_si_sdr(clean_wav, denoised_wav)
    delta_si_sdr = si_sdr_denoised - si_sdr_noisy

    return {
        "noisy_snr_db": round(snr_noisy, 2),
        "denoised_snr_db": round(snr_denoised, 2),
        "delta_snr_db": round(delta_snr, 2),
        "noisy_si_sdr_db": round(si_sdr_noisy, 2),
        "denoised_si_sdr_db": round(si_sdr_denoised, 2),
        "delta_si_sdr_db": round(delta_si_sdr, 2),
    }
