"""
Evaluation script to benchmark denoising performance on test audio files.
Computes SNR improvement, SI-SDR, LSD, and exports comparison audio samples.
"""

import os
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import config
from src.model import SpeechUNet
from src.audio_processing import compute_stft, compute_istft, save_audio
from src.metrics import calculate_snr, calculate_si_sdr, calculate_lsd


def evaluate_model(
    model_path: Path,
    val_loader,
    device: str = config.DEVICE,
    output_dir: Path = config.OUTPUT_DIR / "eval_results",
    save_samples_count: int = 5,
):
    """
    Evaluates the model on test/val set and computes metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    # Load model
    model = SpeechUNet().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from: {model_path} (Trained for {checkpoint.get('epoch', 'N/A')} epochs)")

    noisy_snrs = []
    denoised_snrs = []
    delta_snrs = []

    noisy_si_sdrs = []
    denoised_si_sdrs = []
    delta_si_sdrs = []

    lsds = []
    sample_counter = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            noisy_mag = batch["noisy_mag"].to(device)
            clean_mag = batch["clean_mag"].to(device)
            noisy_phase = batch["noisy_phase"].to(device)
            clean_wav = batch["clean_wav"]
            noisy_wav = batch["noisy_wav"]

            # Model inference
            denoised_mag, mask = model(noisy_mag)

            # Reconstruct time-domain audio
            pred_wav = compute_istft(
                denoised_mag.squeeze(1),
                noisy_phase,
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                win_length=config.WIN_LENGTH,
                length=config.SEGMENT_SAMPLES,
            ).cpu()

            # Compute metrics for each item in batch
            batch_size = noisy_mag.shape[0]
            for i in range(batch_size):
                c_w = clean_wav[i]
                n_w = noisy_wav[i]
                p_w = pred_wav[i]

                # SNR
                s_noisy = calculate_snr(c_w, n_w)
                s_denoised = calculate_snr(c_w, p_w)
                noisy_snrs.append(s_noisy)
                denoised_snrs.append(s_denoised)
                delta_snrs.append(s_denoised - s_noisy)

                # SI-SDR
                sdr_noisy = calculate_si_sdr(c_w, n_w)
                sdr_denoised = calculate_si_sdr(c_w, p_w)
                noisy_si_sdrs.append(sdr_noisy)
                denoised_si_sdrs.append(sdr_denoised)
                delta_si_sdrs.append(sdr_denoised - sdr_noisy)

                # LSD
                lsd_val = calculate_lsd(clean_mag[i].squeeze(), denoised_mag[i].squeeze())
                lsds.append(lsd_val)

                # Save comparison samples
                if sample_counter < save_samples_count:
                    sample_dir = output_dir / f"sample_{sample_counter + 1}"
                    sample_dir.mkdir(exist_ok=True)
                    save_audio(str(sample_dir / "1_clean_reference.wav"), c_w)
                    save_audio(str(sample_dir / "2_noisy_input.wav"), n_w)
                    save_audio(str(sample_dir / "3_denoised_output.wav"), p_w)
                    sample_counter += 1

    # Aggregate results
    results = {
        "Mean Noisy SNR (dB)": float(np.mean(noisy_snrs)),
        "Mean Denoised SNR (dB)": float(np.mean(denoised_snrs)),
        "Mean Delta-SNR Improvement (dB)": float(np.mean(delta_snrs)),
        "Mean Noisy SI-SDR (dB)": float(np.mean(noisy_si_sdrs)),
        "Mean Denoised SI-SDR (dB)": float(np.mean(denoised_si_sdrs)),
        "Mean Delta-SI-SDR Improvement (dB)": float(np.mean(delta_si_sdrs)),
        "Mean Log-Spectral Distance (dB)": float(np.mean(lsds)),
    }

    print("\n" + "="*50)
    print("           BENCHMARK EVALUATION RESULTS           ")
    print("="*50)
    for k, v in results.items():
        print(f"  {k:<35}: {v:+.2f} dB" if "dB" in k else f"  {k:<35}: {v:.3f}")
    print("="*50)
    print(f"Comparison audio saved to: {output_dir}")

    return results
