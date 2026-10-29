"""
Inference script to denoise arbitrary audio files using the trained SpeechUNet.
Generates denoised audio WAV file and side-by-side visual spectrograms.
"""

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

import config
from src.model import SpeechUNet
from src.audio_processing import load_audio, save_audio, compute_stft, compute_istft


def denoise_audio_file(
    input_audio_path: str,
    output_audio_path: str,
    checkpoint_path: str = str(config.CHECKPOINT_DIR / "best_model.pt"),
    plot_spectrograms: bool = True,
    device: str = config.DEVICE,
):
    """
    Denoises an audio file and optionally generates comparison visualization plots.
    """
    device = torch.device(device)
    input_path = Path(input_audio_path)
    output_path = Path(output_audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    # 1. Load Audio
    print(f"--> Loading noisy audio: {input_path}")
    waveform = load_audio(str(input_path), target_sr=config.SAMPLE_RATE).to(device)
    num_samples = waveform.shape[-1]
    duration_sec = num_samples / config.SAMPLE_RATE
    print(f"--> Audio length: {duration_sec:.2f}s ({num_samples} samples @ {config.SAMPLE_RATE} Hz)")

    # 2. STFT Analysis
    mag, phase, full_mag = compute_stft(waveform)
    mag_input = mag.unsqueeze(1)  # (1, 1, 256, T)

    # 3. Load Model
    print(f"--> Loading model from checkpoint: {checkpoint_path}")
    model = SpeechUNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 4. Neural Network Inference
    with torch.no_grad():
        denoised_mag, mask = model(mag_input)

    # 5. iSTFT Synthesis
    denoised_mag = denoised_mag.squeeze(1)  # (1, 256, T)
    denoised_wav = compute_istft(
        denoised_mag,
        phase,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        length=num_samples,
    ).squeeze(0)

    # 6. Save Clean Audio
    save_audio(str(output_path), denoised_wav, sr=config.SAMPLE_RATE)
    print(f"--> Successfully saved denoised audio to: {output_path}")

    # 7. Generate Comparison Plot
    if plot_spectrograms:
        plot_path = output_path.with_suffix(".png")
        print(f"--> Generating visual spectrogram comparison at: {plot_path}")

        noisy_spec_np = 20 * np.log10(np.maximum(mag[0].cpu().numpy(), 1e-5))
        mask_np = mask[0, 0].cpu().numpy()
        denoised_spec_np = 20 * np.log10(np.maximum(denoised_mag[0].cpu().numpy(), 1e-5))

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

        im0 = axes[0].imshow(noisy_spec_np, aspect="auto", origin="lower", cmap="magma", vmin=-60, vmax=20)
        axes[0].set_title("1. Noisy Input Spectrogram (Voice + Noise)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Freq Bins (0 - 8 kHz)")
        fig.colorbar(im0, ax=axes[0], label="Magnitude (dB)")

        im1 = axes[1].imshow(mask_np, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1].set_title("2. Neural Network Predicted Filter Mask [0.0 = Silence, 1.0 = Speech]", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Freq Bins (0 - 8 kHz)")
        fig.colorbar(im1, ax=axes[1], label="Mask Value")

        im2 = axes[2].imshow(denoised_spec_np, aspect="auto", origin="lower", cmap="magma", vmin=-60, vmax=20)
        axes[2].set_title("3. Denoised Clean Output Spectrogram", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Time Frames")
        axes[2].set_ylabel("Freq Bins (0 - 8 kHz)")
        fig.colorbar(im2, ax=axes[2], label="Magnitude (dB)")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"--> Saved spectrogram comparison to: {plot_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise an audio file using trained SpeechUNet")
    parser.add_argument("--input", type=str, required=True, help="Path to input noisy audio file")
    parser.add_argument("--output", type=str, default="outputs/denoised_output.wav", help="Path to save cleaned audio")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pt"), help="Checkpoint file path")
    parser.add_argument("--no_plot", action="store_true", help="Disable spectrogram plot generation")
    args = parser.parse_args()

    denoise_audio_file(
        input_audio_path=args.input,
        output_audio_path=args.output,
        checkpoint_path=args.checkpoint,
        plot_spectrograms=not args.no_plot,
    )
