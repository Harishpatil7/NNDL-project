"""
FastAPI Backend Server for Interactive Speech Denoising Web App.
Provides REST APIs for denoising audio, presets, and spectrogram visualizations.
"""

import io
import os
import base64
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import config
from src.model import SpeechUNet
from src.audio_processing import load_audio, save_audio, compute_stft, compute_istft, mix_audio
from src.metrics import calculate_snr, calculate_si_sdr

app = FastAPI(title="Neural Network Speech Denoising Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
STATIC_DIR = config.BASE_DIR / "web"
MEDIA_DIR = config.OUTPUT_DIR / "web_media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# Global model state
_model = None
_device = torch.device(config.DEVICE)


def get_model():
    global _model
    if _model is None:
        model_path = config.CHECKPOINT_DIR / "best_model.pt"
        if not model_path.exists():
            model_path = config.CHECKPOINT_DIR / "latest_model.pt"
        
        _model = SpeechUNet().to(_device)
        if model_path.exists():
            ckpt = torch.load(model_path, map_location=_device, weights_only=False)
            _model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Backend] Loaded trained weights from {model_path}")
        else:
            print("[Backend] Initialized model with fresh weights (training pending)")
        _model.eval()
    return _model


def render_spectrogram_base64(mag_tensor: torch.Tensor, title: str, cmap: str = "magma", is_mask: bool = False) -> str:
    """Renders spectrogram or mask as a base64 encoded PNG."""
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    data = mag_tensor.detach().cpu().numpy()

    if not is_mask:
        # Convert to dB scale
        db_data = 20 * np.log10(np.maximum(data, 1e-5))
        im = ax.imshow(db_data, aspect="auto", origin="lower", cmap=cmap, vmin=-50, vmax=20)
    else:
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_title(title, fontsize=10, fontweight="bold", color="#f8fafc", pad=8)
    ax.set_xlabel("Time Frames", fontsize=8, color="#94a3b8")
    ax.set_ylabel("Freq (kHz)", fontsize=8, color="#94a3b8")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#020617")

    # Clean borders
    for spine in ax.spines.values():
        spine.set_color("#334155")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=7)
    cbar.outline.set_edgecolor("#334155")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.get("/api/presets")
def list_presets():
    """Returns available audio presets."""
    return [
        {
            "id": "traffic",
            "name": "Speech + Heavy Traffic & Highway",
            "description": "Low-frequency engine rumble, tire noise, and Doppler car pass-bys.",
            "noise_file": "traffic_noise_1.wav",
            "clean_file": "clean_speech_001.wav",
            "default_snr": 0.0,
        },
        {
            "id": "fan",
            "name": "Speech + Electric Desk Fan",
            "description": "Stationary 60/120 Hz motor hum and turbulent high-frequency blade flutter.",
            "noise_file": "fan_noise_1.wav",
            "clean_file": "clean_speech_002.wav",
            "default_snr": 2.0,
        },
        {
            "id": "babble",
            "name": "Speech + Cafeteria / People Talking",
            "description": "Cocktail party background chatter and overlapping human voices.",
            "noise_file": "babble_noise_1.wav",
            "clean_file": "clean_speech_003.wav",
            "default_snr": 3.0,
        },
    ]


@app.get("/api/model_info")
def model_info():
    """Returns architecture specs and hyperparameters."""
    model = get_model()
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "architecture": "2D U-Net with Skip Connections + Recurrent Bottleneck",
        "parameters": f"{total_params:,}",
        "sampling_rate": f"{config.SAMPLE_RATE} Hz",
        "stft_window": f"{config.N_FFT} samples ({config.N_FFT / config.SAMPLE_RATE * 1000:.1f} ms)",
        "hop_length": f"{config.HOP_LENGTH} samples ({config.HOP_LENGTH / config.SAMPLE_RATE * 1000:.1f} ms)",
        "frequency_bins": config.N_FREQ_BINS,
        "mask_type": "Ideal Ratio Mask (Bounded Sigmoid)",
        "device": str(_device),
    }


@app.post("/api/denoise_preset")
def denoise_preset(preset_id: str = Form(...), snr_db: float = Form(3.0)):
    """Runs denoising on a selected preset audio pair with custom SNR."""
    presets = {p["id"]: p for p in list_presets()}
    if preset_id not in presets:
        raise HTTPException(status_code=400, detail="Invalid preset ID")

    preset = presets[preset_id]
    clean_path = config.CLEAN_DIR / preset["clean_file"]
    noise_path = config.NOISE_DIR / preset["noise_file"]

    if not clean_path.exists() or not noise_path.exists():
        raise HTTPException(status_code=404, detail="Preset audio files not found. Generate data first.")

    clean_wav = load_audio(str(clean_path)).to(_device)
    noise_wav = load_audio(str(noise_path)).to(_device)

    # Match length
    min_len = min(clean_wav.shape[-1], noise_wav.shape[-1], 3 * config.SAMPLE_RATE)
    clean_wav = clean_wav[:, :min_len]
    noise_wav = noise_wav[:, :min_len]

    # Synthesize noisy mix
    noisy_wav, clean_wav, _ = mix_audio(clean_wav, noise_wav, snr_db=snr_db)

    return process_and_respond(clean_wav, noisy_wav, label=f"preset_{preset_id}")


@app.post("/api/denoise_upload")
async def denoise_upload(audio_file: UploadFile = File(...)):
    """Uploads arbitrary audio file, applies denoising, and returns results."""
    contents = await audio_file.read()
    temp_input = MEDIA_DIR / f"upload_{audio_file.filename}"
    with open(temp_input, "wb") as f:
        f.write(contents)

    noisy_wav = load_audio(str(temp_input)).to(_device)
    return process_and_respond(None, noisy_wav, label="upload")


def process_and_respond(clean_wav: torch.Tensor, noisy_wav: torch.Tensor, label: str):
    """Core inference, metric calculation, spectrogram rendering, and audio export."""
    model = get_model()
    num_samples = noisy_wav.shape[-1]

    # Compute STFT
    noisy_mag, noisy_phase, _ = compute_stft(noisy_wav)
    noisy_mag_in = noisy_mag.unsqueeze(1)  # (1, 1, 256, T)

    # Model inference
    with torch.no_grad():
        denoised_mag, mask = model(noisy_mag_in)

    denoised_mag = denoised_mag.squeeze(1)  # (1, 256, T)
    denoised_wav = compute_istft(
        denoised_mag,
        noisy_phase,
        length=num_samples,
    )

    # Save audio files for client playback
    out_noisy_file = MEDIA_DIR / f"{label}_noisy.wav"
    out_denoised_file = MEDIA_DIR / f"{label}_denoised.wav"
    save_audio(str(out_noisy_file), noisy_wav)
    save_audio(str(out_denoised_file), denoised_wav)

    has_clean = clean_wav is not None
    if has_clean:
        out_clean_file = MEDIA_DIR / f"{label}_clean.wav"
        save_audio(str(out_clean_file), clean_wav)
        snr_in = calculate_snr(clean_wav, noisy_wav)
        snr_out = calculate_snr(clean_wav, denoised_wav)
        delta_snr = snr_out - snr_in
        si_sdr_in = calculate_si_sdr(clean_wav, noisy_wav)
        si_sdr_out = calculate_si_sdr(clean_wav, denoised_wav)
    else:
        out_clean_file = None
        snr_in, snr_out, delta_snr = 0.0, 0.0, 0.0
        si_sdr_in, si_sdr_out = 0.0, 0.0

    # Render base64 spectrograms
    img_noisy = render_spectrogram_base64(noisy_mag[0], "1. Noisy Input Spectrogram", cmap="magma")
    img_mask = render_spectrogram_base64(mask[0, 0], "2. Neural Filter Mask [0.0 = Noise Cut, 1.0 = Voice Pass]", cmap="viridis", is_mask=True)
    img_denoised = render_spectrogram_base64(denoised_mag[0], "3. Denoised Speech Spectrogram", cmap="magma")

    return {
        "status": "success",
        "has_clean_reference": has_clean,
        "audio_urls": {
            "clean": f"/media/{out_clean_file.name}" if has_clean else None,
            "noisy": f"/media/{out_noisy_file.name}",
            "denoised": f"/media/{out_denoised_file.name}",
        },
        "spectrograms": {
            "noisy": img_noisy,
            "mask": img_mask,
            "denoised": img_denoised,
        },
        "metrics": {
            "noisy_snr": round(snr_in, 2) if has_clean else "N/A",
            "denoised_snr": round(snr_out, 2) if has_clean else "N/A",
            "snr_gain": round(delta_snr, 2) if has_clean else "N/A",
            "noisy_si_sdr": round(si_sdr_in, 2) if has_clean else "N/A",
            "denoised_si_sdr": round(si_sdr_out, 2) if has_clean else "N/A",
        },
    }


# Mount static audio files and web frontend
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
