# NeuralClear: Neural Network Based Speech Noise Removal

A Deep Learning Speech Enhancement system utilizing a **2D U-Net with Skip Connections and a Recurrent Bottleneck** operating in the Time-Frequency (STFT Spectrogram) domain to remove environmental noises (traffic, fan, babble, and cafeteria noise) from corrupted speech.

---

## 📌 Architecture Overview

```
              ┌─────────────────────────────┐
              │  Corrupted Audio x(t)       │
              │  (Speech + Traffic/Fan/etc) │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Short-Time Fourier Transform│
              │         (STFT)              │
              └──────────────┬──────────────┘
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
        ┌─────────────┐             ┌─────────────┐
        │  Magnitude  │             │    Phase    │
        │  |X(f, t)|  │             │  ∠X(f, t)   │
        └──────┬──────┘             └──────┬──────┘
               │                           │
               ▼                           │
     ┌───────────────────┐                 │
     │ 2D U-Net with     │                 │
     │ Skip Connections  │                 │
     │ & BiGRU Bottleneck│                 │
     └─────────┬─────────┘                 │
               │                           │
               ▼                           │
     ┌───────────────────┐                 │
     │ Ideal Ratio Mask  │                 │
     │     M(f, t)       │                 │
     └─────────┬─────────┘                 │
               │                           │
               ▼                           │
     ┌───────────────────┐                 │
     │ Denoised Mag      │                 │
     │ |Ŝ| = M ⊙ |X|     │                 │
     └─────────┬─────────┘                 │
               │                           │
               └─────────────┬─────────────┘
                             ▼
              ┌─────────────────────────────┐
              │    Inverse STFT (iSTFT)     │
              │    Waveform Synthesis       │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │    Clean Speech ŝ(t)        │
              └─────────────────────────────┘
```

---

## 🔬 Mathematical Formulation

### 1. Signal Model
A recorded audio signal in the time domain is modeled as:
$$x(t) = s(t) + n(t)$$
where $s(t)$ is clean human speech and $n(t)$ is additive environmental noise (traffic rumble, fan hum, or background voices).

### 2. Time-Frequency Representation (STFT)
Using a Hann window $w(n)$ of size $N = 512$ (32 ms @ 16 kHz) and hop length $R = 128$ (8 ms, 75% overlap):
$$X(f, t) = \sum_{n=0}^{N-1} x(t \cdot R + n) w(n) e^{-j \frac{2\pi f n}{N}} = |X(f, t)| e^{j \angle X(f, t)}$$
The magnitude spectrogram $|X(f, t)|$ represents the energy distribution across 256 frequency bins and $T$ time frames.

### 3. Spectral Masking vs. Direct Mapping
Instead of predicting unconstrained spectral values, the network predicts an **Ideal Ratio Mask (IRM)** $M(f, t) \in [0.0, 1.0]$ via a bounded Sigmoid output:
$$M(f, t) = \sigma(f_{\theta}(|X(f, t)|))$$
$$|\hat{S}(f, t)| = M(f, t) \odot |X(f, t)|$$
This guarantees numerical stability, preserves the dynamic range of speech formants, and eliminates synthetic amplification artifacts.

### 4. Perceptually-Weighted Multi-Objective Loss
$$L_{total} = \alpha L_{compressed} + \beta L_{mag} + \gamma L_{mask}$$
- **Power-Law Compressed Loss**: Aligns with human auditory loudness perception (Stevens' Power Law):
  $$L_{compressed} = \frac{1}{F \cdot T} \||S|^{0.3} - |\hat{S}|^{0.3}\|_1$$
- **Linear Magnitude Loss**: $\| |S| - |\hat{S}| \|_1$
- **Target IRM Guidance Loss**: $\| M_{pred} - \min(1.0, |S| / (|X| + \epsilon)) \|_1$

---

## 📂 Repository Structure

```
.
├── config.py                 # Central audio, STFT, model, and training hyperparameters
├── train.py                  # PyTorch training pipeline with validation & checkpointing
├── inference.py              # CLI tool to denoise any audio file & generate spectrograms
├── generate_sample_data.py   # Acoustic synthesizer for clean speech & noise categories
├── app.py                    # FastAPI web backend with REST APIs
├── web/
│   ├── index.html            # Studio frontend with telemetry & A/B audio switcher
│   ├── style.css             # Vanilla CSS design system with dark mode & glow effects
│   └── app.js                # Frontend playback logic and spectrogram rendering
├── src/
│   ├── model.py              # 2D U-Net with skip connections and BiGRU bottleneck
│   ├── audio_processing.py   # STFT, iSTFT, dynamic SNR mixing, audio I/O
│   ├── dataset.py            # Dynamic PyTorch Dataset with real-time noise augmentation
│   ├── losses.py             # Multi-resolution perceptual spectral loss
│   ├── metrics.py            # SNR improvement (ΔSNR), SI-SDR, and LSD metrics
│   └── evaluate.py           # Evaluation script for quantitative benchmarking
├── checkpoints/              # Saved model checkpoints (best_model.pt)
├── outputs/                  # Exported plots (loss_curve.png, spectrograms) and audio
└── data/
    ├── clean_speech/         # Clean speech WAV files
    └── noise/                # Noise categories (traffic, fan, babble)
```

---

## 🚀 Quickstart Guide

### 1. Requirements
Ensure Python 3.10+ is installed. Required libraries:
```bash
pip install torch torchaudio soundfile librosa matplotlib fastapi uvicorn tqdm
```

### 2. Generate Audio Datasets
Generate clean speech formants and environmental noise files:
```bash
python generate_sample_data.py
```

### 3. Train the Model
Run training with dynamic noise augmentation:
```bash
python train.py --epochs 15 --batch_size 8 --lr 5e-4
```
Checkpoints will be saved automatically to `checkpoints/best_model.pt`.

### 4. Denoise Any Audio File
Denoise an external audio file and plot side-by-side spectrograms:
```bash
python inference.py --input path/to/noisy.wav --output outputs/clean_result.wav
```

### 5. Launch the Interactive Web Studio
Start the local web application:
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to:
- Test Traffic, Fan, and Babble noise scenarios
- Adjust the input SNR slider in real time (-5 dB to +15 dB)
- Use the **A/B Quick Switch** to compare noisy and clean speech on the fly
- Inspect high-resolution acoustic spectrogram heatmaps and the predicted filter mask
