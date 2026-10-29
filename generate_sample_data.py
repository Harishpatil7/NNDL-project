"""
Sample audio generator: Creates synthetic speech formants and environmental noise
(traffic, fan, babble, cafe) for immediate training, testing, and validation.
"""

import math
from pathlib import Path
import numpy as np
import soundfile as sf
import config


def generate_synthetic_speech_utterance(duration_sec: float = 3.0, sr: int = config.SAMPLE_RATE, seed: int = 42) -> np.ndarray:
    """
    Synthesizes speech-like audio using source-filter acoustic model:
    - Glottal pulse train with fundamental frequency (F0) pitch contour
    - Formant vocal tract resonance filters (F1, F2, F3) for vowels
    - Syllabic amplitude envelope modulation
    """
    np.random.seed(seed)
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # 1. Fundamental frequency (F0) pitch contour (100 - 240 Hz typical human speech)
    base_f0 = np.random.uniform(120, 180)
    pitch_mod = 15.0 * np.sin(2 * np.pi * 1.5 * t) + 10.0 * np.sin(2 * np.pi * 0.7 * t)
    f0 = base_f0 + pitch_mod

    # Cumulative phase for variable frequency excitation
    phase = 2 * np.pi * np.cumsum(f0) / sr

    # Harmonic glottal source
    source = np.zeros(num_samples)
    for h in range(1, 15):
        amp = 1.0 / (h ** 0.8)
        source += amp * np.sin(h * phase)

    # 2. Vocal tract formants (F1, F2, F3 resonances)
    formant_freqs = [
        np.random.uniform(400, 800),    # F1 (vowel height)
        np.random.uniform(1200, 2200),  # F2 (vowel front/back)
        np.random.uniform(2500, 3200),  # F3 (speaker timbre)
    ]
    formant_bw = [80, 110, 140]

    from scipy.signal import lfilter
    speech = np.zeros(num_samples)
    for f_c, bw in zip(formant_freqs, formant_bw):
        decay = np.exp(-np.pi * bw * (1.0 / sr))
        w = 2 * np.pi * f_c / sr
        a1 = 2 * decay * np.cos(w)
        a2 = -decay * decay
        b0 = (1 - decay) * np.sin(w)
        y = lfilter([b0], [1.0, -a1, -a2], source)
        speech += y

    # 3. Syllable amplitude rhythm (syllabic envelope ~3-5 Hz speech rate)
    syllable_rate = np.random.uniform(3.0, 4.5)
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t)) ** 2
    # Add brief pauses / phrase cadence
    pause_mask = (np.sin(2 * np.pi * 0.4 * t) > -0.6).astype(np.float32)
    envelope = envelope * pause_mask

    speech = speech * envelope
    # Normalize peak
    max_amp = np.max(np.abs(speech))
    if max_amp > 1e-4:
        speech = speech / max_amp * 0.9

    return speech.astype(np.float32)


def generate_fan_noise(duration_sec: float = 4.0, sr: int = config.SAMPLE_RATE, seed: int = 101) -> np.ndarray:
    """Stationary fan noise: Pink noise + 60 Hz / 120 Hz electric motor hum + air turbulence."""
    np.random.seed(seed)
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Motor hum
    hum = 0.3 * np.sin(2 * np.pi * 60 * t) + 0.2 * np.sin(2 * np.pi * 120 * t) + 0.1 * np.sin(2 * np.pi * 180 * t)

    # Pink noise (1/f)
    white = np.random.randn(num_samples)
    # Simple pink filter via running cumulative filter
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    from scipy.signal import lfilter
    pink = lfilter(b, a, white)

    fan = hum + 0.7 * pink
    fan = fan / np.max(np.abs(fan)) * 0.7
    return fan.astype(np.float32)


def generate_traffic_noise(duration_sec: float = 4.0, sr: int = config.SAMPLE_RATE, seed: int = 202) -> np.ndarray:
    """Traffic noise: Low-frequency road rumble + vehicle passing sweeps."""
    np.random.seed(seed)
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Road friction rumble (low-pass filtered white noise)
    white = np.random.randn(num_samples)
    from scipy.signal import butter, lfilter
    b, a = butter(4, 400 / (sr / 2), btype="low")
    rumble = lfilter(b, a, white)

    # Vehicle pass-by envelope modulation
    car_pass = 0.5 * np.exp(-((t - duration_sec / 2) ** 2) / 0.5)
    # Doppler frequency modulation
    engine_freq = 150 - 40 * (t / duration_sec)
    engine = 0.3 * np.sin(2 * np.pi * engine_freq * t) * car_pass

    traffic = rumble + engine
    traffic = traffic / np.max(np.abs(traffic)) * 0.7
    return traffic.astype(np.float32)


def generate_babble_noise(duration_sec: float = 4.0, sr: int = config.SAMPLE_RATE, seed: int = 303) -> np.ndarray:
    """Babble / People talking background noise: Sum of multiple incoherent speech voices."""
    babble = np.zeros(int(duration_sec * sr), dtype=np.float32)
    for i in range(8):
        v = generate_synthetic_speech_utterance(duration_sec, sr, seed=seed + i * 37)
        # Shift phase/time randomly
        shift = np.random.randint(0, len(v))
        babble += np.roll(v, shift)

    babble = babble / np.max(np.abs(babble)) * 0.7
    return babble.astype(np.float32)


def generate_dataset(num_clean_samples: int = 30):
    """Generates all clean speech and noise sample files."""
    config.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    config.NOISE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"--> Generating {num_clean_samples} clean speech samples in {config.CLEAN_DIR}...")
    for i in range(1, num_clean_samples + 1):
        speech = generate_synthetic_speech_utterance(duration_sec=3.0, seed=1000 + i)
        filepath = config.CLEAN_DIR / f"clean_speech_{i:03d}.wav"
        sf.write(str(filepath), speech, config.SAMPLE_RATE)

    print(f"--> Generating noise files (fan, traffic, babble, cafe) in {config.NOISE_DIR}...")
    # Fan noise variants
    for i in range(1, 4):
        fan = generate_fan_noise(duration_sec=5.0, seed=500 + i)
        sf.write(str(config.NOISE_DIR / f"fan_noise_{i}.wav"), fan, config.SAMPLE_RATE)

    # Traffic noise variants
    for i in range(1, 4):
        traffic = generate_traffic_noise(duration_sec=5.0, seed=600 + i)
        sf.write(str(config.NOISE_DIR / f"traffic_noise_{i}.wav"), traffic, config.SAMPLE_RATE)

    # Babble / people talking noise variants
    for i in range(1, 4):
        babble = generate_babble_noise(duration_sec=5.0, seed=700 + i)
        sf.write(str(config.NOISE_DIR / f"babble_noise_{i}.wav"), babble, config.SAMPLE_RATE)

    print("--> Synthetic dataset generation complete!")


if __name__ == "__main__":
    generate_dataset()
