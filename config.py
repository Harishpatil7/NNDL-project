"""
Configuration and Hyperparameters for Speech Denoising Neural Network.
"""

import os
from pathlib import Path
import torch

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean_speech"
NOISE_DIR = DATA_DIR / "noise"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

# Audio & STFT Parameters
SAMPLE_RATE = 16000           # 16 kHz sampling rate (standard for speech)
N_FFT = 512                   # Window size (32 ms @ 16 kHz)
HOP_LENGTH = 128              # Hop length (8 ms @ 16 kHz, 75% overlap)
WIN_LENGTH = 512              # Hann window length
N_FREQ_BINS = 256             # Truncate 257 to 256 (power-of-2) for smooth U-Net pooling
AUDIO_DURATION = 2.0          # Segment length in seconds during training
SEGMENT_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)  # 32,000 samples

# Dataset & Noise Augmentation
TRAIN_SNR_RANGE = (-5.0, 15.0)  # Random SNR range (dB) for dynamic mixing
VAL_SNR_LIST = [0.0, 5.0, 10.0] # Fixed SNR evaluation points

# Model Architecture Parameters
MODEL_CHANNELS = [32, 64, 128, 256, 512]  # U-Net encoder channel progression
USE_BOTTLENECK_LSTM = True                # Enable bidirectional GRU/LSTM for temporal speech memory
DROPOUT_RATE = 0.1

# Training Hyperparameters
BATCH_SIZE = 8 if not torch.cuda.is_available() else 16
NUM_EPOCHS = 30
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Power Law Compression for Spectrogram Loss
SPECTRAL_COMPRESSION_POWER = 0.3
