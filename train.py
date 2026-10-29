"""
Training Pipeline for U-Net Speech Denoising Neural Network.
"""

import os
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import config
from src.model import SpeechUNet
from src.dataset import create_dataloaders
from src.losses import DenoisingLoss
from src.audio_processing import compute_istft, save_audio
from src.metrics import calculate_snr


def train(args):
    device = torch.device(config.DEVICE)
    print(f"--> Using device: {device}")

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = config.OUTPUT_DIR / "training_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Dataloaders
    print("--> Preparing audio datasets...")
    train_loader, val_loader = create_dataloaders(
        clean_dir=Path(args.clean_dir),
        noise_dir=Path(args.noise_dir),
        batch_size=args.batch_size,
    )
    print(f"--> Train batches: {len(train_loader)} | Validation batches: {len(val_loader)}")

    # Model, Loss, Optimizer
    model = SpeechUNet().to(device)
    criterion = DenoisingLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--> SpeechUNet total trainable parameters: {total_params:,}")

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # ---------------- TRAINING ----------------
        model.train()
        running_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar:
            noisy_mag = batch["noisy_mag"].to(device)
            clean_mag = batch["clean_mag"].to(device)

            optimizer.zero_grad()
            denoised_mag, mask = model(noisy_mag)

            loss = criterion(denoised_mag, clean_mag, noisy_mag, mask)
            loss.backward()

            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            running_train_loss += loss.item() * noisy_mag.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ---------------- VALIDATION ----------------
        model.eval()
        running_val_loss = 0.0
        snr_improvements = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                noisy_mag = batch["noisy_mag"].to(device)
                clean_mag = batch["clean_mag"].to(device)
                noisy_phase = batch["noisy_phase"].to(device)
                clean_wav = batch["clean_wav"]
                noisy_wav = batch["noisy_wav"]

                denoised_mag, mask = model(noisy_mag)
                loss = criterion(denoised_mag, clean_mag, noisy_mag, mask)
                running_val_loss += loss.item() * noisy_mag.size(0)

                # Time-domain reconstruction for delta-SNR check
                pred_wav = compute_istft(
                    denoised_mag.squeeze(1),
                    noisy_phase,
                    n_fft=config.N_FFT,
                    hop_length=config.HOP_LENGTH,
                    win_length=config.WIN_LENGTH,
                    length=config.SEGMENT_SAMPLES,
                ).cpu()

                for i in range(min(4, noisy_mag.size(0))):
                    noisy_snr = calculate_snr(clean_wav[i], noisy_wav[i])
                    denoised_snr = calculate_snr(clean_wav[i], pred_wav[i])
                    snr_improvements.append(denoised_snr - noisy_snr)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        avg_snr_gain = np.mean(snr_improvements) if snr_improvements else 0.0

        scheduler.step(epoch_val_loss)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Avg SNR Gain: {avg_snr_gain:+.2f} dB | "
            f"Time: {elapsed:.1f}s"
        )

        # Save Best Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "avg_snr_gain": avg_snr_gain,
                },
                checkpoint_dir / "best_model.pt",
            )
            print(f"  [+] Saved new best model checkpoint (Val Loss: {best_val_loss:.4f})")

        # Save Latest Checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": epoch_val_loss,
            },
            checkpoint_dir / "latest_model.pt",
        )

        # Export sample audio from validation set
        if epoch % 5 == 0 or epoch == args.epochs:
            with torch.no_grad():
                first_batch = next(iter(val_loader))
                n_mag = first_batch["noisy_mag"][:1].to(device)
                n_phase = first_batch["noisy_phase"][:1].to(device)
                c_wav = first_batch["clean_wav"][0]
                n_wav = first_batch["noisy_wav"][0]

                d_mag, _ = model(n_mag)
                d_wav = compute_istft(
                    d_mag.squeeze(1),
                    n_phase,
                    length=config.SEGMENT_SAMPLES,
                ).cpu().squeeze(0)

                save_audio(str(samples_dir / f"epoch_{epoch}_clean.wav"), c_wav)
                save_audio(str(samples_dir / f"epoch_{epoch}_noisy.wav"), n_wav)
                save_audio(str(samples_dir / f"epoch_{epoch}_denoised.wav"), d_wav)

    # Save training curves plot
    plt.figure(figsize=(9, 4.5))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss", color="#2563eb", linewidth=2)
    plt.plot(range(1, args.epochs + 1), val_losses, label="Validation Loss", color="#dc2626", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("U-Net Speech Denoising Training & Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "loss_curve.png", dpi=300)
    plt.close()
    print(f"--> Saved loss curve plot to: {config.OUTPUT_DIR / 'loss_curve.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net Speech Denoising Model")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--clean_dir", type=str, default=str(config.CLEAN_DIR), help="Clean speech directory")
    parser.add_argument("--noise_dir", type=str, default=str(config.NOISE_DIR), help="Noise directory")
    parser.add_argument("--checkpoint_dir", type=str, default=str(config.CHECKPOINT_DIR), help="Checkpoint directory")
    args = parser.parse_args()

    train(args)
