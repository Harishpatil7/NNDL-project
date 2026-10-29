"""
2D U-Net Neural Network for Spectrogram Speech Denoising.
Includes skip connections, multi-scale convolutional encoders/decoders,
and a temporal recurrent bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import config


class ConvBlock2D(nn.Module):
    """Convolution + BatchNorm + LeakyReLU with optional Dropout."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 2, padding: int = 2, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.bn(self.conv(x))))


class DeconvBlock2D(nn.Module):
    """Transposed Convolution + BatchNorm + LeakyReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 2, padding: int = 2, output_padding: int = 1, dropout: float = 0.0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.bn(self.deconv(x))))


class RecurrentBottleneck(nn.Module):
    """
    Bidirectional GRU at the bottleneck to capture temporal speech dynamics across frames.
    """
    def __init__(self, channels: int, freq_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.channels = channels
        self.freq_dim = freq_dim
        input_dim = channels * freq_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, input_dim)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        b, c, f, t = x.shape
        # Reshape to (B, T, C * F)
        x_flat = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        
        # Temporal modeling with residual connection
        out, _ = self.gru(x_flat)
        out = self.proj(out)
        out = self.ln(x_flat + out)

        # Reshape back to (B, C, F, T)
        out = out.view(b, t, c, f).permute(0, 2, 3, 1).contiguous()
        return out


class SpeechUNet(nn.Module):
    """
    U-Net architecture with skip connections for spectrogram speech denoising.
    Input:
        Noisy magnitude spectrogram (B, 1, F=256, T)
    Output:
        Estimated clean magnitude spectrogram (B, 1, F=256, T) and Mask (B, 1, F=256, T)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channel_list: List[int] = None,
        use_recurrent_bottleneck: bool = config.USE_BOTTLENECK_LSTM,
        dropout: float = config.DROPOUT_RATE,
    ):
        super().__init__()
        if channel_list is None:
            channel_list = config.MODEL_CHANNELS  # [32, 64, 128, 256, 512]

        c1, c2, c3, c4, c5 = channel_list
        self.use_recurrent = use_recurrent_bottleneck

        # Encoder (Downsampling by 2 at each level)
        # F: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.enc1 = ConvBlock2D(in_channels, c1, kernel_size=5, stride=2, padding=2)
        self.enc2 = ConvBlock2D(c1, c2, kernel_size=5, stride=2, padding=2)
        self.enc3 = ConvBlock2D(c2, c3, kernel_size=5, stride=2, padding=2)
        self.enc4 = ConvBlock2D(c3, c4, kernel_size=5, stride=2, padding=2)
        self.enc5 = ConvBlock2D(c4, c5, kernel_size=5, stride=2, padding=2)

        # Bottleneck (Freq dimension is 256 / 32 = 8)
        if self.use_recurrent:
            self.bottleneck = RecurrentBottleneck(channels=c5, freq_dim=8, hidden_dim=256)
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(c5, c5, kernel_size=3, padding=1),
                nn.BatchNorm2d(c5),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Decoder (Upsampling by 2 at each level + Skip connections)
        self.dec5 = DeconvBlock2D(c5, c4, kernel_size=5, stride=2, padding=2, output_padding=1, dropout=dropout)
        self.dec4_conv = ConvBlock2D(c4 * 2, c4, kernel_size=3, stride=1, padding=1)

        self.dec4 = DeconvBlock2D(c4, c3, kernel_size=5, stride=2, padding=2, output_padding=1, dropout=dropout)
        self.dec3_conv = ConvBlock2D(c3 * 2, c3, kernel_size=3, stride=1, padding=1)

        self.dec3 = DeconvBlock2D(c3, c2, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec2_conv = ConvBlock2D(c2 * 2, c2, kernel_size=3, stride=1, padding=1)

        self.dec2 = DeconvBlock2D(c2, c1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec1_conv = ConvBlock2D(c1 * 2, c1, kernel_size=3, stride=1, padding=1)

        self.dec1 = DeconvBlock2D(c1, 16, kernel_size=5, stride=2, padding=2, output_padding=1)

        # Output Mask Head: Produces Ideal Ratio Mask in [0.0, 1.0]
        self.mask_head = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def _pad_time(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pads the time dimension so it is evenly divisible by 32."""
        t = x.shape[-1]
        div = 32
        rem = t % div
        if rem != 0:
            pad = div - rem
            x = F.pad(x, (0, pad, 0, 0), mode="constant", value=0.0)
            return x, pad
        return x, 0

    def forward(self, noisy_mag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            noisy_mag: (B, 1, F=256, T) or (B, F=256, T)
        Returns:
            denoised_mag: (B, 1, F=256, T)
            mask: (B, 1, F=256, T)
        """
        if noisy_mag.ndim == 3:
            noisy_mag = noisy_mag.unsqueeze(1)

        # Pad time dimension if needed to allow 5 levels of stride-2 downsampling
        x, pad = self._pad_time(noisy_mag)

        # Encoder pass with skip connections
        e1 = self.enc1(x)   # (B, 32, 128, T/2)
        e2 = self.enc2(e1)  # (B, 64, 64, T/4)
        e3 = self.enc3(e2)  # (B, 128, 32, T/8)
        e4 = self.enc4(e3)  # (B, 256, 16, T/16)
        e5 = self.enc5(e4)  # (B, 512, 8, T/32)

        # Bottleneck
        b = self.bottleneck(e5)

        # Decoder pass with skip connections
        d5 = self.dec5(b)
        d5 = self.dec4_conv(torch.cat([d5, e4], dim=1))

        d4 = self.dec4(d5)
        d4 = self.dec3_conv(torch.cat([d4, e3], dim=1))

        d3 = self.dec3(d4)
        d3 = self.dec2_conv(torch.cat([d3, e2], dim=1))

        d2 = self.dec2(d3)
        d2 = self.dec1_conv(torch.cat([d2, e1], dim=1))

        d1 = self.dec1(d2)

        # Generate Ratio Mask
        mask = self.mask_head(d1)

        # Crop back to original time length if padded
        if pad > 0:
            mask = mask[..., :-pad]
            x = x[..., :-pad]

        # Apply mask to noisy input: Clean_Mag = Mask * Noisy_Mag
        denoised_mag = mask * noisy_mag

        return denoised_mag, mask
