"""
Loss functions for speech enhancement and spectrogram denoising.
Includes power-law compressed spectral loss, mask loss, and multi-scale objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class DenoisingLoss(nn.Module):
    """
    Combined speech enhancement loss:
    1. Perceptually compressed spectrogram L1 loss (|S|^p - |S_hat|^p)
    2. Linear magnitude L1 loss
    3. Ideal Ratio Mask (IRM) guidance loss
    """
    def __init__(self, power: float = config.SPECTRAL_COMPRESSION_POWER, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.3):
        super().__init__()
        self.power = power
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        pred_mag: torch.Tensor,
        target_mag: torch.Tensor,
        noisy_mag: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_mag: (B, 1, F, T) estimated clean magnitude
            target_mag: (B, 1, F, T) ground truth clean magnitude
            noisy_mag: (B, 1, F, T) input noisy magnitude
            pred_mask: (B, 1, F, T) predicted ratio mask [0, 1]
        """
        eps = 1e-8

        # 1. Power-law compressed spectral L1 loss (approximates human loudness perception)
        pred_comp = torch.pow(torch.clamp(pred_mag, min=eps), self.power)
        target_comp = torch.pow(torch.clamp(target_mag, min=eps), self.power)
        loss_compressed = F.l1_loss(pred_comp, target_comp)

        # 2. Linear magnitude L1 loss
        loss_mag = F.l1_loss(pred_mag, target_mag)

        # 3. Target Ideal Ratio Mask (IRM) supervision
        target_mask = torch.clamp(target_mag / (noisy_mag + eps), min=0.0, max=1.0)
        loss_mask = F.l1_loss(pred_mask, target_mask)

        # Total combined loss
        total_loss = self.alpha * loss_compressed + self.beta * loss_mag + self.gamma * loss_mask
        return total_loss
