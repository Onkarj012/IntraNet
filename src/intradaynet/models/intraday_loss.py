"""
Intraday-specific loss function for IntradayNet.

Combines:
- Focal loss for direction prediction (asymmetric gamma for up/down)
- Huber loss for magnitude prediction
- BCE for confidence calibration
- Time-weighted loss (predictions near market open get extra weight)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class IntradayLoss(nn.Module):
    """
    Combined loss for intraday multi-horizon predictions.

    L = w_dir * FocalLoss(direction) + w_mag * HuberLoss(magnitude)
      + w_conf * BCE(confidence, was_correct)

    With optional time-weighting: early-session predictions (first 45 min)
    get `time_weight_open` multiplier since they are most actionable.
    """

    def __init__(
        self,
        direction_weight: float = 0.5,
        magnitude_weight: float = 0.3,
        confidence_weight: float = 0.2,
        gamma_pos: float = 2.0,
        gamma_neg: float = 2.0,
        downside_weight: float = 1.2,
        time_weight_open: float = 1.5,
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.confidence_weight = confidence_weight
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.downside_weight = downside_weight
        self.time_weight_open = time_weight_open

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        time_normalized: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            predictions: dict with:
                direction_logits: (B, H) raw logits
                magnitudes: (B, H) predicted magnitudes
                confidences: (B, H) confidence scores [0,1]
            targets: dict with:
                direction: (B, H) binary labels (1=up, 0=down)
                magnitude: (B, H) actual returns
            time_normalized: (B,) normalized time-of-day [0,1]
                             if provided, early bars get extra weight

        Returns:
            dict with total_loss and component losses
        """
        pred_dir = predictions["direction_logits"]      # (B, H)
        pred_mag = predictions["magnitudes"]             # (B, H)
        pred_conf = predictions["confidences"]           # (B, H)

        tgt_dir = targets["direction"]                   # (B, H)
        tgt_mag = targets["magnitude"]                   # (B, H)

        # ── 1. Direction loss (focal, asymmetric) ──
        dir_loss = self._focal_loss(pred_dir, tgt_dir)

        # ── 2. Magnitude loss (Huber / smooth L1) ──
        mag_loss = F.smooth_l1_loss(pred_mag, tgt_mag, reduction='none')
        mag_loss = mag_loss.mean(dim=1)  # mean across horizons

        # ── 3. Confidence loss (BCE) ──
        # Target: was the direction prediction correct?
        pred_direction = (torch.sigmoid(pred_dir) > 0.5).float()
        was_correct = (pred_direction == tgt_dir).float().detach()
        conf_loss = F.binary_cross_entropy(
            pred_conf, was_correct, reduction='none'
        ).mean(dim=1)

        # ── Combine ──
        total = (
            self.direction_weight * dir_loss
            + self.magnitude_weight * mag_loss
            + self.confidence_weight * conf_loss
        )

        # ── Time weighting (optional) ──
        if time_normalized is not None:
            # First 45 min of session → time_normalized < 0.12
            time_weight = torch.where(
                time_normalized < 0.12,
                torch.full_like(time_normalized, self.time_weight_open),
                torch.ones_like(time_normalized),
            )
            total = total * time_weight

        total_loss = total.mean()

        return {
            "total_loss": total_loss,
            "direction_loss": dir_loss.mean(),
            "magnitude_loss": mag_loss.mean(),
            "confidence_loss": conf_loss.mean(),
        }

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Asymmetric focal loss for direction prediction.

        Args:
            logits: (B, H) raw logits
            targets: (B, H) binary targets

        Returns:
            (B,) per-sample loss (mean across horizons)
        """
        probs = torch.sigmoid(logits)
        eps = 1e-7

        # Focal modulation
        p_t = probs * targets + (1 - probs) * (1 - targets)
        p_t = p_t.clamp(eps, 1 - eps)

        # Asymmetric gamma
        gamma = torch.where(targets == 1, self.gamma_pos, self.gamma_neg)
        focal_weight = (1 - p_t) ** gamma

        # Asymmetric class weight (downside misses penalized more)
        class_weight = torch.where(targets == 0, self.downside_weight, 1.0)

        # BCE
        bce = -targets * torch.log(probs + eps) - (1 - targets) * torch.log(1 - probs + eps)

        loss = focal_weight * class_weight * bce
        return loss.mean(dim=1)  # mean across horizons
