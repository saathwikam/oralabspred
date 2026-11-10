"""Masked, endpoint-specific weighted binary cross-entropy for AI-Based Oral Absorption Prediction System.

Targets and logits use column order HIA, HOB20, HOB50 and shape [batch, 3].
NaN targets mean unassayed and are excluded from both loss sums and counts.
Pass raw logits: this module intentionally does not apply sigmoid.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ENDPOINTS = ("HIA", "HOB20", "HOB50")


class MaskedWeightedBCELoss(nn.Module):
    """Compute per-endpoint masked BCEWithLogits with training-only weights.

    Args:
        pos_weights: Three finite nonnegative ``N_negative / N_positive``
            values, calculated from the training split only, in endpoint order.
        endpoint_names: Optional names for reporting; must contain three names.

    Returns a scalar mean over endpoint-specific mean losses for endpoints
    represented in the batch. With ``return_details=True``, also returns a
    dictionary mapping each endpoint to its mean loss, or ``None`` if that
    endpoint has no observed targets in this batch.
    """

    def __init__(
        self,
        pos_weights: Sequence[float],
        endpoint_names: Sequence[str] = ENDPOINTS,
    ) -> None:
        super().__init__()
        if len(pos_weights) != 3:
            raise ValueError("pos_weights must contain exactly three values in HIA, HOB20, HOB50 order.")
        if len(endpoint_names) != 3:
            raise ValueError("endpoint_names must contain exactly three names.")
        weight_tensor = torch.as_tensor(pos_weights, dtype=torch.float32)
        if not torch.isfinite(weight_tensor).all() or (weight_tensor < 0).any():
            raise ValueError("Every endpoint needs a finite, nonnegative training-derived pos_weight.")
        self.register_buffer("pos_weights", weight_tensor)
        self.endpoint_names = tuple(str(name) for name in endpoint_names)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        *,
        return_details: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Optional[Tensor]]]:
        """Return total loss and optionally the separate observed task losses."""
        if logits.ndim != 2 or tuple(logits.shape) != tuple(targets.shape):
            raise ValueError("logits and targets must have the same [batch_size, 3] shape.")
        if logits.shape[1] != len(self.endpoint_names):
            raise ValueError("Second tensor dimension must follow HIA, HOB20, HOB50.")
        target_values = targets.to(device=logits.device, dtype=logits.dtype)
        if torch.isinf(target_values).any():
            raise ValueError("Missing targets must be NaN; infinite target values are invalid.")
        mask = torch.isfinite(target_values)
        observed = target_values[mask]
        if observed.numel() and not torch.logical_or(observed == 0, observed == 1).all():
            raise ValueError("Observed targets must be binary 0/1 values or NaN.")

        safe_targets = torch.where(mask, target_values, torch.zeros_like(target_values))
        elementwise = F.binary_cross_entropy_with_logits(
            logits,
            safe_targets,
            pos_weight=self.pos_weights.to(device=logits.device, dtype=logits.dtype),
            reduction="none",
        )

        endpoint_losses: dict[str, Optional[Tensor]] = {}
        observed_losses: list[Tensor] = []
        for column, endpoint in enumerate(self.endpoint_names):
            endpoint_mask = mask[:, column]
            if endpoint_mask.any():
                task_loss = elementwise[endpoint_mask, column].mean()
                endpoint_losses[endpoint] = task_loss
                observed_losses.append(task_loss)
            else:
                endpoint_losses[endpoint] = None
        total = torch.stack(observed_losses).mean() if observed_losses else logits.sum() * 0.0
        return (total, endpoint_losses) if return_details else total
