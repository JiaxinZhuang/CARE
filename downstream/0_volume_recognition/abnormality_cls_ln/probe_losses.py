"""Balanced binary cross-entropy for CT-RATE linear probes."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ProbeLoss(nn.Module):
    def __init__(
        self,
        classnames: list[str],
        pos_weight: torch.Tensor,
        spec: dict[str, Any],
    ) -> None:
        super().__init__()
        if spec.get("type", "balanced-bce") != "balanced-bce":
            raise ValueError(f"Only balanced BCE is supported: {spec}")
        if spec.get("target_classnames"):
            raise ValueError("Targeted losses are not supported")
        if pos_weight.ndim != 1 or len(pos_weight) != len(classnames):
            raise ValueError("pos_weight must have one value per class")
        if not torch.isfinite(pos_weight).all() or (pos_weight <= 0).any():
            raise ValueError("pos_weight values must be finite and positive")
        self.spec = {"type": "balanced-bce"}
        self.register_buffer("pos_weight", pos_weight.float())

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weight,
        )


def make_loss_spec(
    loss_type: str,
    target_classnames: list[str] | None = None,
    focal_gamma: float = 2.0,
    asl_gamma_pos: float = 1.0,
    asl_gamma_neg: float = 4.0,
    asl_clip: float = 0.05,
) -> dict[str, Any]:
    del focal_gamma, asl_gamma_pos, asl_gamma_neg, asl_clip
    if loss_type != "balanced-bce":
        raise ValueError("Only balanced BCE is supported")
    if target_classnames:
        raise ValueError("Targeted losses are not supported")
    return {"type": "balanced-bce"}


def build_probe_loss(
    classnames: list[str],
    pos_weight: torch.Tensor,
    spec: dict[str, Any] | None,
) -> ProbeLoss:
    return ProbeLoss(
        classnames=classnames,
        pos_weight=pos_weight,
        spec=spec or {"type": "balanced-bce"},
    )
