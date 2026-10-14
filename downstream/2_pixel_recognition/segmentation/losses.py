from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedSegmentationLoss(nn.Module):
    """Foreground Dice plus masked CE/focal and optional boundary Dice."""

    def __init__(
        self,
        num_classes: int,
        class_weights: list[float],
        variant: str = "dice_ce",
        boundary_weight: float = 0.1,
        focal_gamma: float = 2.0,
        boundary_classes: tuple[int, ...] | None = None,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if variant not in {"dice_ce", "dice_focal", "dice_ce_boundary"}:
            raise ValueError(f"Unknown loss variant: {variant}")
        if len(class_weights) != num_classes:
            raise ValueError("class_weights length must match num_classes")
        weights = torch.tensor(class_weights, dtype=torch.float32)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("class_weights must be finite and non-negative")
        self.num_classes = num_classes
        self.variant = variant
        self.boundary_weight = float(boundary_weight)
        self.focal_gamma = float(focal_gamma)
        self.boundary_classes = boundary_classes
        if boundary_classes is not None:
            if not boundary_classes or any(
                class_index <= 0 or class_index >= num_classes
                for class_index in boundary_classes
            ):
                raise ValueError("boundary_classes must contain foreground class indices")
        self.register_buffer("class_weights", weights)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _validate_inputs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.ndim < 4:
            raise ValueError(f"Expected logits [B,C,...], got {tuple(logits.shape)}")
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} logit classes, got {logits.shape[1]}"
            )
        if labels.ndim != logits.ndim or labels.shape[1] != 1:
            raise ValueError(f"Expected labels [B,1,...], got {tuple(labels.shape)}")
        if labels.shape[0] != logits.shape[0] or labels.shape[2:] != logits.shape[2:]:
            raise ValueError(
                f"Logit/label shape mismatch: {tuple(logits.shape)}, "
                f"{tuple(labels.shape)}"
            )
        if valid_mask is None:
            valid = torch.ones_like(labels[:, :1], dtype=logits.dtype)
        else:
            if valid_mask.shape != labels.shape:
                raise ValueError(
                    f"valid_mask must match labels, got {tuple(valid_mask.shape)} "
                    f"and {tuple(labels.shape)}"
                )
            valid = valid_mask[:, :1].to(device=logits.device, dtype=logits.dtype)
            if not torch.logical_or(valid == 0, valid == 1).all():
                raise ValueError("valid_mask must contain only boolean/0/1 values")

        target = labels[:, 0].to(device=logits.device, dtype=torch.long)
        valid_voxels = valid[:, 0].bool()
        invalid_class = valid_voxels & ((target < 0) | (target >= self.num_classes))
        if invalid_class.any():
            observed = torch.unique(target[invalid_class]).detach().cpu().tolist()
            raise ValueError(f"Valid voxels contain labels outside class range: {observed}")
        target = torch.where(valid_voxels, target, torch.zeros_like(target))
        return target, valid

    def _dice_loss(
        self,
        probabilities: torch.Tensor,
        target_onehot: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = probabilities[:, 1:] * valid
        target_onehot = target_onehot[:, 1:] * valid
        spatial_dims = tuple(range(2, probabilities.ndim))
        intersection = (probabilities * target_onehot).sum(dim=spatial_dims)
        denominator = probabilities.sum(dim=spatial_dims) + target_onehot.sum(
            dim=spatial_dims
        )
        return (1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()

    def _boundary_loss(
        self,
        probabilities: torch.Tensor,
        target_onehot: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if probabilities.ndim != 5:
            raise ValueError("dice_ce_boundary requires 3D logits shaped [B,C,D,H,W]")
        class_indices = (
            list(range(1, self.num_classes))
            if self.boundary_classes is None
            else list(self.boundary_classes)
        )
        probabilities = probabilities[:, class_indices] * valid
        target_onehot = target_onehot[:, class_indices] * valid

        def boundary(values):
            dilated = F.max_pool3d(values, kernel_size=3, stride=1, padding=1)
            eroded = -F.max_pool3d(-values, kernel_size=3, stride=1, padding=1)
            return (dilated - eroded).clamp_(0.0, 1.0)

        pred_boundary = boundary(probabilities)
        target_boundary = boundary(target_onehot)
        spatial_dims = tuple(range(2, probabilities.ndim))
        intersection = (pred_boundary * target_boundary).sum(dim=spatial_dims)
        denominator = pred_boundary.sum(dim=spatial_dims) + target_boundary.sum(
            dim=spatial_dims
        )
        return (1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Volumetric Dice/boundary sums can exceed FP16's finite range.
        logits = logits.float()
        target, valid = self._validate_inputs(logits, labels, valid_mask)
        target_onehot = F.one_hot(target, num_classes=self.num_classes)
        target_onehot = target_onehot.movedim(-1, 1).to(dtype=logits.dtype)
        probabilities = torch.softmax(logits, dim=1)

        dice = self._dice_loss(probabilities, target_onehot, valid)
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights.to(device=logits.device, dtype=logits.dtype),
            reduction="none",
        )
        if self.variant == "dice_focal":
            true_probability = probabilities.gather(1, target[:, None]).squeeze(1)
            ce = ce * (1.0 - true_probability).pow(self.focal_gamma)
        data_term = self._masked_mean(ce, valid[:, 0])
        loss = dice + data_term
        if self.variant == "dice_ce_boundary":
            loss = loss + self.boundary_weight * self._boundary_loss(
                probabilities, target_onehot, valid
            )
        return loss
