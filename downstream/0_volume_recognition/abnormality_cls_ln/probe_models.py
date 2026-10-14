"""Affine probe head used by cached-feature training and evaluation."""

from __future__ import annotations

from typing import Any

from torch import nn


def make_probe_spec(
    probe_type: str,
    nonlinear_classnames: list[str] | None = None,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    topk: int = 8,
    mean_weight: float = 0.5,
) -> dict[str, Any]:
    del hidden_dim, dropout, topk, mean_weight
    if probe_type != "linear":
        raise ValueError("Only an affine linear probe is supported")
    if nonlinear_classnames:
        raise ValueError("Nonlinear probe classes are not supported")
    return {"type": "linear"}


def build_probe(
    feature_dim: int,
    classnames: list[str],
    spec: dict[str, Any] | None = None,
) -> nn.Module:
    spec = spec or {"type": "linear"}
    if spec.get("type", "linear") != "linear":
        raise ValueError(f"Only an affine linear probe is supported: {spec}")
    if spec.get("nonlinear_classnames"):
        raise ValueError("Nonlinear probe classes are not supported")
    if feature_dim < 1:
        raise ValueError("feature_dim must be positive")
    if not classnames or len(classnames) != len(set(classnames)):
        raise ValueError("classnames must be non-empty and unique")
    return nn.Linear(feature_dim, len(classnames))
