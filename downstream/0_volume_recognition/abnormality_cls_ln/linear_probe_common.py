"""Shared helpers for the CT-RATE linear-probe pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[4]

CTRATE_REPORT_CLASSNAMES = [
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Hiatal hernia",
    "Arterial wall calcification",
]

CTRATE_PAPER14_CLASSNAMES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Emphysema",
    "Atelectasis",
    "Lung opacity",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]


def ensure_external_artifact_path(path: Path) -> Path:
    """Reject generated artifacts inside the source checkout."""
    resolved = path.expanduser().resolve()
    if resolved == SOURCE_ROOT or SOURCE_ROOT in resolved.parents:
        raise ValueError(
            f"Generated artifacts must be outside the repository: {resolved}"
        )
    return resolved


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> Dict[str, Any]:
    """Cheap provenance for very large pretrained checkpoints."""
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path = ensure_external_artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(partial, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path = ensure_external_artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, path)


def atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path = ensure_external_artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(payload, partial)
    os.replace(partial, path)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
