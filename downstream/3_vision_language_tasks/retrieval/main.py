"""CARE zero-shot CT report/volume retrieval feature extraction and metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from datasets import ctrg as ctrg_dataset
from datasets import nifti as nifti_dataset
from metrics import compute_pooled_metrics, load_feature_cache
from models.ours import OursWrapper


LOGGER = logging.getLogger("care.retrieval")
RELEASE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = RELEASE_ROOT / "ckpts" / "care.pth"
NiftiRetrievalDataset = nifti_dataset.NiftiRetrievalDataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("extract", "metrics"))
    parser.add_argument("--manifest")
    parser.add_argument("--data_root")
    parser.add_argument("--checkpoint_path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--text_assets_path")
    parser.add_argument("--features_file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--text_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--pool_partitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _required(args: argparse.Namespace, *names: str) -> None:
    missing = [f"--{name}" for name in names if getattr(args, name) in (None, "")]
    if missing:
        raise ValueError(f"{args.mode} requires {', '.join(missing)}")


def _load_extraction_rows(
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    _required(args, "data_root")
    return ctrg_dataset.load_ctrg_rows(args.manifest, args.data_root)


def _device(name: str | None) -> torch.device:
    resolved = name or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(resolved)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {resolved}")
    return device


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)


def _preprocessing_metadata() -> Any:
    getter = getattr(nifti_dataset, "preprocessing_spec", None)
    if getter is not None:
        return getter("ctrg")
    registry = getattr(nifti_dataset, "PREPROCESSING", None)
    if registry is None or "ctrg" not in registry:
        raise ValueError("No preprocessing metadata found for ctrg")
    return registry["ctrg"]


@torch.no_grad()
def _encode_images(
    model: OursWrapper,
    rows: list[dict[str, str]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> np.ndarray:
    dataset = NiftiRetrievalDataset(rows, "ctrg")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    ordered: list[np.ndarray | None] = [None] * len(rows)
    amp_enabled = use_amp and device.type == "cuda"
    for batch in loader:
        volumes = batch["volume"].to(device)
        indices = batch["index"].tolist()
        with autocast(enabled=amp_enabled):
            features = model.encode_image(volumes)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        features = F.normalize(features.float(), dim=-1).cpu().numpy()
        if len(features) != len(indices):
            raise ValueError("Image encoder batch size does not match dataset indices")
        for index, feature in zip(indices, features):
            ordered[int(index)] = feature
    if any(feature is None for feature in ordered):
        raise ValueError("Image extraction did not return every manifest row")
    matrix = np.stack(ordered).astype(np.float32)
    if matrix.shape[1] != 1024:
        raise ValueError(f"CARE image embeddings must be 1024-D, got {matrix.shape}")
    return matrix


@torch.no_grad()
def _encode_texts(
    model: OursWrapper,
    rows: list[dict[str, str]],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    output = []
    amp_enabled = use_amp and device.type == "cuda"
    for start in range(0, len(rows), batch_size):
        texts = [row["text_en"] for row in rows[start:start + batch_size]]
        with autocast(enabled=amp_enabled):
            features = model.encode_text(texts)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        output.append(F.normalize(features.float(), dim=-1).cpu())
    matrix = torch.cat(output, dim=0).numpy().astype(np.float32)
    if matrix.shape != (len(rows), 1024):
        raise ValueError(f"CARE text embeddings must have shape ({len(rows)}, 1024), got {matrix.shape}")
    return matrix


def run_extract(args: argparse.Namespace) -> None:
    _required(args, "manifest", "data_root")
    if args.batch_size < 1 or args.text_batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch sizes must be positive and num_workers non-negative")
    rows, deduplication, source_details = _load_extraction_rows(args)
    device = _device(args.device)
    checkpoint = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")

    model = OursWrapper(
        str(checkpoint), device=str(device), text_assets_path=args.text_assets_path
    )
    if hasattr(model, "eval"):
        model.eval()
    image_volume_features = _encode_images(
        model, rows, device, args.batch_size,
        args.num_workers, args.use_amp,
    )
    text_features = _encode_texts(
        model, rows, args.text_batch_size, device, args.use_amp
    )
    image_features = image_volume_features
    ids = [row["pair_id"] for row in rows]
    patient_keys = [row.get("patient_key", "").strip() for row in rows]
    if any(patient_keys) and not all(patient_keys):
        raise ValueError("patient_key must be present for every row or omitted for all rows")
    if text_features.shape != image_features.shape:
        raise ValueError("Final paired text/image feature shapes do not match")

    public_source_details = dict(source_details)
    public_source_details.pop("data_root", None)
    public_source_details.pop("annotations_path", None)
    metadata = {
        "schema": "care_retrieval_features_v1",
        "preprocessing": _jsonable(_preprocessing_metadata()),
        "aggregation": "none",
        "source": {
            "manifest_source": "user-provided CTRG pair manifest",
            "site": "chest",
            "volume_count": len(rows),
            "candidate_count": len(ids),
            "text_deduplication": deduplication,
            **public_source_details,
        },
        "checkpoint_source": "user-provided CARE checkpoint",
        "checkpoint_provenance": "weights not verified by this cache",
        "feature_settings": {
            "use_amp": bool(args.use_amp),
            "text_assets_override": bool(args.text_assets_path),
        },
        "feature_dim": 1024,
        "normalized": True,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "features.npz"
    cache = {
        "text_features": text_features,
        "image_features": image_features,
        "ids": np.asarray(ids, dtype=np.str_),
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True), dtype=np.str_
        ),
    }
    if all(patient_keys):
        cache["patient_keys"] = np.asarray(patient_keys, dtype=np.str_)
    np.savez_compressed(output_path, **cache)
    LOGGER.info("Saved %d paired candidates to %s", len(ids), output_path)


def run_metrics(args: argparse.Namespace) -> None:
    _required(args, "features_file")
    cache = load_feature_cache(args.features_file)
    output_dir = Path(args.output_dir).expanduser().resolve()
    results = compute_pooled_metrics(
        cache["text_features"], cache["image_features"], cache["ids"],
        output_dir, pool_partitions=args.pool_partitions, seed=args.seed,
    )
    cache_metadata = cache["metadata"]
    checkpoint_provenance = "UNKNOWN"
    if isinstance(cache_metadata, dict):
        checkpoint_provenance = str(
            cache_metadata.get("checkpoint_provenance") or "UNKNOWN"
        )
    results.update({
        "features_source": "user-provided feature cache",
        "cache_protocol_metadata": cache_metadata,
        "cache_checkpoint_provenance": checkpoint_provenance,
        "legacy_pickle_ids": cache["legacy_pickle_ids"],
        "cache_weight_provenance_verified": False,
    })
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved pooled retrieval metrics to %s", output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parser().parse_args()
    if args.mode == "extract":
        run_extract(args)
    else:
        run_metrics(args)


if __name__ == "__main__":
    main()
