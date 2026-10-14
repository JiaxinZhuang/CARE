"""Extract frozen CARE features for CT-RATE linear probing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


TASK_DIR = Path(__file__).resolve().parent
CARE_ROOT = TASK_DIR.parents[2]
if str(CARE_ROOT) not in sys.path:
    sys.path.insert(0, str(CARE_ROOT))

from linear_probe_common import (  # noqa: E402
    atomic_save_npz,
    atomic_write_json,
    ensure_external_artifact_path,
    file_fingerprint,
    set_reproducible_seed,
    sha256_file,
)
from model_loading.models import get_model  # noqa: E402


DEFAULT_CHECKPOINT = CARE_ROOT / "ckpts" / "care.pth"
MODEL_INPUT_SIZE = (112, 336, 336)
IMAGE_NET_SCALAR_MEAN = float(torch.as_tensor((0.485, 0.456, 0.406)).mean())
IMAGE_NET_SCALAR_STD = float(torch.as_tensor((0.229, 0.224, 0.225)).mean())
POOLING_CHOICES = ("cls", "mean", "cls-mean", "cls-mean-max")
NPZ_KEYS = {
    "features",
    "labels",
    "case_ids",
    "group_ids",
    "classnames",
    "metadata_json",
}


def _load_ctrate_dataset_class():
    path = TASK_DIR / "datasets" / "ctrate.py"
    spec = importlib.util.spec_from_file_location("care_ln_ctrate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CT-RATE dataset module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DataSet


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train", "validation", "valid"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--label-path", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--ours-pooling", required=True, choices=POOLING_CHOICES)
    parser.add_argument("--z-windows", required=True, type=int, choices=[1, 3])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--use-amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args(argv)


def make_z_windows(volumes: torch.Tensor, window_depth: int = 112) -> torch.Tensor:
    if volumes.ndim != 5 or volumes.shape[0] != 1:
        raise ValueError("z-window extraction requires one [B,C,D,H,W] volume")
    depth = volumes.shape[2]
    if depth <= window_depth:
        pad = window_depth - depth
        return F.pad(
            volumes,
            (0, 0, 0, 0, pad // 2, pad - pad // 2),
            value=0,
        )
    maximum_start = depth - window_depth
    starts = sorted({0, int(round(maximum_start / 2)), maximum_start})
    return torch.cat(
        [volumes[:, :, start : start + window_depth] for start in starts], dim=0
    )


def preprocess_for_care(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if images.ndim != 5:
        raise ValueError(f"Expected [B,C,D,H,W], got {tuple(images.shape)}")
    images = F.interpolate(
        images,
        size=MODEL_INPUT_SIZE,
        mode="trilinear",
        align_corners=False,
    )
    return (images - IMAGE_NET_SCALAR_MEAN) / IMAGE_NET_SCALAR_STD


def compose_image_feature(components, pooling: str) -> torch.Tensor:
    names = pooling.split("-")
    unknown = [name for name in names if name not in components]
    if unknown:
        raise ValueError(f"Unsupported image pooling components: {unknown}")
    return F.normalize(torch.cat([components[name] for name in names], dim=-1), dim=-1)


def encode_projected_features(
    visual: torch.nn.Module, images: torch.Tensor, pooling: str
) -> torch.Tensor:
    output = visual(preprocess_for_care(images))
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("CARE visual must return projected (pooled, tokens)")
    pooled, tokens = output
    if pooled.ndim != 2 or tokens.ndim != 3 or pooled.shape[0] != tokens.shape[0]:
        raise RuntimeError(
            f"Unexpected CARE visual outputs: pooled={tuple(pooled.shape)}, "
            f"tokens={tuple(tokens.shape)}"
        )
    components = {
        "cls": F.normalize(pooled, dim=-1),
        "mean": F.normalize(tokens.mean(dim=1), dim=-1),
        "max": F.normalize(tokens.amax(dim=1), dim=-1),
    }
    return compose_image_feature(components, pooling)


def encode_volume_features(
    visual: torch.nn.Module,
    volumes: torch.Tensor,
    pooling: str,
    z_windows: int,
) -> tuple[torch.Tensor, list[int]]:
    if z_windows == 1:
        return encode_projected_features(visual, volumes, pooling), [1] * len(volumes)
    if z_windows != 3:
        raise ValueError(f"Unsupported z-window count: {z_windows}")
    windows = make_z_windows(volumes, window_depth=MODEL_INPUT_SIZE[0])
    window_features = encode_projected_features(visual, windows, pooling)
    features = F.normalize(window_features.mean(dim=0, keepdim=True), dim=-1)
    return features, [len(windows)]


def _selected_indices(dataset, manifest_path: Path | None, limit: int | None):
    case_ids = [str(sample["case_id"]) for sample in dataset.samples]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("CT-RATE labels contain duplicate case IDs")
    indices = list(range(len(dataset)))
    if manifest_path is not None:
        manifest = pd.read_csv(manifest_path)
        if "case_id" not in manifest.columns:
            raise ValueError(f"Manifest has no case_id column: {manifest_path}")
        requested = manifest["case_id"].astype(str).tolist()
        if len(requested) != len(set(requested)):
            raise ValueError(f"Duplicate case IDs in manifest: {manifest_path}")
        index_by_case = {case_id: index for index, case_id in enumerate(case_ids)}
        missing = [case_id for case_id in requested if case_id not in index_by_case]
        if missing:
            raise ValueError(
                f"Manifest has {len(missing)} cases absent from labels; "
                f"examples: {missing[:5]}"
            )
        indices = [index_by_case[case_id] for case_id in requested]
    if limit is not None:
        indices = indices[:limit]
    return indices


def _selection_sha256(dataset, indices) -> str:
    selected = [
        [
            dataset.samples[index]["case_id"],
            dataset.samples[index]["group_id"],
            dataset.samples[index]["image_path"],
            dataset.samples[index]["labels"],
        ]
        for index in indices
    ]
    payload = json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _feature_definition(args, checkpoint_sha256: str) -> dict:
    components = args.ours_pooling.split("-")
    return {
        "ours_pooling": args.ours_pooling,
        "pooling_components": components,
        "pooling_component_count": len(components),
        "z_windows": args.z_windows,
        "window_depth": MODEL_INPUT_SIZE[0],
        "possible_unique_window_counts": [1] if args.z_windows == 1 else [1, 2, 3],
        "checkpoint_sha256": checkpoint_sha256,
        "window": [-1150.0, 350.0],
        "size": list(MODEL_INPUT_SIZE),
        "model_normalization": [IMAGE_NET_SCALAR_MEAN, IMAGE_NET_SCALAR_STD],
        "feature_dim": 1024 * len(components),
        "normalization_version": "component-l2_concat-l2_window-mean-l2_v1",
        "pooling_version": "care-visual-projected-tuple_v1",
    }


def _validate_reuse(
    output_path: Path,
    summary_path: Path,
    expected: dict,
    dataset,
    indices,
) -> dict:
    if not output_path.is_file() or not summary_path.is_file():
        raise FileExistsError(
            "Reuse requires both the NPZ and its .summary.json sidecar"
        )
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"Existing feature metadata mismatch for {key}: "
                f"{summary.get(key)!r} != {value!r}"
            )
    if "ours_pooling" not in summary or "z_windows" not in summary:
        raise ValueError("Existing metadata lacks explicit ours_pooling/z_windows")

    expected_case_ids = np.asarray(
        [str(dataset.samples[index]["case_id"]) for index in indices], dtype=str
    )
    expected_group_ids = np.asarray(
        [str(dataset.samples[index]["group_id"]) for index in indices], dtype=str
    )
    expected_labels = np.asarray(
        [dataset.samples[index]["labels"] for index in indices], dtype=np.int8
    )
    with np.load(output_path, allow_pickle=False) as artifact:
        if set(artifact.files) != NPZ_KEYS:
            raise ValueError(
                f"Existing NPZ schema mismatch: {sorted(artifact.files)}"
            )
        embedded = json.loads(str(artifact["metadata_json"].item()))
        if embedded != summary:
            raise ValueError("NPZ metadata_json differs from summary metadata")
        if not np.array_equal(artifact["case_ids"].astype(str), expected_case_ids):
            raise ValueError("Existing NPZ case IDs do not match the requested selection")
        if not np.array_equal(artifact["group_ids"].astype(str), expected_group_ids):
            raise ValueError("Existing NPZ group IDs do not match the requested selection")
        if not np.array_equal(artifact["labels"], expected_labels):
            raise ValueError("Existing NPZ labels do not match the requested selection")
        if not np.array_equal(
            artifact["classnames"].astype(str), np.asarray(dataset.get_classnames())
        ):
            raise ValueError("Existing NPZ class names do not match CT-RATE")
        features = artifact["features"]
        if features.dtype != np.float32 or features.ndim != 2:
            raise ValueError(f"Invalid existing feature array: {features.shape}/{features.dtype}")
        if features.shape != (len(indices), expected["feature_definition"]["feature_dim"]):
            raise ValueError(f"Existing feature shape mismatch: {features.shape}")
        if not np.isfinite(features).all():
            raise FloatingPointError("Existing features contain non-finite values")
        deviation = np.abs(np.linalg.norm(features, axis=1) - 1.0)
        if deviation.size and float(deviation.max()) > 0.02:
            raise ValueError("Existing features fail the L2-normalization gate")
    return summary


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid data-loader settings")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.z_windows == 3 and args.batch_size != 1:
        raise ValueError("--z-windows 3 requires --batch-size 1")

    output_path = ensure_external_artifact_path(Path(args.output))
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    label_path = Path(args.label_path).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CARE checkpoint not found: {checkpoint_path}")
    if manifest_path is not None and not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    set_reproducible_seed(args.seed)
    transform_mode = "z_raw" if args.z_windows == 3 else "crop"
    dataset_class = _load_ctrate_dataset_class()
    dataset = dataset_class(
        task="ctrate",
        split=args.split,
        data_root=str(data_root),
        label_path=str(label_path),
        transform_mode=transform_mode,
    )
    indices = _selected_indices(dataset, manifest_path, args.limit)
    if not indices:
        raise ValueError("The requested CT-RATE selection is empty")

    checkpoint_sha256 = sha256_file(checkpoint_path)
    feature_definition = _feature_definition(args, checkpoint_sha256)
    device = torch.device(args.device)
    amp_enabled = bool(args.use_amp and device.type == "cuda")
    source_config = {
        "extractor_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_sha256": sha256_file(TASK_DIR / "datasets" / "ctrate.py"),
    }
    expected_metadata = {
        "dataset": "ctrate",
        "split": "validation" if args.split == "valid" else args.split,
        "model": "care",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_fingerprint": file_fingerprint(checkpoint_path),
        "ours_pooling": args.ours_pooling,
        "z_windows": args.z_windows,
        "feature_definition": feature_definition,
        "preprocessing": dataset.linear_probe_preprocessing,
        "source_config": source_config,
        "source_label_path": str(label_path),
        "source_label_sha256": sha256_file(label_path),
        "data_root": str(data_root),
        "manifest": str(manifest_path) if manifest_path else None,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path else None,
        "selected_case_ids_sha256": _selection_sha256(dataset, indices),
        "samples": len(indices),
        "source_split_samples": len(dataset),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": args.device,
        "amp": amp_enabled,
        "limit": args.limit,
    }

    if output_path.exists() or summary_path.exists():
        if not args.reuse_existing:
            raise FileExistsError(f"Refusing to overwrite feature artifacts: {output_path}")
        metadata = _validate_reuse(
            output_path, summary_path, expected_metadata, dataset, indices
        )
        print(json.dumps({"reused": str(output_path), "metadata": metadata}, indent=2))
        return

    extraction_dataset = Subset(dataset, indices)
    loader = DataLoader(
        extraction_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    care_model = get_model(args.device, str(checkpoint_path))
    visual = care_model.visual
    visual.eval().requires_grad_(False)
    del care_model

    feature_batches = []
    label_batches = []
    case_ids = []
    group_ids = []
    window_counts = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"care:ctrate:{args.split}"):
            volumes = batch["volume"].to(device, non_blocking=True)
            if not torch.isfinite(volumes).all():
                raise FloatingPointError(
                    f"Non-finite input values for cases: {list(batch['case_id'])}"
                )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                features, counts = encode_volume_features(
                    visual, volumes, args.ours_pooling, args.z_windows
                )
            if features.ndim != 2 or not torch.isfinite(features).all():
                raise FloatingPointError(
                    f"Invalid features for cases: {list(batch['case_id'])}"
                )
            feature_batches.append(features.float().cpu().numpy())
            label_batches.append(batch["labels"].numpy().astype(np.int8, copy=False))
            case_ids.extend(str(value) for value in batch["case_id"])
            group_ids.extend(str(value) for value in batch["group_id"])
            window_counts.extend(counts)

    features = np.concatenate(feature_batches, axis=0).astype(np.float32, copy=False)
    labels = np.concatenate(label_batches, axis=0).astype(np.int8, copy=False)
    case_ids_array = np.asarray(case_ids, dtype=str)
    group_ids_array = np.asarray(group_ids, dtype=str)
    classnames = dataset.get_classnames()
    expected_shape = (len(indices), feature_definition["feature_dim"])
    if features.shape != expected_shape:
        raise AssertionError(f"Feature shape mismatch: {features.shape} != {expected_shape}")
    if labels.shape != (len(indices), len(classnames)):
        raise AssertionError(f"Label shape mismatch: {labels.shape}")
    if len(np.unique(case_ids_array)) != len(indices):
        raise AssertionError("Duplicate case IDs in extracted features")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Non-binary CT-RATE labels detected")
    feature_norms = np.linalg.norm(features, axis=1)
    maximum_deviation = float(np.max(np.abs(feature_norms - 1.0)))
    if maximum_deviation > 0.02:
        raise RuntimeError(
            f"Feature normalization gate failed: {maximum_deviation:.6f}"
        )

    metadata = dict(expected_metadata)
    metadata.update(
        {
            "status": "complete",
            "equivalence": "behavioral_port_only",
            "feature_dim": int(features.shape[1]),
            "feature_dtype": str(features.dtype),
            "label_count": len(classnames),
            "classnames": classnames,
            "feature_norm_mean": float(feature_norms.mean()),
            "feature_norm_max_deviation": maximum_deviation,
            "window_count_histogram": {
                str(count): int(window_counts.count(count))
                for count in sorted(set(window_counts))
            },
        }
    )
    atomic_save_npz(
        output_path,
        features=features,
        labels=labels,
        case_ids=case_ids_array,
        group_ids=group_ids_array,
        classnames=np.asarray(classnames, dtype=str),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    atomic_write_json(summary_path, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
