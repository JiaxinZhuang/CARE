from __future__ import annotations

import argparse
import csv
import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from datasets.kipa22 import (
    KiPA22Dataset,
    build_kipa22_transforms,
    get_kipa22_split_metadata,
)
from metrics import KIPA22_LABELS as LABELS
from metrics import PolyMetrics, prepare_metric_batch
from models.net import get_model


CARE_ROOT = Path(__file__).resolve().parents[3]


def default_data_dir() -> Path:
    return CARE_ROOT / "datasets" / "downstream" / "segmentation" / "kipa22"


def default_split_dir() -> Path:
    return CARE_ROOT / "splits" / "downstream" / "segmentation" / "kipa22"


def default_checkpoint_dir() -> Path:
    return CARE_ROOT / "ckpts" / "downstream" / "segmentation" / "kipa22"


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen CARE segmentation checkpoint."
    )
    parser.add_argument("--dataset", required=True, choices=("kipa22",))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--run-dir", default=None, type=Path)
    parser.add_argument("--config", default=None, type=Path)
    parser.add_argument("--checkpoint", default=None, type=Path)
    parser.add_argument("--data-dir", default=None, type=Path)
    split_source = parser.add_mutually_exclusive_group()
    split_source.add_argument("--split-dir", default=None, type=Path)
    split_source.add_argument("--manifest-path", default=None, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--sw-batch-size", default=None, type=int)
    parser.add_argument("--allow_noncanonical_split", action="store_true")
    return parser.parse_args()


def load_frozen_config(config_path: Path) -> Namespace:
    config_path = Path(config_path)
    if config_path.is_dir():
        config_path = config_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Evaluation config not found: {config_path}")
    config = json.loads(config_path.read_text())
    if str(config.get("dataset_name", "")).lower() != "kipa22":
        raise ValueError(f"Expected a KiPA22 config, got {config.get('dataset_name')}")
    return Namespace(**config)


def _config_path(args, value):
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute() and getattr(args, "path_base", None) == "care_root":
        path = CARE_ROOT / path
    return path


def _resolve_runtime(cli) -> tuple[Namespace, Path]:
    run_dir = cli.run_dir.resolve() if cli.run_dir else None
    resource_dir = default_checkpoint_dir()
    config_path = cli.config or (
        run_dir / "config.json" if run_dir else resource_dir / "config.json"
    )
    checkpoint_path = cli.checkpoint or (
        run_dir / cli.checkpoint_name
        if run_dir
        else resource_dir / "care_segmentation_kipa22.pt"
    )
    args = load_frozen_config(config_path)
    if str(args.dataset_name).lower() != cli.dataset:
        raise ValueError(
            f"Config dataset {args.dataset_name!r} does not match --dataset {cli.dataset!r}"
        )

    configured_data_dir = _config_path(args, getattr(args, "data_dir", None))
    args.data_dir = Path(cli.data_dir or configured_data_dir or default_data_dir())
    configured_manifest = _config_path(args, getattr(args, "manifest_path", None))
    configured_split_dir = _config_path(args, getattr(args, "split_dir", None))
    if cli.split_dir:
        args.split_dir = cli.split_dir
        args.manifest_path = None
    elif cli.manifest_path:
        args.split_dir = None
        args.manifest_path = cli.manifest_path
    elif configured_manifest:
        args.split_dir = None
        args.manifest_path = configured_manifest
    elif configured_split_dir:
        args.split_dir = configured_split_dir
        args.manifest_path = None
    elif args.data_dir.resolve() == default_data_dir().resolve():
        args.split_dir = default_split_dir()
        args.manifest_path = None
    else:
        args.split_dir = None
        args.manifest_path = args.data_dir / "manifest.csv"
    return args, Path(checkpoint_path).resolve()


def atomic_write_json(path: Path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, fieldnames, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_npz(path: Path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _preprocessed_spacing(data_dir: Path):
    plans_path = data_dir / "nnUNetPlans.json"
    if not plans_path.is_file():
        return None
    plans = json.loads(plans_path.read_text())
    configuration = plans.get("configurations", {}).get("3d_fullres", {})
    spacing = configuration.get("spacing")
    return None if spacing is None else np.asarray(spacing, dtype=np.float32)


def _nullable(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main():
    cli = get_args()
    if cli.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if cli.sw_batch_size is not None and cli.sw_batch_size < 1:
        raise ValueError("--sw-batch-size must be positive")

    args, checkpoint_path = _resolve_runtime(cli)
    args.num_workers = cli.num_workers
    if cli.sw_batch_size is not None:
        args.sw_batch_size = cli.sw_batch_size
    configured_allow_noncanonical = (
        bool(args.allow_noncanonical_split)
        if hasattr(args, "allow_noncanonical_split")
        else bool(getattr(args, "allow_synthetic_smoke_data", False))
    )
    args.allow_noncanonical_split = bool(
        cli.allow_noncanonical_split or configured_allow_noncanonical
    )
    split_metadata = get_kipa22_split_metadata(args)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    default_output = Path("/tmp/care_segmentation_kipa22/evaluation") / cli.split
    output_dir = (cli.output_dir or default_output).resolve()
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    if cli.dataset != "kipa22":
        raise NotImplementedError(f"Unsupported segmentation dataset: {cli.dataset}")
    transform = build_kipa22_transforms(args, split=cli.split)
    dataset = KiPA22Dataset(args, split=cli.split, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=cli.device == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = get_model(args, load_pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    spacing = _preprocessed_spacing(Path(args.data_dir))
    case_rows = []
    metrics = PolyMetrics(args, mode="fast")
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            if "label" not in batch:
                raise RuntimeError(
                    f"{cli.split} metrics require labels in the inference cache"
                )
            inputs = batch["image"].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = sliding_window_inference(
                    inputs=inputs,
                    roi_size=(args.roi_x, args.roi_y, args.roi_z),
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                    overlap=0.5,
                )
            predictions, targets, valid = prepare_metric_batch(
                logits, batch["label"], batch.get("valid_mask")
            )
            case_id = str(batch["case_id"][0])
            record = metrics.update(
                predictions, targets, valid, case_ids=[case_id]
            )[0]
            prediction = predictions[0].cpu().numpy().astype(np.uint8)
            valid_mask = valid[0].cpu().numpy().astype(bool)
            prediction[~valid_mask] = 0

            arrays = {
                "prediction": prediction,
                "valid_mask": valid_mask,
                "case_id": np.asarray(case_id),
                "split": np.asarray(cli.split),
                "space": np.asarray("nnUNet_preprocessed"),
                "axis_order": np.asarray("DHW"),
                "grid_shape_dhw": np.asarray(prediction.shape, dtype=np.int32),
                "allow_noncanonical_split": np.asarray(
                    split_metadata["allow_noncanonical_split"]
                ),
                "split_counts": np.asarray(
                    json.dumps(split_metadata["split_counts"], sort_keys=True)
                ),
                "canonical_split_counts": np.asarray(
                    split_metadata["canonical_split_counts"]
                ),
            }
            if spacing is not None:
                arrays["spacing_dhw"] = spacing
            if "properties" in batch:
                arrays["source_properties"] = np.asarray(str(batch["properties"][0]))
            atomic_write_npz(mask_dir / f"{case_id}.npz", **arrays)

            row = {"case_id": case_id, "split": cli.split}
            for class_id, name in LABELS.items():
                score = record["DSC_per_class"][name]
                row[f"class_{class_id}_{name}_DSC"] = (
                    "" if not np.isfinite(score) else float(score)
                )
            case_rows.append(row)
            print(f"Evaluated {index}/{len(dataset)}: {case_id}")

    if not case_rows:
        raise RuntimeError("Evaluation dataset is empty")
    computed = metrics.compute()
    metadata = checkpoint.get("metadata", {})
    aggregate = {
        "split": cli.split,
        "num_cases": len(case_rows),
        "checkpoint": str(checkpoint_path),
        "best_metric": _nullable(
            checkpoint.get("best_metric", metadata.get("best_metric", float("nan")))
        ),
        "best_metric_epoch": int(
            checkpoint.get("best_metric_epoch", metadata.get("best_metric_epoch", -1))
        ),
        "strict_checkpoint_load": True,
        "space": "nnUNet_preprocessed",
        "original_space_nifti_written": False,
        **split_metadata,
        "DSC": _nullable(computed["DSC"]),
        "DSC_per_class": {
            name: _nullable(score)
            for name, score in computed["DSC_per_class"].items()
        },
        "foreground_macro_DSC": _nullable(computed["foreground_macro_DSC"]),
        "focus_DSC": _nullable(computed["focus_DSC"]),
        "anatomy_group_DSC": _nullable(computed["anatomy_group_DSC"]),
        "tumor_group_DSC": _nullable(computed["tumor_group_DSC"]),
        "tumor_foreground_class_4_DSC": _nullable(
            computed["DSC_per_class"]["tumor"]
        ),
    }
    fieldnames = ["case_id", "split"] + [
        f"class_{class_id}_{name}_DSC" for class_id, name in LABELS.items()
    ]
    atomic_write_csv(output_dir / "per_case_per_class_dsc.csv", fieldnames, case_rows)
    atomic_write_json(output_dir / "aggregate_metrics.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
