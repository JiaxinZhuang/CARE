"""Evaluate one CARE fine-tuning checkpoint on an explicit NIfTI manifest."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import torch

TASK_DIR = Path(__file__).resolve().parent
CARE_ROOT = TASK_DIR.parents[2]
for import_root in (str(CARE_ROOT), str(TASK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

import train as training


FROZEN_CONFIG_FIELDS = (
    "labels", "aux_labels", "windows", "input_size", "resize_mode", "orientation",
    "voxel_ratio", "normalization", "arm", "rank", "lora_blocks", "last_n",
    "head_type", "pool_type", "feature_l2", "schedule_mode", "train_droppath",
)


def load_full_checkpoint(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a dictionary")
    if not isinstance(checkpoint.get("config"), dict):
        raise ValueError("Checkpoint is missing its training configuration")
    if not isinstance(checkpoint.get("parameters"), dict):
        raise ValueError("Checkpoint is missing fine-tuned parameters")
    return checkpoint


def checkpoint_config(requested_config: dict, checkpoint: dict, workers: int | None = None) -> dict:
    saved = checkpoint["config"]
    missing = "<missing>"
    differences = {}
    for key in FROZEN_CONFIG_FIELDS:
        requested = requested_config.get(key, missing)
        trained = saved.get(key, missing)
        if requested != trained:
            differences[key] = (requested, trained)
    if differences:
        raise ValueError(f"Checkpoint architecture/preprocessing configuration differs: {differences}")
    config = dict(saved)
    if workers is not None:
        config["workers"] = workers
    return config


def evaluate_checkpoint(args, requested_config, device):
    checkpoint = model = dataset = loader = None
    try:
        checkpoint = load_full_checkpoint(args.checkpoint)
        config = checkpoint_config(requested_config, checkpoint, args.workers)
        supplied_sha256 = training.digest(args.pretrained)
        trained_sha256 = checkpoint.get("pretrained", {}).get("sha256")
        if trained_sha256 != supplied_sha256:
            raise ValueError(
                "The supplied CARE base checkpoint differs from the one used for fine-tuning")
        model = training.build_model(config, args.pretrained, device)
        model.optimizer()
        model.restore(checkpoint["parameters"])
        model.requires_grad_(False)
        dataset = training.InputDataset(
            args.manifest, config, args.split_name, augment=False,
            require_labels=not args.no_metrics)
        loader = training.make_loader(dataset, config, device=device)
        result = training.validate(
            model, loader, device, config, args.output_dir, args.split_name,
            compute_basic_metrics=not args.no_metrics, view=args.view)
        summary = {
            "split": args.split_name,
            "view": args.view,
            "checkpoint": Path(args.checkpoint).name,
            "checkpoint_sha256": training.digest(args.checkpoint),
            "pretrained": {"filename": Path(args.pretrained).name, "sha256": supplied_sha256},
            "metrics": result,
            "trained_during_evaluation": False,
        }
        training.write_json(Path(args.output_dir) / "results.json", summary)
        return summary
    finally:
        del loader, dataset, model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TASK_DIR / "config.example.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True,
                        help="CARE base checkpoint used by the fine-tuned checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--view", choices=("base", "tta2"), default="base")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Permit a test manifest without label columns")
    return parser


def main():
    args = get_args_parser().parse_args()
    requested_config, _ = training.load_config(args.config)
    device = training.resolve_device(args.device)
    summary = evaluate_checkpoint(args, requested_config, device)
    metrics = summary["metrics"]
    print({
        "split": summary["split"],
        "view": summary["view"],
        "volume_predictions": metrics["volume_predictions"],
        "group_predictions": metrics["group_predictions"],
        "macro_auc": metrics["macro_auc"],
        "output_dir": str(Path(args.output_dir).resolve()),
    }, flush=True)


if __name__ == "__main__":
    main()
