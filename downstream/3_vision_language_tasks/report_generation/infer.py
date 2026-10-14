"""Run CARE report generation from a released split definition."""

import argparse
import csv
import hashlib
import json
from argparse import Namespace
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import torch

from dataset.flare_dataset import FLARECapDataset
from models.ours import (
    PROJ_TOKENS,
    SYSTEM_PROMPT,
    USER_PROMPT,
    create_ours,
    load_report_config,
)


MAX_NEW_TOKENS = 512
FLARE_TASK_NAME = "flare25_task05"


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split-dir")
    parser.add_argument("--json-path", nargs="+")
    parser.add_argument("--data-root", nargs="+")
    return parser


def _parse_args(argv=None):
    parser = _parser()
    args, unsupported = parser.parse_known_args(argv)
    if unsupported:
        parser.error(
            "unsupported arguments: "
            + " ".join(unsupported)
            + "; TTA, classifier flags, and baseline-model options are not supported"
        )
    missing = [
        name
        for name in ("split_dir", "json_path", "data_root")
        if not getattr(args, name)
    ]
    if missing:
        parser.error(
            "FLARE25 requires "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )
    expected_jsons = 2 if args.split == "validation" else 1
    if len(args.json_path) != expected_jsons:
        parser.error(
            f"FLARE25 {args.split} requires {expected_jsons} --json-path value(s)"
        )
    if len(args.data_root) != 2:
        parser.error(
            "FLARE25 requires two --data-root values ordered as CT-AMOS CT-RATE"
        )
    return args


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_provenance(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _resolve_device(name):
    resolved = name or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(resolved)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {resolved}")
    return device


def _flare_task_provenance(checkpoint, config):
    if getattr(config, "task_name", None) != FLARE_TASK_NAME:
        raise ValueError(
            "FLARE25 inference requires a newly trained CARE ViT-H + Qwen3 merged "
            "checkpoint with config.task_name='flare25_task05'"
        )
    path = checkpoint / "task_config.json"
    provenance = _file_provenance(path)
    with path.open(encoding="utf-8") as stream:
        task_config = json.load(stream)
    expected = {
        "task": "FLARE25 Task05 report generation",
        "release_split": "RELEASE_SPLIT_V1",
        "selected_by": "eval_loss",
    }
    mismatches = {
        key: (task_config.get(key), value)
        for key, value in expected.items()
        if task_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported FLARE25 task_config.json: {mismatches}")
    return provenance


def _build_dataset(args, tokenizer, input_size, proj_out_num):
    dataset_args = Namespace(
        split_dir=args.split_dir,
        json_path=args.json_path,
        data_root=args.data_root,
        proj_out_num=proj_out_num,
        prompt="simple",
        data_img_size=input_size,
        max_length=1024,
    )
    return FLARECapDataset(dataset_args, tokenizer, mode=args.split)


def _run_config_path(output):
    return output.with_suffix(".run.json")


def main(argv=None):
    args = _parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if output.suffix.lower() != ".csv":
        raise ValueError(f"--output must be a CSV path, got: {output}")
    checkpoint_config = _file_provenance(checkpoint / "config.json")
    checkpoint_index = _file_provenance(checkpoint / "model.safetensors.index.json")
    config = load_report_config(checkpoint)
    task_config = _flare_task_provenance(checkpoint, config)
    input_size = tuple(config.input_size)
    device = _resolve_device(args.device)
    model, tokenizer = create_ours(checkpoint, device, config=config)
    dataset = _build_dataset(args, tokenizer, input_size, config.proj_out_num)

    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample_id", "generated_report", "reference_report"),
        )
        writer.writeheader()
        for index in range(len(dataset)):
            item = dataset[index]
            if not isinstance(item, Mapping):
                raise TypeError(f"Dataset item {index} must be a mapping")
            missing = {"sample_id", "image", "report"} - set(item)
            if missing:
                raise ValueError(f"Dataset item {index} is missing fields: {sorted(missing)}")

            image = item["image"]
            if not isinstance(image, torch.Tensor):
                raise TypeError(f"Dataset item {index} image must be a torch.Tensor")
            if image.ndim == 4:
                image = image.unsqueeze(0)
            if image.ndim != 5 or image.shape[0] != 1:
                raise ValueError(
                    f"Dataset item {index} image must be [C,D,H,W] or [1,C,D,H,W], "
                    f"got {tuple(image.shape)}"
                )

            generated = model.generate(image, max_new_tokens=MAX_NEW_TOKENS)
            writer.writerow(
                {
                    "sample_id": item["sample_id"],
                    "generated_report": generated,
                    "reference_report": str(item["report"]),
                }
            )
            stream.flush()
            rows_written += 1

    registry_split = "val" if args.split == "validation" else "test"
    registry = Path(args.split_dir).expanduser().resolve() / f"{registry_split}.csv"
    data_selection = {
        "dataset": FLARE_TASK_NAME,
        "split": args.split,
        "release_registry": _file_provenance(registry),
        "annotation_jsons": [_file_provenance(Path(path)) for path in args.json_path],
        "ct_roots": [str(Path(path).expanduser().resolve()) for path in args.data_root],
    }

    run_config = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(checkpoint),
            "config": checkpoint_config,
            "index": checkpoint_index,
            "strict_full_state_validation": True,
            "external_visual_resume_used": False,
        },
        "model": {
            "architecture": "VLMQwen3ForCausalLM",
            "input_size": list(input_size),
            "feature_mode": "tokens_cls",
            "projected_image_tokens": PROJ_TOKENS,
        },
        "prompt": {
            "system": getattr(config, "system_prompt", SYSTEM_PROMPT),
            "content": getattr(config, "user_prompt", USER_PROMPT),
        },
        "generation": {
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "device": str(device),
        "output_csv": str(output),
        "rows_written": rows_written,
    }
    run_config["checkpoint"]["task_config"] = task_config
    run_config.update(data_selection)
    run_path = _run_config_path(output)
    with run_path.open("w", encoding="utf-8") as stream:
        json.dump(run_config, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
