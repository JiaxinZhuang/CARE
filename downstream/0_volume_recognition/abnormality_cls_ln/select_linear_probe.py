"""Select one cached-feature FC run using validation metrics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from linear_probe_common import atomic_write_json, ensure_external_artifact_path


VALIDATION_METRICS = {
    "best_holdout_macro_auc",
    "best_val_macro_auc_table11",
    "best_selection_auc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", default="best_val_macro_auc_table11")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metric not in VALIDATION_METRICS:
        raise ValueError(
            f"Selection metric must be validation-only; choose from {sorted(VALIDATION_METRICS)}"
        )
    output_path = ensure_external_artifact_path(Path(args.output))
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite selection: {output_path}")

    candidates = []
    feature_hashes = set()
    train_manifest_hashes = set()
    holdout_manifest_hashes = set()
    selection_modes = set()
    selection_classname_orders = set()
    for requested_path in args.candidate:
        path = Path(requested_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "accepted":
            raise ValueError(f"Candidate is not accepted: {path}")
        if args.metric not in payload or payload[args.metric] is None:
            raise KeyError(f"Candidate has no usable {args.metric}: {path}")
        checkpoint = Path(payload["checkpoint"]).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Candidate checkpoint is missing: {checkpoint}")
        feature_hashes.add(payload.get("feature_sha256"))
        train_manifest_hashes.add(payload.get("train_manifest_sha256"))
        holdout_manifest_hashes.add(payload.get("holdout_manifest_sha256"))
        selection_mode = payload.get("selection_classes")
        selection_classnames = payload.get("selection_classnames")
        if not isinstance(selection_mode, str) or not selection_mode:
            raise ValueError(f"Candidate has no selection_classes: {path}")
        if not isinstance(selection_classnames, list) or not selection_classnames:
            raise ValueError(f"Candidate has no ordered selection_classnames: {path}")
        selection_modes.add(selection_mode)
        selection_classname_orders.add(tuple(map(str, selection_classnames)))
        candidates.append(
            {
                "summary": str(path),
                "checkpoint": str(checkpoint),
                "metric": float(payload[args.metric]),
                "learning_rate": payload.get("learning_rate"),
                "batch_size": payload.get("batch_size"),
                "max_epochs": payload.get("max_epochs"),
                "scheduler": payload.get("scheduler"),
                "best_epoch": payload.get("best_epoch"),
            }
        )

    for name, values in (
        ("feature artifacts", feature_hashes),
        ("train manifests", train_manifest_hashes),
        ("holdout manifests", holdout_manifest_hashes),
    ):
        if len(values) != 1 or None in values:
            raise ValueError(f"Candidates use missing or inconsistent {name}: {values}")
    if len(selection_modes) != 1:
        raise ValueError(
            f"Candidates use inconsistent selection_classes: {selection_modes}"
        )
    if len(selection_classname_orders) != 1:
        raise ValueError(
            "Candidates use inconsistent ordered selection_classnames: "
            f"{selection_classname_orders}"
        )

    candidates.sort(key=lambda item: item["metric"], reverse=True)
    summary = {
        "status": "accepted",
        "selection_source": "validation only; test labels and test metrics were not read",
        "selection_metric": args.metric,
        "selected": candidates[0],
        "candidates": candidates,
        "acceptance": {
            "validation_only_metric": True,
            "consistent_feature_artifact": True,
            "consistent_train_manifest": True,
            "consistent_holdout_manifest": True,
            "consistent_selection_classes": True,
            "consistent_ordered_selection_classnames": True,
            "all_checkpoints_exist": True,
        },
    }
    atomic_write_json(output_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
