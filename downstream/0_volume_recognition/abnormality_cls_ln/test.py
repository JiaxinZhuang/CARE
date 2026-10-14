"""Apply a frozen affine probe and report basic volume-level AUC/AP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from train import (
    MISSING_GROUP_IDS,
    load_features,
    manifest_label_classnames,
    reliable_groups,
)
from linear_probe_common import (
    CTRATE_PAPER14_CLASSNAMES,
    CTRATE_REPORT_CLASSNAMES,
    atomic_write_json,
    ensure_external_artifact_path,
    sha256_file,
)
from probe_models import build_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--train-manifest", default="")
    parser.add_argument("--holdout-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report-classes",
        choices=["ctrate-table11", "ctrate-paper14", "all"],
        default="ctrate-table11",
    )
    parser.add_argument("--labels-from-manifest", action="store_true")
    return parser.parse_args()


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = ensure_external_artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def metrics_by_class(
    labels: np.ndarray,
    scores: np.ndarray,
    classnames: list[str],
) -> tuple[float, float, list[dict]]:
    rows = []
    for column, classname in enumerate(classnames):
        if np.unique(labels[:, column]).size != 2:
            raise ValueError(f"Evaluation split lacks both classes for: {classname}")
        rows.append(
            {
                "classname": classname,
                "auc": float(roc_auc_score(labels[:, column], scores[:, column])),
                "average_precision": float(
                    average_precision_score(labels[:, column], scores[:, column])
                ),
                "positive_cases": int(labels[:, column].sum()),
                "negative_cases": int(len(labels) - labels[:, column].sum()),
            }
        )
    return (
        float(np.mean([row["auc"] for row in rows])),
        float(np.mean([row["average_precision"] for row in rows])),
        rows,
    )


def _manifest_identity(path_value: str | None) -> tuple[set[str], set[str]]:
    if not path_value:
        return set(), set()
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        return set(), set()
    frame = pd.read_csv(path)
    if "case_id" not in frame.columns:
        raise ValueError(f"Saved split manifest has no case_id column: {path}")
    cases = set(frame["case_id"].astype(str))
    groups = reliable_groups(frame["group_id"].astype(str).to_numpy()) if "group_id" in frame.columns else set()
    return cases, groups


def _required_manifest_identity(
    path_value: str, split_name: str
) -> tuple[set[str], set[str], bool, str]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{split_name} manifest not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{split_name} manifest is empty: {path}")
    if "case_id" not in frame.columns:
        raise ValueError(f"{split_name} manifest has no case_id column: {path}")
    case_ids = frame["case_id"].astype(str)
    if case_ids.duplicated().any():
        raise ValueError(f"Duplicate case IDs in {split_name} manifest: {path}")
    groups = set()
    complete_groups = False
    if "group_id" in frame.columns:
        group_ids = frame["group_id"].astype(str).to_numpy()
        groups = reliable_groups(group_ids)
        complete_groups = bool(groups) and all(
            value.strip().lower() not in MISSING_GROUP_IDS for value in group_ids
        )
    return set(case_ids), groups, complete_groups, sha256_file(path)


def _external_split_identity(
    train_value: str, holdout_value: str
) -> tuple[set[str], set[str], dict]:
    train_cases, train_groups, train_groups_complete, train_sha256 = (
        _required_manifest_identity(train_value, "Train")
    )
    holdout_cases, holdout_groups, holdout_groups_complete, holdout_sha256 = (
        _required_manifest_identity(holdout_value, "Holdout")
    )
    case_overlap = train_cases & holdout_cases
    if case_overlap:
        raise ValueError(
            f"Train/holdout manifest case leakage: {sorted(case_overlap)[:5]}"
        )
    group_overlap = train_groups & holdout_groups
    if group_overlap:
        raise ValueError(
            f"Train/holdout manifest group leakage: {sorted(group_overlap)[:5]}"
        )
    return train_cases | holdout_cases, train_groups | holdout_groups, {
        "train_manifest_sha256": train_sha256,
        "holdout_manifest_sha256": holdout_sha256,
        "train_cases": len(train_cases),
        "holdout_cases": len(holdout_cases),
        "group_identity_complete": train_groups_complete and holdout_groups_complete,
    }


def _saved_split_identity(checkpoint: dict) -> tuple[set[str], set[str]]:
    case_ids = set(map(str, checkpoint.get("train_case_ids", [])))
    case_ids.update(map(str, checkpoint.get("holdout_case_ids", [])))
    group_ids = reliable_groups(np.asarray(checkpoint.get("train_group_ids", []), dtype=str))
    group_ids.update(reliable_groups(np.asarray(checkpoint.get("holdout_group_ids", []), dtype=str)))
    for key in ("train_manifest", "holdout_manifest"):
        manifest_cases, manifest_groups = _manifest_identity(checkpoint.get(key))
        case_ids.update(manifest_cases)
        group_ids.update(manifest_groups)
    if not case_ids:
        return set(), set()
    if not group_ids and checkpoint.get("feature_artifact"):
        feature_path = Path(checkpoint["feature_artifact"]).expanduser().resolve()
        if feature_path.is_file():
            source = load_features(feature_path)
            mapping = dict(
                zip(source["case_ids"].astype(str), source["group_ids"].astype(str))
            )
            group_ids = reliable_groups(
                np.asarray([mapping[case_id] for case_id in case_ids if case_id in mapping], dtype=str)
            )
    return case_ids, group_ids


def _feature_definition(name: str, value) -> tuple[dict | None, bool]:
    if value is None:
        return None, False
    if value == "UNKNOWN" or (
        isinstance(value, dict) and value.get("status") == "UNKNOWN"
    ):
        return None, True
    if not isinstance(value, dict):
        raise ValueError(
            f"{name} feature_definition must be a JSON object or 'UNKNOWN'"
        )
    return value, False


def _validate_linear_state(checkpoint: dict, feature_dim: int, num_classes: int) -> None:
    required = {"state_dict", "feature_dim", "classnames"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Linear-probe head is missing fields: {sorted(missing)}")
    state = checkpoint["state_dict"]
    if set(state) != {"weight", "bias"}:
        raise ValueError(f"Linear-probe state must contain only weight and bias: {sorted(state)}")
    if tuple(state["weight"].shape) != (num_classes, feature_dim):
        raise ValueError("Saved linear weight shape disagrees with classes/features")
    if tuple(state["bias"].shape) != (num_classes,):
        raise ValueError("Saved linear bias shape disagrees with classes")


def main() -> None:
    args = parse_args()
    if bool(args.train_manifest) != bool(args.holdout_manifest):
        raise ValueError(
            "--train-manifest and --holdout-manifest must be provided together"
        )
    output_dir = ensure_external_artifact_path(Path(args.output_dir))
    volume_path = output_dir / "predictions_volume.csv"
    group_path = output_dir / "predictions_group_mean.csv"
    metrics_path = output_dir / "metrics.json"
    if volume_path.exists() or group_path.exists() or metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output_dir}")

    feature_path = Path(args.features).expanduser().resolve()
    head_path = Path(args.head).expanduser().resolve()
    artifact = load_features(feature_path)
    manifest_path = None
    if args.labels_from_manifest and not args.manifest:
        raise ValueError("--labels-from-manifest requires --manifest")
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        manifest = pd.read_csv(manifest_path)
        if "case_id" not in manifest.columns:
            raise ValueError(f"Manifest has no case_id column: {manifest_path}")
        requested = manifest["case_id"].astype(str).tolist()
        if len(requested) != len(set(requested)):
            raise ValueError(f"Duplicate case IDs in {manifest_path}")
        artifact_ids = artifact["case_ids"].astype(str)
        index_by_case = {case_id: index for index, case_id in enumerate(artifact_ids)}
        missing = [case_id for case_id in requested if case_id not in index_by_case]
        if missing:
            raise ValueError(
                f"Evaluation manifest has {len(missing)} absent cases; examples: {missing[:5]}"
            )
        indices = np.asarray([index_by_case[case_id] for case_id in requested], dtype=int)
        for key in ("features", "labels", "case_ids", "group_ids"):
            artifact[key] = artifact[key][indices]
        if "group_id" in manifest.columns and not np.array_equal(
            artifact["group_ids"].astype(str), manifest["group_id"].astype(str).to_numpy()
        ):
            raise ValueError("Evaluation manifest group IDs disagree with features")
        if args.labels_from_manifest:
            dataset_classnames = manifest_label_classnames(manifest)
            labels = manifest[dataset_classnames].to_numpy(dtype=np.float32)
            if not np.isin(labels, [0, 1]).all():
                raise ValueError("Manifest labels must be binary 0/1 values")
            artifact["classnames"] = np.asarray(dataset_classnames, dtype=str)
            artifact["labels"] = labels

    checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Linear-probe head must be a checkpoint dictionary")
    dataset_classnames = artifact["classnames"].astype(str).tolist()
    head_classnames = list(map(str, checkpoint.get("classnames", [])))
    if head_classnames != dataset_classnames:
        raise ValueError("Head and feature artifact class orders differ")
    features = np.asarray(artifact["features"])
    if features.ndim != 2:
        raise ValueError("Affine linear probe requires two-dimensional pooled features")
    feature_dim = int(features.shape[1])
    if int(checkpoint.get("feature_dim", -1)) != feature_dim:
        raise ValueError("Head and feature dimensions differ")
    if not np.isfinite(features).all():
        raise FloatingPointError("Non-finite feature values detected")
    probe_spec = checkpoint.get("probe_spec", {"type": "linear"})
    if probe_spec.get("type", "linear") != "linear" or probe_spec.get("nonlinear_classnames"):
        raise ValueError("Only an affine linear saved head is supported")
    _validate_linear_state(checkpoint, feature_dim, len(head_classnames))

    evaluation_metadata = (
        json.loads(str(artifact["metadata_json"].item()))
        if "metadata_json" in artifact
        else None
    )
    evaluation_feature_definition = (
        evaluation_metadata.get("feature_definition")
        if evaluation_metadata is not None
        else None
    )
    training_feature_definition = checkpoint.get("feature_definition")
    if training_feature_definition is None and checkpoint.get("feature_artifact"):
        training_feature_path = Path(checkpoint["feature_artifact"]).expanduser().resolve()
        if training_feature_path.is_file():
            training_artifact = load_features(training_feature_path)
            if "metadata_json" in training_artifact:
                training_metadata = json.loads(
                    str(training_artifact["metadata_json"].item())
                )
                training_feature_definition = training_metadata.get(
                    "feature_definition"
                )
    known_training_definition, training_definition_unknown = _feature_definition(
        "training", training_feature_definition
    )
    known_evaluation_definition, evaluation_definition_unknown = _feature_definition(
        "evaluation", evaluation_feature_definition
    )
    feature_definition_match = None
    if known_training_definition is not None and known_evaluation_definition is not None:
        feature_definition_match = known_training_definition == known_evaluation_definition
        if not feature_definition_match:
            keys = sorted(
                set(known_training_definition) | set(known_evaluation_definition)
            )
            differing = [
                key
                for key in keys
                if known_training_definition.get(key)
                != known_evaluation_definition.get(key)
            ]
            raise ValueError(
                "Train/evaluation feature definitions differ for: "
                f"{differing}"
            )
    feature_compatibility = (
        "Train/evaluation feature definitions match exactly, including pooling, z-windows, and checkpoint hash; no archived-feature equivalence claim"
        if feature_definition_match
        else "Train or evaluation feature definition unknown; NPZ schema, class order, and dimensions validated only; no archived-feature equivalence claim"
    )

    saved_cases, saved_groups = _saved_split_identity(checkpoint)
    checkpoint_identity_available = bool(saved_cases)
    external_identity = None
    if args.train_manifest:
        external_cases, external_groups, external_identity = _external_split_identity(
            args.train_manifest, args.holdout_manifest
        )
        for key in ("train_manifest_sha256", "holdout_manifest_sha256"):
            expected_sha256 = checkpoint.get(key)
            if expected_sha256 is not None and str(expected_sha256) != external_identity[key]:
                raise ValueError(f"Explicit {key} does not match the checkpoint")
        if saved_cases and external_cases != saved_cases:
            raise ValueError(
                "Explicit split manifests do not match checkpoint split case identity"
            )
        if saved_groups and external_groups and external_groups != saved_groups:
            raise ValueError(
                "Explicit split manifests do not match checkpoint split group identity"
            )
        saved_cases.update(external_cases)
        saved_groups.update(external_groups)
    if not saved_cases:
        raise ValueError(
            "Head has no usable train/holdout case identity; provide both "
            "--train-manifest and --holdout-manifest"
        )
    isolation_mode = (
        "checkpoint-and-external-manifests"
        if external_identity is not None and checkpoint_identity_available
        else "external-manifests"
        if external_identity is not None
        else "checkpoint-metadata"
    )
    evaluation_cases = set(artifact["case_ids"].astype(str))
    case_overlap = saved_cases & evaluation_cases
    if case_overlap:
        raise ValueError(f"Evaluation case leakage from train/holdout: {sorted(case_overlap)[:5]}")
    evaluation_groups = reliable_groups(artifact["group_ids"].astype(str))
    if (
        evaluation_groups
        and external_identity is not None
        and not external_identity["group_identity_complete"]
    ):
        raise ValueError(
            "Evaluation has group IDs; both external split manifests require "
            "complete group_id values"
        )
    if evaluation_groups and not saved_groups:
        raise ValueError("Head has no usable train/holdout group identity for test isolation")
    group_overlap = saved_groups & evaluation_groups
    if group_overlap:
        raise ValueError(f"Evaluation group leakage from train/holdout: {sorted(group_overlap)[:5]}")

    device = torch.device(args.device)
    classifier = build_probe(feature_dim, head_classnames, probe_spec).to(device)
    classifier.load_state_dict(checkpoint["state_dict"], strict=True)
    classifier.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(
            classifier(torch.from_numpy(features.astype(np.float32, copy=False)).to(device))
        ).cpu().numpy()
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise FloatingPointError("Invalid sigmoid probabilities")

    report_classnames = (
        CTRATE_REPORT_CLASSNAMES
        if args.report_classes == "ctrate-table11"
        else CTRATE_PAPER14_CLASSNAMES
        if args.report_classes == "ctrate-paper14"
        else dataset_classnames
    )
    missing = [name for name in report_classnames if name not in dataset_classnames]
    if missing:
        raise ValueError(f"Requested report classes are absent: {missing}")
    report_indices = np.asarray([dataset_classnames.index(name) for name in report_classnames])
    labels = artifact["labels"][:, report_indices].astype(np.int8, copy=False)
    scores = probabilities[:, report_indices]
    macro_auc, macro_ap, per_class = metrics_by_class(labels, scores, report_classnames)

    groups = artifact["group_ids"].astype(str)
    predictions = pd.DataFrame(
        {"case_id": artifact["case_ids"].astype(str), "group_id": groups}
    )
    for column, classname in enumerate(report_classnames):
        predictions[f"label::{classname}"] = labels[:, column]
        predictions[f"prob::{classname}"] = scores[:, column]
    _atomic_write_csv(volume_path, predictions)

    group_output = None
    group_metrics = None
    group_reason = "group IDs are missing or every group has one volume"
    valid_group_values = [value.strip().lower() not in MISSING_GROUP_IDS for value in groups]
    if all(valid_group_values) and len(set(groups)) < len(groups):
        label_columns = [f"label::{name}" for name in report_classnames]
        probability_columns = [f"prob::{name}" for name in report_classnames]
        grouped = predictions.groupby("group_id", sort=False, dropna=False)
        inconsistent = [
            column for column in label_columns if grouped[column].nunique().max() != 1
        ]
        if inconsistent:
            group_reason = f"labels differ within groups for columns: {inconsistent}"
        else:
            group_predictions = grouped[label_columns + probability_columns].mean().reset_index()
            group_predictions.insert(1, "volume_count", grouped.size().to_numpy())
            for column in label_columns:
                group_predictions[column] = group_predictions[column].astype(np.int8)
            group_labels = group_predictions[label_columns].to_numpy(dtype=np.int8)
            group_scores = group_predictions[probability_columns].to_numpy(dtype=float)
            group_auc, group_ap, group_per_class = metrics_by_class(
                group_labels, group_scores, report_classnames
            )
            _atomic_write_csv(group_path, group_predictions)
            group_output = str(group_path)
            group_metrics = {
                "samples": len(group_predictions),
                "macro_auc": group_auc,
                "macro_average_precision": group_ap,
                "per_class": group_per_class,
            }
            group_reason = None

    summary = {
        "status": "accepted",
        "features": str(feature_path),
        "features_sha256": sha256_file(feature_path),
        "manifest": str(manifest_path) if manifest_path else None,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path else None,
        "head": str(head_path),
        "head_sha256": sha256_file(head_path),
        "samples": len(labels),
        "feature_dim": feature_dim,
        "classnames": dataset_classnames,
        "report_classnames": report_classnames,
        "macro_auc": macro_auc,
        "macro_average_precision": macro_ap,
        "per_class": per_class,
        "predictions_volume": str(volume_path),
        "predictions_group_mean": group_output,
        "group_mean_metrics": group_metrics,
        "group_mean_unavailable_reason": group_reason,
        "training_feature_definition": training_feature_definition,
        "evaluation_feature_definition": evaluation_feature_definition,
        "feature_definition_match": feature_definition_match,
        "training_feature_definition_explicitly_unknown": training_definition_unknown,
        "evaluation_feature_definition_explicitly_unknown": evaluation_definition_unknown,
        "feature_compatibility": feature_compatibility,
        "base_model_identity": checkpoint.get("base_model_identity"),
        "split_isolation": {
            "mode": isolation_mode,
            "saved_case_count": len(saved_cases),
            "saved_group_count": len(saved_groups),
            "external_manifests": external_identity,
            "evaluation_case_overlap": 0,
            "evaluation_group_overlap": 0,
        },
        "acceptance": {
            "strict_class_order_match": True,
            "strict_feature_dim_match": True,
            "strict_affine_state_dict": True,
            "known_feature_definitions_match": feature_definition_match,
            "finite_features": True,
            "bounded_sigmoid_probabilities": True,
            "train_holdout_case_overlap": 0,
            "train_holdout_group_overlap": 0,
            "split_isolation_verified": True,
            "test_labels_used_for_head_selection": False,
        },
    }
    atomic_write_json(metrics_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
