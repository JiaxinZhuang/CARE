"""Fit only an affine FC layer on frozen, cached CT-RATE features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from linear_probe_common import (
    CTRATE_REPORT_CLASSNAMES,
    atomic_torch_save,
    atomic_write_json,
    ensure_external_artifact_path,
    set_reproducible_seed,
    sha256_file,
)
from probe_losses import build_probe_loss, make_loss_spec
from probe_models import build_probe, make_probe_spec


MANIFEST_META_COLUMNS = frozenset({"case_id", "group_id", "split", "image_path"})
MISSING_GROUP_IDS = frozenset({"", "nan", "none", "null"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--holdout-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--selection-classes",
        choices=["all", "ctrate-table11"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", choices=["ctrate"], default="ctrate")
    parser.add_argument("--labels-from-manifest", action="store_true")
    parser.add_argument(
        "--allow-undefined-holdout-label",
        action="append",
        default=[],
    )
    return parser.parse_args()


def manifest_label_classnames(manifest: pd.DataFrame) -> list[str]:
    names = [name for name in manifest.columns if name not in MANIFEST_META_COLUMNS]
    if not names:
        raise ValueError("Manifest has no label columns")
    return names


def macro_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    allow_undefined_indices: frozenset[int] = frozenset(),
) -> tuple[float, list[float | None]]:
    aucs = []
    for column in range(labels.shape[1]):
        if np.unique(labels[:, column]).size != 2:
            if column not in allow_undefined_indices:
                raise ValueError(
                    f"Label column {column} lacks a positive or negative example"
                )
            aucs.append(None)
            continue
        aucs.append(float(roc_auc_score(labels[:, column], scores[:, column])))
    valid = [value for value in aucs if value is not None]
    if not valid:
        raise ValueError("No holdout labels have both positive and negative examples")
    return float(np.mean(valid)), aucs


def load_features(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as artifact:
        required = {"features", "labels", "case_ids", "group_ids", "classnames"}
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"Feature artifact is missing arrays: {sorted(missing)}")
        result = {name: artifact[name] for name in required}
        if "metadata_json" in artifact.files:
            metadata_json = artifact["metadata_json"]
            if metadata_json.shape != ():
                raise ValueError("metadata_json must be a scalar JSON string")
            try:
                metadata = json.loads(str(metadata_json.item()))
            except json.JSONDecodeError as error:
                raise ValueError("metadata_json is not valid JSON") from error
            if not isinstance(metadata, dict):
                raise ValueError("metadata_json must encode a JSON object")
            result["metadata_json"] = metadata_json
    features = result["features"]
    labels = result["labels"]
    case_ids = result["case_ids"].astype(str)
    group_ids = result["group_ids"].astype(str)
    classnames = result["classnames"].astype(str)
    if features.ndim != 2 or labels.ndim != 2:
        raise ValueError("features and labels must both be two-dimensional")
    if len(features) != len(labels) or len(case_ids) != len(labels) or len(group_ids) != len(labels):
        raise ValueError("Feature artifact arrays have inconsistent sample counts")
    if labels.shape[1] != len(classnames):
        raise ValueError("Label width does not match classnames")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Feature artifact has duplicate case IDs")
    if len(classnames) != len(set(classnames)):
        raise ValueError("Feature artifact has duplicate class names")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Feature labels must be binary 0/1 values")
    return result


def select_manifest(
    artifact: dict,
    manifest_path: Path,
    split_name: str,
    labels_from_manifest: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = pd.read_csv(manifest_path)
    if "case_id" not in manifest.columns:
        raise ValueError(f"{manifest_path} has no case_id column")
    requested = manifest["case_id"].astype(str).tolist()
    if len(requested) != len(set(requested)):
        raise ValueError(f"Duplicate case IDs in {manifest_path}")
    artifact_ids = artifact["case_ids"].astype(str)
    index_by_case = {case_id: index for index, case_id in enumerate(artifact_ids)}
    missing = [case_id for case_id in requested if case_id not in index_by_case]
    if missing:
        raise ValueError(
            f"{split_name} manifest has {len(missing)} cases absent from features; "
            f"examples: {missing[:5]}"
        )
    indices = np.asarray([index_by_case[case_id] for case_id in requested], dtype=int)
    features = artifact["features"][indices].astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise FloatingPointError(f"Non-finite {split_name} features")
    if "group_id" in manifest.columns:
        expected_groups = artifact["group_ids"][indices].astype(str)
        manifest_groups = manifest["group_id"].astype(str).to_numpy()
        if not np.array_equal(expected_groups, manifest_groups):
            raise ValueError(f"{split_name} manifest group IDs disagree with features")
    if labels_from_manifest:
        classnames = manifest_label_classnames(manifest)
        labels = manifest[classnames].to_numpy(dtype=np.float32)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{split_name} manifest labels must be binary 0/1 values")
        return features, labels, artifact_ids[indices]
    labels = artifact["labels"][indices].astype(np.float32, copy=False)
    classnames = artifact["classnames"].astype(str).tolist()
    if all(name in manifest.columns for name in classnames):
        if not np.array_equal(labels, manifest[classnames].to_numpy(dtype=np.float32)):
            raise ValueError(f"{split_name} manifest labels disagree with feature labels")
    return features, labels, artifact_ids[indices]


def groups_for_cases(artifact: dict, case_ids: np.ndarray) -> np.ndarray:
    index = {
        case_id: group_id
        for case_id, group_id in zip(
            artifact["case_ids"].astype(str), artifact["group_ids"].astype(str)
        )
    }
    return np.asarray([index[str(case_id)] for case_id in case_ids], dtype=str)


def reliable_groups(group_ids: np.ndarray) -> set[str]:
    return {
        value
        for value in group_ids.astype(str)
        if value.strip().lower() not in MISSING_GROUP_IDS
    }


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    features: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    loss_function: nn.Module,
    device: torch.device,
    allow_undefined_indices: frozenset[int] = frozenset(),
) -> tuple[float, float, list[float | None]]:
    feature_tensor = features.to(device) if torch.is_tensor(features) else torch.from_numpy(features).to(device)
    label_tensor = labels.to(device) if torch.is_tensor(labels) else torch.from_numpy(labels).to(device)
    label_array = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    logits = model(feature_tensor)
    loss = float(loss_function(logits, label_tensor))
    mean_auc, per_class_auc = macro_auc(
        label_array,
        torch.sigmoid(logits).cpu().numpy(),
        allow_undefined_indices,
    )
    return loss, mean_auc, per_class_auc


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if args.early_stopping_min_delta < 0 or args.min_learning_rate < 0:
        raise ValueError("early-stop delta and minimum learning rate must be non-negative")
    if args.scheduler == "cosine" and args.min_learning_rate >= args.learning_rate:
        raise ValueError("cosine minimum learning rate must be below initial learning rate")
    output_dir = ensure_external_artifact_path(Path(args.output_dir))
    checkpoint_path = output_dir / "best_fc.pt"
    summary_path = output_dir / "summary.json"
    if checkpoint_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite linear probe: {output_dir}")

    set_reproducible_seed(args.seed)
    feature_path = Path(args.features).expanduser().resolve()
    train_manifest = Path(args.train_manifest).expanduser().resolve()
    holdout_manifest = Path(args.holdout_manifest).expanduser().resolve()
    artifact = load_features(feature_path)
    train_x, train_y, train_ids = select_manifest(
        artifact, train_manifest, "train", args.labels_from_manifest
    )
    holdout_x, holdout_y, holdout_ids = select_manifest(
        artifact, holdout_manifest, "holdout", args.labels_from_manifest
    )
    case_overlap = set(train_ids.astype(str)) & set(holdout_ids.astype(str))
    if case_overlap:
        raise ValueError(f"Train/holdout case leakage: {sorted(case_overlap)[:5]}")
    train_groups = groups_for_cases(artifact, train_ids)
    holdout_groups = groups_for_cases(artifact, holdout_ids)
    group_overlap = reliable_groups(train_groups) & reliable_groups(holdout_groups)
    if group_overlap:
        raise ValueError(f"Train/holdout group leakage: {sorted(group_overlap)[:5]}")

    if args.labels_from_manifest:
        classnames = manifest_label_classnames(pd.read_csv(train_manifest))
        if manifest_label_classnames(pd.read_csv(holdout_manifest)) != classnames:
            raise ValueError("Train/holdout manifests have different label columns")
    else:
        classnames = artifact["classnames"].astype(str).tolist()
    if train_y.shape[1] != len(classnames) or holdout_y.shape[1] != len(classnames):
        raise ValueError("Label width does not match classnames")
    if len(args.allow_undefined_holdout_label) != len(set(args.allow_undefined_holdout_label)):
        raise ValueError("Duplicate --allow-undefined-holdout-label values")
    unknown_allowed = sorted(set(args.allow_undefined_holdout_label) - set(classnames))
    if unknown_allowed:
        raise ValueError(f"Unknown allowed holdout labels: {unknown_allowed}")

    train_positives = train_y.sum(axis=0)
    train_bad = np.flatnonzero((train_positives == 0) | (train_positives == len(train_y)))
    if len(train_bad):
        raise ValueError(f"train lacks both classes for label columns {train_bad.tolist()}")
    holdout_positives = holdout_y.sum(axis=0)
    holdout_bad = np.flatnonzero((holdout_positives == 0) | (holdout_positives == len(holdout_y)))
    holdout_undefined = [classnames[index] for index in holdout_bad]
    if set(holdout_undefined) != set(args.allow_undefined_holdout_label):
        raise ValueError(
            "holdout undefined labels do not match the explicit allowance: "
            f"observed={holdout_undefined}, allowed={args.allow_undefined_holdout_label}"
        )
    undefined_indices = frozenset(int(index) for index in holdout_bad)
    report_missing = [name for name in CTRATE_REPORT_CLASSNAMES if name not in classnames]
    if report_missing and args.selection_classes == "ctrate-table11":
        raise ValueError(f"CT-RATE report classes are absent: {report_missing}")
    report_indices = None if report_missing else np.asarray(
        [classnames.index(name) for name in CTRATE_REPORT_CLASSNAMES], dtype=int
    )
    if args.selection_classes == "ctrate-table11" and set(holdout_undefined) & set(CTRATE_REPORT_CLASSNAMES):
        raise ValueError("A CT-RATE selection label is undefined in the holdout split")
    selection_classnames = (
        list(CTRATE_REPORT_CLASSNAMES)
        if args.selection_classes == "ctrate-table11"
        else [name for name in classnames if name not in holdout_undefined]
    )

    device = torch.device(args.device)
    feature_dim = int(train_x.shape[1])
    if holdout_x.shape[1] != feature_dim:
        raise ValueError("Train and holdout feature dimensions differ")
    feature_metadata = (
        json.loads(str(artifact["metadata_json"].item()))
        if "metadata_json" in artifact
        else None
    )
    feature_definition = (
        feature_metadata.get("feature_definition")
        if feature_metadata is not None
        else None
    )
    if feature_definition is not None:
        if not isinstance(feature_definition, dict):
            raise ValueError("feature_definition must be a JSON object")
        if int(feature_definition.get("feature_dim", -1)) != feature_dim:
            raise ValueError("feature_definition feature_dim disagrees with features")
    probe_spec = make_probe_spec("linear")
    classifier = build_probe(feature_dim, classnames, probe_spec).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    pos_weight = (len(train_y) - train_positives) / train_positives
    loss_spec = make_loss_spec("balanced-bce")
    loss_function = build_probe_loss(
        classnames, torch.from_numpy(pos_weight).float().to(device), loss_spec
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_learning_rate
        )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        drop_last=False,
    )
    train_eval_x = torch.from_numpy(train_x).to(device)
    train_eval_y = torch.from_numpy(train_y).to(device)
    holdout_eval_x = torch.from_numpy(holdout_x).to(device)
    holdout_eval_y = torch.from_numpy(holdout_y).to(device)

    history = []
    best_auc = float("-inf")
    patience_best_auc = float("-inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    early_stopped = False
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        total_loss = 0.0
        total_count = 0
        for feature_batch, label_batch in train_loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(classifier(feature_batch), label_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(feature_batch)
            total_count += len(feature_batch)
        classifier.eval()
        train_loss, train_auc, train_per_class = evaluate_classifier(
            classifier, train_eval_x, train_eval_y, loss_function, device
        )
        holdout_loss, holdout_auc, holdout_per_class = evaluate_classifier(
            classifier, holdout_eval_x, holdout_eval_y, loss_function, device, undefined_indices
        )
        train_table11 = None if report_indices is None else float(np.mean(np.asarray(train_per_class)[report_indices]))
        holdout_table11 = None if report_indices is None else float(np.mean(np.asarray(holdout_per_class)[report_indices]))
        selection_auc = holdout_table11 if args.selection_classes == "ctrate-table11" else holdout_auc
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimization_train_loss": total_loss / total_count,
            "train_loss": train_loss,
            "train_macro_auc": train_auc,
            "train_macro_auc_table11": train_table11,
            "train_per_class_auc": dict(zip(classnames, train_per_class)),
            "holdout_loss": holdout_loss,
            "holdout_macro_auc": holdout_auc,
            "holdout_macro_auc_table11": holdout_table11,
            "holdout_per_class_auc": dict(zip(classnames, holdout_per_class)),
            "selection_auc": selection_auc,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if selection_auc > best_auc + 1e-8:
            best_auc = selection_auc
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in classifier.state_dict().items()
            }
        if selection_auc > patience_best_auc + args.early_stopping_min_delta:
            patience_best_auc = selection_auc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                early_stopped = True
                break
        if scheduler is not None:
            scheduler.step()

    if best_state is None:
        raise RuntimeError("No FC checkpoint was selected")
    classifier.load_state_dict(best_state, strict=True)
    classifier.eval()
    final_train_loss, final_train_auc, final_train_per_class = evaluate_classifier(
        classifier, train_eval_x, train_eval_y, loss_function, device
    )
    final_holdout_loss, final_holdout_auc, final_holdout_per_class = evaluate_classifier(
        classifier, holdout_eval_x, holdout_eval_y, loss_function, device, undefined_indices
    )
    final_train_table11 = None if report_indices is None else float(np.mean(np.asarray(final_train_per_class)[report_indices]))
    final_holdout_table11 = None if report_indices is None else float(np.mean(np.asarray(final_holdout_per_class)[report_indices]))
    final_selection_auc = final_holdout_table11 if args.selection_classes == "ctrate-table11" else final_holdout_auc
    if abs(final_selection_auc - best_auc) > 1e-7:
        raise AssertionError("Reloaded best FC does not reproduce its holdout AUC")

    feature_sha256 = sha256_file(feature_path)
    train_manifest_sha256 = sha256_file(train_manifest)
    holdout_manifest_sha256 = sha256_file(holdout_manifest)
    checkpoint = {
        "format_version": 1,
        "dataset": args.dataset_name,
        "state_dict": best_state,
        "feature_dim": feature_dim,
        "classnames": classnames,
        "feature_artifact": str(feature_path),
        "feature_sha256": feature_sha256,
        "feature_definition": feature_definition,
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": train_manifest_sha256,
        "holdout_manifest": str(holdout_manifest),
        "holdout_manifest_sha256": holdout_manifest_sha256,
        "train_case_ids": sorted(set(map(str, train_ids))),
        "holdout_case_ids": sorted(set(map(str, holdout_ids))),
        "train_group_ids": sorted(map(str, reliable_groups(train_groups))),
        "holdout_group_ids": sorted(map(str, reliable_groups(holdout_groups))),
        "best_epoch": best_epoch,
        "best_holdout_macro_auc": final_holdout_auc,
        "best_val_macro_auc_table11": final_holdout_table11,
        "selection_classes": args.selection_classes,
        "selection_classnames": selection_classnames,
        "holdout_undefined_classnames": holdout_undefined,
        "best_selection_auc": best_auc,
        "balanced_loss": True,
        "pos_weight": pos_weight.astype(float).tolist(),
        "loss_spec": loss_spec,
        "probe_spec": probe_spec,
        "optimizer": "AdamW",
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "min_learning_rate": args.min_learning_rate,
    }
    atomic_torch_save(checkpoint_path, checkpoint)
    summary = {
        "status": "accepted",
        "dataset": args.dataset_name,
        "checkpoint": str(checkpoint_path),
        "features": str(feature_path),
        "feature_sha256": feature_sha256,
        "feature_definition": feature_definition,
        "feature_compatibility": (
            "Feature-worker definition recorded; no archived-feature equivalence claim"
            if feature_definition is not None
            else "Legacy feature definition unknown; NPZ schema, class order, and dimensions validated only; no archived-feature equivalence claim"
        ),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": train_manifest_sha256,
        "holdout_manifest": str(holdout_manifest),
        "holdout_manifest_sha256": holdout_manifest_sha256,
        "train_samples": len(train_x),
        "holdout_samples": len(holdout_x),
        "feature_dim": feature_dim,
        "num_classes": len(classnames),
        "classnames": classnames,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "early_stopped": early_stopped,
        "best_train_loss": final_train_loss,
        "best_train_macro_auc": final_train_auc,
        "best_train_macro_auc_table11": final_train_table11,
        "best_holdout_loss": final_holdout_loss,
        "best_holdout_macro_auc": final_holdout_auc,
        "best_val_macro_auc_table11": final_holdout_table11,
        "selection_classes": args.selection_classes,
        "selection_classnames": selection_classnames,
        "holdout_undefined_classnames": holdout_undefined,
        "best_selection_auc": best_auc,
        "best_train_per_class_auc": dict(zip(classnames, final_train_per_class)),
        "best_holdout_per_class_auc": dict(zip(classnames, final_holdout_per_class)),
        "trainable_parameters": sum(parameter.numel() for parameter in classifier.parameters()),
        "frozen_feature_extractor": True,
        "optimizer": "AdamW",
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "min_learning_rate": args.min_learning_rate,
        "patience": args.patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "balanced_loss": True,
        "loss_spec": loss_spec,
        "probe_spec": probe_spec,
        "seed": args.seed,
        "acceptance": {
            "case_overlap": 0,
            "group_overlap": 0,
            "train_group_ids_available": bool(reliable_groups(train_groups)),
            "holdout_group_ids_available": bool(reliable_groups(holdout_groups)),
            "finite_features": True,
            "both_classes_present_for_all_train_labels": True,
            "holdout_undefined_labels_explicitly_allowed": set(holdout_undefined) == set(args.allow_undefined_holdout_label),
            "best_checkpoint_reproduced": True,
            "test_labels_used_for_selection": False,
        },
        "history": history,
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
