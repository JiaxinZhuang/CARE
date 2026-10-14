"""Train or evaluate the frozen CARE ROI classifier for PANORAMA."""

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from ct_classifier import CT3DClassifier
from loader import make_3d_dataset
from metrics import build_linear_metric
from utils import AverageMeter


torch.multiprocessing.set_sharing_strategy("file_system")
SUPPORTED_DATASETS = ("panorama",)
SPLITS = ("train", "val", "test")
CHECKPOINT_FORMAT_VERSION = 1


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--data_items_path", required=True, type=Path)
    parser.add_argument("--npz_dir", required=True, type=Path)
    parser.add_argument("--care_checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    return parser


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("--lr must be positive and --weight_decay non-negative")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if args.inference_only and args.checkpoint_path is None:
        raise ValueError("--checkpoint_path is required with --inference_only")
    if not args.inference_only and args.checkpoint_path is not None:
        raise ValueError("--checkpoint_path is only valid with --inference_only")

    for name in ("data_items_path", "npz_dir", "care_checkpoint", "output_dir"):
        value = getattr(args, name)
        setattr(args, name, value.expanduser().resolve())
    if args.checkpoint_path is not None:
        args.checkpoint_path = args.checkpoint_path.expanduser().resolve()


def resolve_device(device_name):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device_name}")
    return device


def make_dataset(args, split):
    return make_3d_dataset(
        f"{args.dataset}:{split}",
        data_items_path=args.data_items_path,
        npz_dir=args.npz_dir,
    )


def class_spec(dataset):
    names = tuple(str(name) for name in dataset.cls_names)
    num_classes = int(dataset.num_classes)
    if num_classes != len(names) or len(set(names)) != len(names):
        raise ValueError("Dataset class names must be unique and match num_classes")
    return num_classes, names


def validate_datasets(datasets):
    reference = class_spec(datasets["train"])
    identifiers = {}
    for split, dataset in datasets.items():
        if class_spec(dataset) != reference:
            raise ValueError(f"Class mapping for {split} does not match train")
        missing = {"image", "npz_path"}.difference(dataset.data_df.columns)
        if missing:
            raise ValueError(f"Manifest for {split} is missing columns: {sorted(missing)}")
        identifiers[split] = {}
        for column in ("image", "npz_path"):
            if dataset.data_df[column].isna().any():
                raise ValueError(f"Manifest for {split} contains empty {column} values")
            identifiers[split][column] = set(dataset.data_df[column].astype(str))

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for column in ("image", "npz_path"):
            overlap = identifiers[left][column] & identifiers[right][column]
            if overlap:
                raise ValueError(
                    f"Split isolation failed for {column}: {left}/{right} share "
                    f"{len(overlap)} identifier(s)"
                )
    return reference


def load_datasets(args):
    datasets = {split: make_dataset(args, split) for split in SPLITS}
    num_classes, cls_names = validate_datasets(datasets)
    return datasets, num_classes, cls_names


def make_loader(dataset, args, device, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def build_model(args, num_classes, device):
    model = CT3DClassifier(
        encoder_name="ours",
        num_classes=num_classes,
        pretrained_path=args.care_checkpoint,
        freeze_encoder=True,
    )
    return model.to(device)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()
    for data, (_, targets) in tqdm(dataloader, desc="train"):
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        outputs = model(data)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), data.size(0))
    return losses.avg


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device, num_classes):
    model.eval()
    losses = AverageMeter()
    all_logits, all_targets, all_indices = [], [], []

    for data, (indices, targets) in tqdm(dataloader, desc="evaluate"):
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        outputs = model(data)
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(0)
        if outputs.ndim != 2 or outputs.shape[1] != num_classes:
            raise RuntimeError(
                f"Expected logits shaped [batch, {num_classes}], got {tuple(outputs.shape)}"
            )
        loss = criterion(outputs, targets)
        losses.update(loss.item(), data.size(0))
        all_logits.append(outputs.cpu())
        all_targets.append(targets.cpu())
        if torch.is_tensor(indices):
            all_indices.extend(int(index) for index in indices.tolist())
        else:
            all_indices.extend(int(index) for index in indices)

    if not all_logits:
        raise RuntimeError("Cannot evaluate an empty split")
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = build_linear_metric(num_classes)
    metrics.update(logits, targets)
    results = {name: float(value.item()) for name, value in metrics.compute().items()}
    return losses.avg, results, logits, targets, all_indices


def checkpoint_metadata(args, num_classes, cls_names):
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "dataset": args.dataset,
        "num_classes": num_classes,
        "cls_names": list(cls_names),
        "encoder_name": "ours",
        "freeze_encoder": True,
        "selection_metric": "AUC",
        "care_checkpoint": str(args.care_checkpoint),
    }


def validate_checkpoint(checkpoint, args, num_classes, cls_names):
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain a full model_state_dict")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata is missing")
    expected = checkpoint_metadata(args, num_classes, cls_names)
    for key in (
        "format_version",
        "dataset",
        "num_classes",
        "cls_names",
        "encoder_name",
        "freeze_encoder",
        "selection_metric",
    ):
        if metadata.get(key) != expected[key]:
            raise ValueError(
                f"Checkpoint metadata mismatch for {key}: "
                f"expected {expected[key]!r}, got {metadata.get(key)!r}"
            )


def write_roi_predictions(path, dataset, logits, targets, indices, num_classes):
    if len(indices) != len(targets) or len(set(indices)) != len(indices):
        raise RuntimeError("Prediction indices must identify each ROI exactly once")

    probabilities = softmax(logits, dim=1).numpy()
    records = []
    for row_number, (index, label) in enumerate(zip(indices, targets.tolist())):
        if index < 0 or index >= len(dataset.data_df):
            raise IndexError(f"Dataset index is out of range: {index}")
        row = dataset.data_df.iloc[index]
        if "label" in row and int(row["label"]) != int(label):
            raise ValueError("Cached ROI label does not match its manifest label")
        record = {
            "roi_id": str(row["npz_path"]),
            "case_id": str(row["image"]),
            "label": int(label),
        }
        for class_index in range(num_classes):
            record[f"prob_class_{class_index}"] = float(
                probabilities[row_number, class_index]
            )
        records.append(record)
    pd.DataFrame.from_records(records).to_csv(path, index=False)


def run_training(args, datasets, num_classes, cls_names, device):
    train_loader = make_loader(datasets["train"], args, device, shuffle=True)
    val_loader = make_loader(datasets["val"], args, device, shuffle=False)
    model = build_model(args, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_auc = -math.inf
    best_path = args.output_dir / "best_model.pth"
    metadata = checkpoint_metadata(args, num_classes, cls_names)
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _, _ = evaluate_epoch(
            model, val_loader, criterion, device, num_classes
        )
        scheduler.step()
        val_auc = val_metrics["AUC"]
        if not math.isfinite(val_auc):
            raise RuntimeError("Macro validation AUC is non-finite")
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_metrics={val_metrics}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "metadata": metadata,
                },
                best_path,
            )
            print(f"Saved best macro-AUC checkpoint to {best_path}")
    print(f"Best validation macro AUC: {best_auc:.6f}")


def run_inference(args, datasets, num_classes, cls_names, device):
    checkpoint = torch.load(
        args.checkpoint_path, map_location="cpu", weights_only=True
    )
    validate_checkpoint(checkpoint, args, num_classes, cls_names)
    model = build_model(args, num_classes, torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    dataset = datasets[args.split]
    dataloader = make_loader(dataset, args, device, shuffle=False)
    loss, metrics, logits, targets, indices = evaluate_epoch(
        model, dataloader, nn.CrossEntropyLoss(), device, num_classes
    )
    output_path = args.output_dir / f"predictions_{args.split}.csv"
    write_roi_predictions(
        output_path, dataset, logits, targets, indices, num_classes
    )
    print(f"{args.split}_loss={loss:.6f} {args.split}_metrics={metrics}")
    print(f"Saved per-ROI predictions to {output_path}")


def main():
    args = get_args_parser().parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, num_classes, cls_names = load_datasets(args)
    if args.inference_only:
        run_inference(args, datasets, num_classes, cls_names, device)
    else:
        run_training(args, datasets, num_classes, cls_names, device)


if __name__ == "__main__":
    main()
