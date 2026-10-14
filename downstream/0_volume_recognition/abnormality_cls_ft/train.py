"""Fine-tune CARE abnormality classification with validation-only selection."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

TASK_DIR = Path(__file__).resolve().parent
CARE_ROOT = TASK_DIR.parents[2]
for import_root in (str(CARE_ROOT), str(TASK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from datasets.ct_dataset import CTDataset

_DATASET_MODULE_PATH = Path(sys.modules[CTDataset.__module__].__file__).resolve()
if TASK_DIR / "datasets" not in _DATASET_MODULE_PATH.parents:
    raise ImportError(f"Resolved a non-task datasets package: {_DATASET_MODULE_PATH}")


def digest(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def save_checkpoint(path: str | Path, value) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_config(path: str | Path) -> tuple[dict, dict]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw.get("training"), dict):
        raise ValueError("Configuration requires a training object")
    config = dict(raw["training"])
    config["labels"] = raw.get("labels")
    config["aux_labels"] = raw.get("aux_labels", [])
    config["windows"] = raw.get("windows")
    config["source_version"] = raw.get("source_version", "unspecified")
    required = {
        "labels", "windows", "seed", "epochs", "microbatch", "accumulation",
        "workers", "head_lr", "encoder_lr", "lora_lr", "llrd", "weight_decay",
        "rank", "lora_blocks", "last_n", "head_type", "pool_type", "feature_l2",
        "schedule_mode", "train_droppath", "input_size", "resize_mode", "normalization",
    }
    missing = sorted(key for key in required if config.get(key) is None)
    if missing:
        raise ValueError(f"Configuration is missing required fields: {missing}")
    if not isinstance(config["labels"], list) or not config["labels"]:
        raise ValueError("labels must be a non-empty ordered list")
    if not isinstance(config["aux_labels"], list):
        raise ValueError("aux_labels must be an ordered list")
    names = list(config["labels"]) + list(config["aux_labels"])
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("labels and aux_labels must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("labels and aux_labels must be unique and disjoint")
    return config, raw


def default_pretrained_path() -> Path:
    return CARE_ROOT / "ckpts" / "care.pth"


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _model_logits(value):
    return value[0] if isinstance(value, tuple) else value


def orient_ras(tensor, spacing_zyx, affine):
    import nibabel as nib
    orientation = nib.orientations.io_orientation(affine)
    transform = nib.orientations.ornt_transform(
        orientation, nib.orientations.axcodes2ornt(("R", "A", "S")))
    xyzc = tensor.numpy().transpose(3, 2, 1, 0)
    canonical = nib.orientations.apply_orientation(xyzc, transform)
    output = torch.from_numpy(np.ascontiguousarray(canonical.transpose(3, 2, 1, 0)))
    spacing = np.asarray(spacing_zyx)[::-1][np.argsort(transform[:, 0])][::-1]
    return output, spacing.tolist()


def physical_resize(tensor, spacing, ratio, target):
    ratio = np.asarray(ratio, dtype=float)
    if ratio.shape != (3,) or not np.isfinite(ratio).all() or np.any(ratio <= 0):
        raise ValueError("voxel_ratio must have three finite positive values")
    units = np.asarray(tensor.shape[1:]) * np.asarray(spacing) / ratio
    scale = min(np.asarray(target) / units)
    shape = tuple(min(t, max(1, round(n * scale))) for t, n in zip(target, units))
    image = nn.functional.interpolate(
        tensor[None], size=shape, mode="trilinear", align_corners=False)[0]
    padding = [p for t, n in reversed(list(zip(target, shape)))
               for p in ((t - n) // 2, (t - n) - (t - n) // 2)]
    return nn.functional.pad(image, padding).contiguous()


def target_columns(config: dict) -> list[str]:
    return list(config["labels"]) + list(config.get("aux_labels") or [])


class ConfiguredDataSet(CTDataset):
    def _resize_to_input(self, tensor, row=None):
        config = getattr(self, "tuning_config", {})
        if config.get("orientation", "original") == "ras":
            import nibabel as nib
            tensor, spacing = orient_ras(
                tensor, [float(row[f"spacing_{axis}"]) for axis in "zyx"],
                nib.load(row["image_path"]).affine)
            row = dict(row)
            row.update({f"spacing_{axis}": value for axis, value in zip("zyx", spacing)})
        if "voxel_ratio" in config:
            return physical_resize(
                tensor, [float(row[f"spacing_{axis}"]) for axis in "zyx"],
                config["voxel_ratio"], self.input_size)
        return super()._resize_to_input(tensor, row)

    def _augment_multiwindow(self, image):
        if getattr(self, "tuning_config", {}).get("augmentation") == "flip":
            return image.flip(-1) if torch.rand(()) < 0.5 else image
        return super()._augment_multiwindow(image)

    def _make_multi_window_image_for_row(self, img_hu, row):
        out = np.empty((len(self.CT_WINDOWS), *img_hu.shape), dtype=np.float32)
        src = img_hu.astype(np.float32, copy=False)
        for index, (width, level) in enumerate(self.CT_WINDOWS):
            np.subtract(src, level - width / 2.0, out=out[index])
            np.divide(out[index], width, out=out[index])
            np.clip(out[index], 0.0, 1.0, out=out[index])
        return out


class InputDataset(ConfiguredDataSet):
    def __init__(self, manifest, config, split, *, augment=None, require_labels=True):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.config, self.epoch, self.split = config, 0, split
        super().__init__(
            manifest_path=manifest,
            label_columns=(target_columns(config) if split != "test" else list(config["labels"])),
            windows=config["windows"],
            input_size=tuple(config["input_size"]),
            require_labels=require_labels,
            augment=(split == "train") if augment is None else augment,
            resize_mode=config.get("resize_mode", "physical"),
            normalization=config.get("normalization", "pretrained"),
        )
        self.tuning_config = config
        self.frame = self.df

    def __getitem__(self, index):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config["seed"] + 1000003 * self.epoch + index)
            sample = super().__getitem__(index)
        sample["split"] = self.split
        return sample


def make_loader(dataset, config, *, shuffle=False, device=None):
    generator = torch.Generator().manual_seed(config["seed"] + getattr(dataset, "epoch", 0))
    workers = int(config.get("workers", 0))
    return DataLoader(
        dataset, batch_size=int(config.get("microbatch", 1)), shuffle=shuffle,
        num_workers=workers, prefetch_factor=1 if workers else None,
        pin_memory=getattr(device, "type", None) == "cuda", persistent_workers=False,
        generator=generator)


def classification_metrics(predictions: pd.DataFrame, labels: list[str]) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    per_class = {}
    for label in labels:
        y = predictions["label_" + label]
        probability = predictions["prob_" + label]
        valid = y.nunique() == 2
        per_class[label] = {
            "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
            "auc": float(roc_auc_score(y, probability)) if valid else None,
            "ap": float(average_precision_score(y, probability)) if valid else None}
    aucs = [per_class[label]["auc"] for label in labels]
    return {"n": len(predictions),
            "macro_auc": float(np.mean(aucs)) if all(value is not None for value in aucs) else None,
            "per_class": per_class}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def predict(model, loader, device, config, view="base") -> pd.DataFrame:
    if view not in {"base", "tta2"}:
        raise ValueError(f"Unsupported evaluation view: {view}")
    model.eval()
    rows = []
    primary = list(config["labels"])
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            with _autocast(device):
                base = _model_logits(model(image)).float().sigmoid().cpu().numpy()
            if base.ndim != 2 or base.shape[1] < len(primary):
                raise ValueError(
                    f"Model returned {base.shape} probabilities for {len(primary)} primary labels")
            views = [base[:, :len(primary)]]
            if view == "tta2":
                with _autocast(device):
                    flipped = _model_logits(model(image.flip(-1))).float().sigmoid().cpu().numpy()
                if flipped.shape != base.shape:
                    raise ValueError(
                        f"TTA views returned different probability shapes: {base.shape} and {flipped.shape}")
                views.append(flipped[:, :len(primary)])
            probabilities = np.mean(
                [values.astype(np.float64, copy=False) for values in views],
                axis=0, dtype=np.float64)
            labels = batch["label"].cpu().numpy()
            labeled = bool(torch.as_tensor(batch["has_labels"]).all())
            if labeled and (labels.ndim != 2 or labels.shape[1] < len(primary)):
                raise ValueError(
                    f"Loader returned {labels.shape} targets for {len(primary)} primary labels")
            count = len(probabilities)
            for index in range(count):
                row = {
                    "case_id": str(batch["case_id"][index]),
                    "group_id": str(batch["group_id"][index]),
                    "split": str(batch.get("split", ["predict"] * count)[index])}
                if labeled:
                    row.update({"label_" + label: int(value)
                                for label, value in zip(primary, labels[index, :len(primary)])})
                row.update({"prob_" + label: float(value)
                            for label, value in zip(primary, probabilities[index])})
                rows.append(row)
    predictions = pd.DataFrame(rows)
    if predictions.empty or predictions.case_id.duplicated().any():
        raise ValueError("Evaluation produced empty or duplicate case predictions")
    probability_columns = ["prob_" + label for label in primary]
    values = predictions[probability_columns].to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Evaluation produced invalid sigmoid probabilities")
    return predictions


def group_predictions(predictions: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    probability_columns = ["prob_" + label for label in labels]
    label_columns = ["label_" + label for label in labels]
    rows = []
    for group_id, frame in predictions.groupby("group_id", sort=False):
        row = {"group_id": str(group_id), "case_count": len(frame)}
        row.update({column: float(frame[column].mean()) for column in probability_columns})
        if set(label_columns).issubset(frame.columns):
            for column in label_columns:
                if frame[column].nunique() != 1:
                    raise ValueError(f"Label {column} is inconsistent within group {group_id}")
                row[column] = int(frame[column].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def write_prediction_outputs(predictions: pd.DataFrame, output_dir, name="predictions",
                             labels=None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(predictions, output_dir / f"predictions_{name}_volume.csv")
    if labels is not None:
        _atomic_csv(group_predictions(predictions, labels),
                    output_dir / f"predictions_{name}_group.csv")


def validate(model, loader, device, config, output_dir=None, name="val",
             compute_basic_metrics=True, view="base"):
    predictions = predict(model, loader, device, config, view=view)
    labeled = all("label_" + label in predictions for label in config["labels"])
    if compute_basic_metrics and not labeled:
        raise ValueError("Metrics require all configured label columns")
    result = classification_metrics(predictions, config["labels"]) if compute_basic_metrics else {
        "n": len(predictions), "macro_auc": None, "per_class": None}
    result["volume_predictions"] = len(predictions)
    groups = group_predictions(predictions, config["labels"])
    result["group_predictions"] = len(groups)
    if compute_basic_metrics:
        result["group"] = classification_metrics(groups, config["labels"])
    if output_dir is not None:
        write_prediction_outputs(predictions, output_dir, name, config["labels"])
    return result


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _dataset_labels(dataset, labels):
    frame = getattr(dataset, "frame", None)
    if frame is not None:
        return frame[labels].to_numpy(dtype=np.float32)
    return np.stack([dataset[index]["label"].numpy() for index in range(len(dataset))])


def train(model, train_manifest, validation_manifest, config, output, *, pretrained_path,
          resume=False, train_dataset=None, val_dataset=None, device=None):
    device = device or next(model.parameters()).device
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_dataset = train_dataset or InputDataset(train_manifest, config, "train")
    val_dataset = val_dataset or InputDataset(
        validation_manifest, config, "val", augment=False)
    train_frame, val_frame = getattr(train_dataset, "frame", None), getattr(val_dataset, "frame", None)
    if train_frame is not None and val_frame is not None:
        overlap = set(train_frame["group_id"].astype(str)) & set(val_frame["group_id"].astype(str))
        if overlap:
            raise ValueError(f"Group leakage between train and validation: {len(overlap)} keys")
    y = _dataset_labels(train_dataset, target_columns(config))
    positives = y.sum(0)
    if np.any(positives == 0) or np.any(positives == len(y)):
        raise ValueError("Training labels require positive and negative examples for every class")
    pos_weight = np.clip(np.sqrt((len(y) - positives) / positives), 0.25, 7.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32))
    optimizer = model.optimizer()
    history, best, bad, start, best_payload = [], -1.0, 0, 0, None
    pretrained_sha256 = digest(pretrained_path)
    audit = dict(getattr(model, "weight_audit", {}))
    audit.update(filename=Path(pretrained_path).name, sha256=pretrained_sha256)
    if resume:
        saved = torch.load(output / "last.pth", map_location="cpu", weights_only=False)
        if saved["config"] != config or saved["pretrained"]["sha256"] != pretrained_sha256:
            raise ValueError("Resume configuration or CARE base checkpoint differs")
        model.restore(saved["parameters"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, bad, start = saved["history"], saved["best"], saved["bad"], saved["epoch"]
        best_payload = saved["best_payload"]
        save_checkpoint(output / "best.pth", best_payload)
        restore_rng(saved["rng"])
    write_json(output / "training_config.json", {
        "config": config, "pos_weight": pos_weight.tolist(), "pretrained": audit,
        "train": len(train_dataset), "validation": len(val_dataset), "resumed": resume})
    for epoch in range(start, config["epochs"]):
        if bad >= config.get("patience", 6):
            break
        train_dataset.epoch = epoch
        scope = model.set_stage(optimizer, epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loader = make_loader(train_dataset, config, shuffle=True, device=device)
        losses = []
        for step, batch in enumerate(loader):
            with _autocast(device):
                raw = criterion(
                    _model_logits(model(batch["image"].to(device, non_blocking=True))),
                    batch["label"].to(device, non_blocking=True))
            if not torch.isfinite(raw):
                raise ValueError("Nonfinite training loss")
            accumulation = config["accumulation"]
            count = min(accumulation, len(loader) - (step // accumulation) * accumulation)
            (raw / count).backward()
            losses.append(float(raw.detach()))
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                model.update_lrs(optimizer, epoch + (step + 1) / len(loader))
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation = validate(
            model, make_loader(val_dataset, config, device=device), device, config, output, "val")
        selected = validation["macro_auc"]
        if selected is None:
            raise ValueError("A configured class is absent from validation")
        if selected > best:
            best, bad = selected, 0
            best_payload = {"parameters": model.delta(), "config": config, "epoch": epoch + 1,
                            "validation": validation, "pretrained": audit}
            save_checkpoint(output / "best.pth", best_payload)
            for level in ("volume", "group"):
                source = output / f"predictions_val_{level}.csv"
                if source.exists():
                    (output / f"predictions_val_best_{level}.csv").write_bytes(source.read_bytes())
        elif epoch >= config.get("early_stop_start", 6):
            bad += 1
        row = {"epoch": epoch + 1, "scope": scope, "train_loss": float(np.mean(losses)),
               "validation": validation, "best_val_macro_auc": best, "bad_epochs": bad,
               "lr_groups": [{"start": group["start"], "peak": group["peak"], "lr": group["lr"]}
                             for group in optimizer.param_groups]}
        history.append(row)
        save_checkpoint(output / "last.pth", {
            "parameters": model.delta(), "optimizer": optimizer.state_dict(), "config": config,
            "epoch": epoch + 1, "history": history, "best": best, "bad": bad,
            "best_payload": best_payload, "rng": rng_state(), "pretrained": audit})
        print(f"[EPOCH] {epoch + 1} val_macro_auc={selected:.6f} best={best:.6f}", flush=True)
    if best_payload is None:
        raise RuntimeError("Training completed without a validation-selected checkpoint")
    model.restore(best_payload["parameters"])
    replay = validate(
        model, make_loader(val_dataset, config, device=device), device, config, output, "val_best")
    if abs(replay["macro_auc"] - best) > 1e-6:
        raise ValueError("Best checkpoint validation replay mismatch")
    result = {"config": config, "best_epoch": best_payload["epoch"], "validation": replay,
              "selection": "validation volume macro AUC only",
              "checkpoint": "best.pth", "checkpoint_sha256": digest(output / "best.pth"),
              "pretrained": audit, "completed_epochs": len(history)}
    write_json(output / "results.json", result)
    return result


def build_model(config, pretrained_path, device):
    from model import TuningModel
    module_path = Path(sys.modules[TuningModel.__module__].__file__).resolve()
    if module_path.parent != TASK_DIR:
        raise ImportError(f"Resolved a non-task TuningModel: {module_path}")
    pretrained = {"path": str(Path(pretrained_path).resolve()), "sha256": digest(pretrained_path)}
    return TuningModel(config, pretrained, vendor=None, inject=True).to(device)


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TASK_DIR / "config.example.json")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True,
                        help="CARE base checkpoint used to initialize the visual encoder")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = get_args_parser().parse_args()
    config, _ = load_config(args.config)
    if args.workers is not None:
        config["workers"] = args.workers
    if args.epochs is not None:
        config["epochs"] = args.epochs
    output = args.output_dir.expanduser().resolve()
    if not args.resume and (output / "last.pth").exists():
        raise FileExistsError("Refusing implicit resume; use --resume or a new output directory")
    device = resolve_device(args.device)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    model = build_model(config, args.pretrained, device)
    train(model, args.train_manifest, args.validation_manifest, config, output,
          pretrained_path=args.pretrained, resume=args.resume, device=device)


if __name__ == "__main__":
    main()
