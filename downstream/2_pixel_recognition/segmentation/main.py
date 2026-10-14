from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from datasets.kipa22 import get_kipa22_split_metadata
from loader import get_loader
from losses import MaskedSegmentationLoss
from metrics import KIPA22_LABELS as LABELS
from metrics import PolyMetrics
from models.net import build_optimizer_parameter_groups, get_model


def default_care_checkpoint() -> Path:
    return Path(__file__).resolve().parents[3] / "ckpts" / "care.pth"


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune CARE on KiPA22.")
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--split_dir", default=None, type=Path)
    parser.add_argument("--manifest_path", default=None, type=Path)
    parser.add_argument("--logdir", required=True, type=Path)
    parser.add_argument("--persistent_cache_dir", default=None, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow_noncanonical_split", action="store_true")

    parser.add_argument("--dataset_name", default="kipa22")
    parser.add_argument("--model_name", choices=("unetr_huge",), default="unetr_huge")
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=5, type=int)
    parser.add_argument("--roi_x", default=160, type=int)
    parser.add_argument("--roi_y", default=128, type=int)
    parser.add_argument("--roi_z", default=112, type=int)
    parser.add_argument("--sw_batch_size", default=1, type=int)

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--cache_rate", default=1.0, type=float)
    parser.add_argument("--cache_workers", default=8, type=int)
    parser.add_argument("--class_crop_ratios", default="1,3,1,3,4")
    parser.add_argument("--num_samples", default=2, type=int)
    parser.add_argument("--disable_augmentation", action="store_true")
    parser.add_argument("--intensity_mode", default="offset_window")
    parser.add_argument("--raw_offset", default=1024.0, type=float)
    parser.add_argument("--a_min", default=-150.0, type=float)
    parser.add_argument("--a_max", default=400.0, type=float)

    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--val_every", default=10, type=int)
    parser.add_argument("--checkpoint_every", default=10, type=int)
    parser.add_argument("--optim_lr", default=3e-4, type=float)
    parser.add_argument("--encoder_lr", default=1e-4, type=float)
    parser.add_argument("--patch_embed_lr", default=5e-5, type=float)
    parser.add_argument("--reg_weight", default=1e-5, type=float)
    parser.add_argument("--grad_clip_norm", default=1.0, type=float)
    parser.add_argument("--grad_accum_steps", default=1, type=int)

    parser.add_argument("--loss_variant", default="dice_ce_boundary")
    parser.add_argument("--class_weights", default="0.2,2,0.5,3,3")
    parser.add_argument("--boundary_weight", default=0.1, type=float)
    parser.add_argument("--boundary_classes", default="4")
    parser.add_argument("--focal_gamma", default=2.0, type=float)

    parser.add_argument("--encoder_tuning", choices=("lora",), default="lora")
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16.0, type=float)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--pretrained_path", default=default_care_checkpoint(), type=Path)
    parser.add_argument("--seed", default=20260904, type=int)
    return parser.parse_args()


def _validate_args(args):
    if args.dataset_name.lower() != "kipa22":
        raise ValueError("This release entry point supports only KiPA22")
    if args.out_channels != len(LABELS):
        raise ValueError("KiPA22 requires five output classes")
    if args.max_epochs < 1 or not 0 <= args.warmup_epochs <= args.max_epochs:
        raise ValueError("Require 0 <= warmup_epochs <= max_epochs")
    if args.val_every < 1 or args.checkpoint_every < 1:
        raise ValueError("Validation and checkpoint intervals must be positive")
    if args.grad_accum_steps != 1:
        raise ValueError("This entry point supports grad_accum_steps=1 only")
    if args.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive")
    if not args.pretrained_path.is_file():
        raise FileNotFoundError(f"CARE checkpoint not found: {args.pretrained_path}")


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def atomic_write_json(path: Path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )
    os.replace(temporary, path)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def atomic_torch_save(path: Path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    model, optimizer, scheduler, scaler, epoch, global_step, best_metric,
    best_metric_epoch,
):
    return {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "best_metric_epoch": int(best_metric_epoch),
        "rng_state": capture_rng_state(),
    }


def save_training_checkpoint(path: Path, *args):
    atomic_torch_save(path, checkpoint_payload(*args))


def _parse_numbers(value, cast=float):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _build_scheduler(optimizer, args):
    if args.warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    warmup = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    decay_epochs = args.max_epochs - args.warmup_epochs
    if decay_epochs == 0:
        return warmup
    cosine = CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=1e-6)
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs]
    )


def _infer(model, inputs, args, amp_enabled):
    with torch.amp.autocast("cuda", enabled=amp_enabled):
        return sliding_window_inference(
            inputs=inputs,
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=0.5,
        )


def validate(model, loader, args, device):
    model.eval()
    metrics = PolyMetrics(args, mode="fast")
    amp_enabled = device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            valid = batch.get("valid_mask")
            if valid is not None:
                valid = valid.to(device)
            logits = _infer(model, inputs, args, amp_enabled)
            metrics.update_from_logits(
                logits, labels, valid, case_ids=batch.get("case_id")
            )
    return metrics.compute()


def _append_metrics(path: Path, payload):
    with path.open("a") as handle:
        handle.write(
            json.dumps(_jsonable(payload), allow_nan=False, sort_keys=True) + "\n"
        )


def main():
    args = get_args()
    _validate_args(args)
    args.logdir.mkdir(parents=True, exist_ok=True)
    if args.persistent_cache_dir is not None:
        args.persistent_cache_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.logdir / "config.json"
    if config_path.exists():
        raise FileExistsError(f"Run already initialized: {config_path}")
    split_metadata = get_kipa22_split_metadata(args)
    config = _jsonable(vars(args))
    config.update(split_metadata)
    atomic_write_json(config_path, config)

    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    train_loader, val_loader = get_loader(args)
    model = get_model(args, load_pretrained=True).to(device)
    parameter_groups = build_optimizer_parameter_groups(model, args)
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.reg_weight)
    scheduler = _build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_function = MaskedSegmentationLoss(
        args.out_channels,
        _parse_numbers(args.class_weights),
        variant=args.loss_variant,
        boundary_weight=args.boundary_weight,
        focal_gamma=args.focal_gamma,
        boundary_classes=tuple(_parse_numbers(args.boundary_classes, int)),
    ).to(device)

    global_step = 0
    best_metric = -math.inf
    best_metric_epoch = -1

    amp_enabled = device.type == "cuda"
    latest_path = args.logdir / "latest_model.pth"
    best_path = args.logdir / "best_model.pth"
    for epoch in range(args.max_epochs):
        model.train()
        losses = []
        for batch in train_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            valid_mask = batch.get("valid_mask")
            if valid_mask is not None:
                valid_mask = valid_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(inputs)
                loss = loss_function(logits, labels, valid_mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, step {global_step + 1}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            losses.append(float(loss.detach()))
        if not losses:
            raise RuntimeError("Training loader is empty")
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{args.max_epochs} loss={np.mean(losses):.6f} "
            f"grad_norm={float(grad_norm):.6f}"
        )

        metrics = None
        if (epoch + 1) % args.val_every == 0:
            metrics = validate(model, val_loader, args, device)
            metrics.update(
                {"epoch": epoch, "global_step": global_step, **split_metadata}
            )
            _append_metrics(args.logdir / "metrics.jsonl", metrics)
            print(json.dumps(_jsonable(metrics), allow_nan=False, sort_keys=True))
            if metrics["focus_DSC"] > best_metric:
                best_metric = metrics["focus_DSC"]
                best_metric_epoch = epoch
                save_training_checkpoint(
                    best_path, model, optimizer, scheduler, scaler, epoch,
                    global_step, best_metric, best_metric_epoch,
                )

        if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.max_epochs:
            save_training_checkpoint(
                latest_path, model, optimizer, scheduler, scaler, epoch,
                global_step, best_metric, best_metric_epoch,
            )

    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was produced; check val_every")

    del loss_function, optimizer, scheduler, scaler, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded_model = get_model(args, load_pretrained=False).to(device)
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    reloaded_model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    final_metrics = validate(reloaded_model, val_loader, args, device)
    final_metrics.update(
        {
            "best_metric": float(best_checkpoint["best_metric"]),
            "best_metric_epoch": int(best_checkpoint["best_metric_epoch"]),
            "checkpoint": str(best_path),
            "split": "val",
            "strict_independent_reload": True,
            **split_metadata,
        }
    )
    atomic_write_json(args.logdir / "best_val_metrics.json", final_metrics)
    print(
        json.dumps(
            _jsonable(final_metrics), allow_nan=False, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
