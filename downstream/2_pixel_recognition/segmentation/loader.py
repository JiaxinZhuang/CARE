from __future__ import annotations

from pathlib import Path

from monai import data
from monai.data import CacheDataset, PersistentDataset
from monai.data.utils import pickle_hashing

from datasets.kipa22 import KiPA22Dataset, build_kipa22_transforms


def _build_dataset(args, split: str):
    transform = build_kipa22_transforms(args, split=split)
    base = KiPA22Dataset(args, split=split, transform=None)
    persistent_cache_dir = getattr(args, "persistent_cache_dir", None)
    if persistent_cache_dir:
        cache_dir = Path(persistent_cache_dir) / split
        return PersistentDataset(
            data=base.data,
            transform=transform,
            cache_dir=cache_dir,
            hash_transform=pickle_hashing,
        )

    cache_rate = float(getattr(args, "cache_rate", 0.0))
    if not 0.0 <= cache_rate <= 1.0:
        raise ValueError("cache_rate must be between 0 and 1")
    if cache_rate > 0:
        return CacheDataset(
            data=base.data,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=int(getattr(args, "cache_workers", 0)),
        )
    base.transform = transform
    return base


def get_seg_loader(args, train_ds, val_ds):
    num_workers = int(getattr(args, "num_workers", 0))
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    common = {
        "num_workers": num_workers,
        "pin_memory": bool(getattr(args, "pin_memory", True)),
        "persistent_workers": num_workers > 0,
    }
    train_loader = data.DataLoader(
        train_ds,
        batch_size=int(getattr(args, "batch_size", 1)),
        shuffle=True,
        **common,
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader


def get_loader(args):
    if str(args.dataset_name).lower() != "kipa22":
        raise NotImplementedError(
            f"Dataset {args.dataset_name} is not included in this release slice"
        )
    # Training deliberately constructs only development splits. Test evaluation
    # must instantiate KiPA22Dataset(args, split="test", ...) explicitly.
    train_ds = _build_dataset(args, split="train")
    val_ds = _build_dataset(args, split="val")
    return get_seg_loader(args, train_ds, val_ds)
