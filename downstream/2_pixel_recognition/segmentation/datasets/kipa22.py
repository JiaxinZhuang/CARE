from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from monai.data import Dataset
from monai.transforms import (
    ClassesToIndicesd,
    Compose,
    EnsureTyped,
    MapTransform,
    RandAdjustContrastd,
    RandAffined,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
)


LABELS = {
    0: "background",
    1: "renal_vein",
    2: "kidney",
    3: "renal_artery",
    4: "tumor",
}
EXPECTED_SPLIT_COUNTS = {"train": 49, "val": 7, "test": 14}


def read_b2nd(path: str | Path) -> np.ndarray:
    import blosc2

    array = blosc2.open(str(path))
    try:
        return np.asarray(array[:])
    except Exception:
        return np.asarray(array[...])


class LoadKiPA22B2ndd(MapTransform):
    """Load an nnU-Net image/seg pair and preserve its valid-voxel mask."""

    def __init__(self, image_key="image", label_key="label"):
        super().__init__([image_key, label_key])
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, data):
        result = dict(data)
        image = read_b2nd(result[self.image_key]).astype(np.float32, copy=False)
        label = read_b2nd(result[self.label_key]).astype(np.int16, copy=False)
        if image.shape != label.shape:
            raise ValueError(
                f"Image/label shape mismatch for {result.get('case_id')}: "
                f"{image.shape} != {label.shape}"
            )
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"Expected [1,D,H,W], got {image.shape}")

        valid_mask = label >= 0
        valid_values = set(int(value) for value in np.unique(label[valid_mask]))
        if not valid_values.issubset(LABELS):
            raise ValueError(
                f"Unexpected labels for {result.get('case_id')}: "
                f"{sorted(valid_values)}"
            )
        result[self.image_key] = image
        result[self.label_key] = np.maximum(label, 0).astype(np.int16, copy=False)
        result["valid_mask"] = valid_mask.astype(bool, copy=False)
        return result


LoadKiPA22Arrayd = LoadKiPA22B2ndd


class KiPA22Intensityd(MapTransform):
    """Select nnU-Net z-score input or reconstruct offset-corrected HU."""

    def __init__(
        self,
        keys,
        mode: str,
        nnunet_mean: float,
        nnunet_std: float,
        raw_offset: float,
        window_min: float,
        window_max: float,
    ):
        super().__init__(keys)
        if mode not in {"nnunet", "offset_window"}:
            raise ValueError(f"Unsupported KiPA22 intensity mode: {mode}")
        if window_max <= window_min:
            raise ValueError("window_max must be greater than window_min")
        self.mode = mode
        self.nnunet_mean = float(nnunet_mean)
        self.nnunet_std = float(nnunet_std)
        self.raw_offset = float(raw_offset)
        self.window_min = float(window_min)
        self.window_max = float(window_max)

    def __call__(self, data):
        result = dict(data)
        if self.mode == "nnunet":
            return result
        for key in self.key_iterator(result):
            image = np.asarray(result[key], dtype=np.float32)
            image = image * self.nnunet_std + self.nnunet_mean - self.raw_offset
            image = np.clip(image, self.window_min, self.window_max)
            image = (image - self.window_min) / (self.window_max - self.window_min)
            result[key] = image.astype(np.float32, copy=False)
        return result


def _fingerprint_statistics(preprocessed_root: Path) -> tuple[float, float]:
    with (preprocessed_root / "dataset_fingerprint.json").open() as handle:
        fingerprint = json.load(handle)
    properties = fingerprint["foreground_intensity_properties_per_channel"]["0"]
    return float(properties["mean"]), float(properties["std"])


def build_kipa22_transform_parts(args, split: str):
    if split not in EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unknown KiPA22 split: {split}")
    root = Path(args.data_dir)
    mean, std = _fingerprint_statistics(root)
    common = [
        LoadKiPA22B2ndd(),
        KiPA22Intensityd(
            keys=["image"],
            mode=getattr(args, "intensity_mode", "offset_window"),
            nnunet_mean=mean,
            nnunet_std=std,
            raw_offset=getattr(args, "raw_offset", 1024.0),
            window_min=getattr(args, "a_min", -150.0),
            window_max=getattr(args, "a_max", 400.0),
        ),
    ]
    if split != "train":
        return Compose(
            common + [EnsureTyped(keys=["image", "label", "valid_mask"])]
        ), None

    roi = (
        int(getattr(args, "roi_x", 160)),
        int(getattr(args, "roi_y", 128)),
        int(getattr(args, "roi_z", 112)),
    )
    if any(size <= 0 for size in roi):
        raise ValueError(f"KiPA22 ROI dimensions must be positive, got {roi}")
    configured_ratios = getattr(args, "class_crop_ratios", "1,3,1,3,4")
    ratios = (
        [float(value) for value in configured_ratios.split(",")]
        if isinstance(configured_ratios, str)
        else [float(value) for value in configured_ratios]
    )
    if len(ratios) != len(LABELS) or any(value < 0 for value in ratios):
        raise ValueError("class_crop_ratios must contain five non-negative values")
    if sum(ratios) <= 0:
        raise ValueError("class_crop_ratios must contain a positive value")
    num_samples = int(getattr(args, "num_samples", 2))
    if num_samples < 1:
        raise ValueError("num_samples must be positive")

    deterministic = common + [
        SpatialPadd(keys=["image", "label", "valid_mask"], spatial_size=roi),
        ClassesToIndicesd(
            keys=["label"], num_classes=len(LABELS), max_samples_per_class=10000
        ),
    ]
    random = [
        RandCropByLabelClassesd(
            keys=["image", "label", "valid_mask"],
            label_key="label",
            spatial_size=roi,
            ratios=ratios,
            num_classes=len(LABELS),
            num_samples=num_samples,
            indices_key="label_cls_indices",
            max_samples_per_class=10000,
        )
    ]
    if not getattr(args, "disable_augmentation", False):
        random.extend(
            [
                RandAffined(
                    keys=["image", "label", "valid_mask"],
                    prob=0.2,
                    rotate_range=(0.15, 0.15, 0.15),
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest", "nearest"),
                    padding_mode="zeros",
                ),
                RandFlipd(
                    keys=["image", "label", "valid_mask"], prob=0.5, spatial_axis=0
                ),
                RandFlipd(
                    keys=["image", "label", "valid_mask"], prob=0.5, spatial_axis=1
                ),
                RandFlipd(
                    keys=["image", "label", "valid_mask"], prob=0.5, spatial_axis=2
                ),
                RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.05),
                RandGaussianSmoothd(
                    keys=["image"],
                    prob=0.1,
                    sigma_x=(0.5, 1.0),
                    sigma_y=(0.5, 1.0),
                    sigma_z=(0.5, 1.0),
                ),
                RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.5)),
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.15),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15),
            ]
        )
    random.append(EnsureTyped(keys=["image", "label", "valid_mask"]))
    return Compose(deterministic), Compose(random)


def build_kipa22_transforms(args, split: str):
    deterministic, random = build_kipa22_transform_parts(args, split)
    if random is None:
        return deterministic
    return Compose(list(deterministic.transforms) + list(random.transforms))


def _read_split_csv(path: Path, split: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"KiPA22 split CSV has no header: {path}")
        missing = {"case_id", "split"}.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"KiPA22 split CSV is missing columns {sorted(missing)}: {path}"
            )
        rows = list(reader)
    for row in rows:
        declared = row["split"].strip()
        if declared != split:
            raise ValueError(
                f"Split {declared!r} in {path} does not match file split {split!r}"
            )
        row["split"] = split
    return rows


def _allows_noncanonical_split(args) -> bool:
    if hasattr(args, "allow_noncanonical_split"):
        return bool(args.allow_noncanonical_split)
    return bool(getattr(args, "allow_synthetic_smoke_data", False))


def _validated_manifest(
    args, root: Path
) -> tuple[Path, list[dict[str, str]], dict[str, int], bool]:
    configured_split_dir = getattr(args, "split_dir", None)
    configured_manifest = getattr(args, "manifest_path", None)
    if configured_split_dir:
        source_path = Path(configured_split_dir)
        rows = []
        for split in EXPECTED_SPLIT_COUNTS:
            rows.extend(_read_split_csv(source_path / f"{split}.csv", split))
    else:
        source_path = (
            Path(configured_manifest) if configured_manifest else root / "manifest.csv"
        )
        with source_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"KiPA22 manifest has no header: {source_path}")
            missing = {"case_id", "split"}.difference(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"KiPA22 manifest is missing columns: {sorted(missing)}"
                )
            rows = list(reader)

    seen: dict[str, str] = {}
    counts = {name: 0 for name in EXPECTED_SPLIT_COUNTS}
    for row in rows:
        case_id = row["case_id"].strip()
        split = row["split"].strip()
        if not case_id:
            raise ValueError(f"Empty case_id in {source_path}")
        if split not in EXPECTED_SPLIT_COUNTS:
            raise ValueError(f"Unknown split {split!r} for {case_id}")
        if case_id in seen:
            raise ValueError(
                f"Case ID {case_id} appears in both {seen[case_id]} and {split}; "
                "KiPA22 split IDs must be disjoint"
            )
        seen[case_id] = split
        counts[split] += 1
        row["case_id"] = case_id
        row["split"] = split

    canonical_split_counts = counts == EXPECTED_SPLIT_COUNTS
    if not canonical_split_counts and not _allows_noncanonical_split(args):
        raise ValueError(
            f"Unexpected KiPA22 split counts: {counts}; expected "
            f"{EXPECTED_SPLIT_COUNTS}. Set allow_noncanonical_split=True to "
            "permit different split counts."
        )
    return source_path, rows, counts, canonical_split_counts


def get_kipa22_split_metadata(args) -> dict[str, object]:
    """Report count status; this does not compare IDs with a reference split."""
    _, _, counts, canonical_split_counts = _validated_manifest(
        args, Path(args.data_dir)
    )
    return {
        "allow_noncanonical_split": _allows_noncanonical_split(args),
        "split_counts": counts,
        "canonical_split_counts": canonical_split_counts,
    }


class KiPA22Dataset(Dataset):
    def __init__(self, args, split="train", transform=None):
        if split not in EXPECTED_SPLIT_COUNTS:
            raise ValueError(f"Unknown KiPA22 split: {split}")
        root = Path(args.data_dir)
        manifest_path, rows, split_counts, canonical_split_counts = (
            _validated_manifest(args, root)
        )
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            raise ValueError(f"KiPA22 split {split!r} is empty in {manifest_path}")

        if getattr(args, "split_dir", None):
            plan_dir = root / split
        else:
            split_cache_dir = root / split
            plan_dir = (
                split_cache_dir
                if split_cache_dir.is_dir()
                else root / "nnUNetPlans_3d_fullres"
            )
        data_list = []
        for row in split_rows:
            case_id = row["case_id"]
            item = {
                "case_id": case_id,
                "split": split,
                "image": str(plan_dir / f"{case_id}.b2nd"),
                "label": str(plan_dir / f"{case_id}_seg.b2nd"),
                "properties": str(plan_dir / f"{case_id}.pkl"),
                "reference_image": row.get("image_path", ""),
                "reference_label": row.get("label_path", ""),
            }
            missing = [
                key
                for key in ("image", "label", "properties")
                if not Path(item[key]).is_file()
            ]
            if missing:
                raise FileNotFoundError(f"Missing {missing} for {case_id}")
            data_list.append(item)

        self.args = args
        self.split = split
        self.manifest_path = manifest_path
        self.split_counts = split_counts
        self.canonical_split_counts = canonical_split_counts
        self.case_ids = [item["case_id"] for item in data_list]
        super().__init__(data=data_list, transform=transform)
