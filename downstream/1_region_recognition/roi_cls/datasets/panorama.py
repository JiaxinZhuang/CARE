"""PANORAMA ROI classification backed by existing preprocessed NPZ files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)
from torch.utils.data import Dataset

from .data_utils import OrganCropTransform


@dataclass(frozen=True)
class DataConfig:
    data_items_path: Path
    npz_dir: Path
    roi_size = (96, 96, 96)
    spacing = (1.0, 1.0, 3.0)
    intensity_range = (-150, 300)
    crop_margin = 5
    min_crop_size = 8
    class_names = (
        "background",
        "PDAC lesion",
        "Veins",
        "Arteries",
        "Pancreas parenchyma",
        "Pancreatic duct",
        "Common bile duct",
    )
    valid_labels = (0, 1, 2, 3, 4, 5, 6)

    def __post_init__(self):
        object.__setattr__(self, "data_items_path", Path(self.data_items_path).expanduser())
        object.__setattr__(self, "npz_dir", Path(self.npz_dir).expanduser())


class DataSet(Dataset):
    def __init__(
        self,
        data_items_path,
        npz_dir,
        split="train",
        fold=None,
        transformer=None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.config = DataConfig(data_items_path=data_items_path, npz_dir=npz_dir)
        self.split = split
        self.data_df = self._load_data_items()
        self.transform = transformer if transformer is not None else self._get_transforms()

    @property
    def num_classes(self):
        return len(self.config.class_names)

    @property
    def cls_names(self):
        return list(self.config.class_names)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        npz_path = self.config.npz_dir / row["npz_path"]
        if not npz_path.is_file():
            raise FileNotFoundError(f"Cached PANORAMA ROI not found: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as data:
            missing = {"image", "label", "organ_label"}.difference(data.files)
            if missing:
                raise ValueError(f"{npz_path} is missing NPZ arrays: {sorted(missing)}")
            image = torch.from_numpy(data["image"]).float()
            label_array = np.asarray(data["label"])
            organ_label_array = np.asarray(data["organ_label"])
        if tuple(image.shape) != (1, *self.config.roi_size):
            raise ValueError(f"Unexpected PANORAMA ROI shape in {npz_path}: {tuple(image.shape)}")
        if label_array.size != 1:
            raise ValueError(f"Expected one PANORAMA label in {npz_path}, got {label_array.shape}")
        if organ_label_array.size != 1:
            raise ValueError(
                f"Expected one PANORAMA organ label in {npz_path}, got {organ_label_array.shape}"
            )
        cached_label = float(label_array.reshape(-1)[0])
        cached_organ_label = float(organ_label_array.reshape(-1)[0])
        if not np.isfinite(cached_label) or not cached_label.is_integer():
            raise ValueError(f"Cached PANORAMA label is not an integer: {npz_path}")
        if not np.isfinite(cached_organ_label) or not cached_organ_label.is_integer():
            raise ValueError(f"Cached PANORAMA organ label is not an integer: {npz_path}")
        label = torch.tensor(int(cached_label), dtype=torch.long)
        organ_label = int(cached_organ_label)
        if label.item() != int(row["label"]):
            raise ValueError(f"Cached PANORAMA label disagrees with manifest: {npz_path}")
        if organ_label != int(row["organ_label"]):
            raise ValueError(f"Cached PANORAMA organ label disagrees with manifest: {npz_path}")
        if organ_label != label.item():
            raise ValueError(f"PANORAMA organ label disagrees with class label: {npz_path}")
        return image, (idx, label)

    def _get_transforms(self):
        return Compose(
            [
                LoadImaged(keys=["image", "mask"]),
                EnsureChannelFirstd(keys=["image", "mask"]),
                Orientationd(keys=["image", "mask"], axcodes="RAS"),
                Spacingd(keys=["image", "mask"], pixdim=self.config.spacing, mode=("bilinear", "nearest")),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=self.config.intensity_range[0],
                    a_max=self.config.intensity_range[1],
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                EnsureTyped(keys=["image", "mask"]),
                OrganCropTransform(
                    keys=["image", "mask"],
                    margin=self.config.crop_margin,
                    min_size=self.config.min_crop_size,
                ),
                ResizeWithPadOrCropd(keys=["image"], spatial_size=self.config.roi_size),
                ToTensord(keys=["image"]),
            ]
        )

    def _load_data_items(self):
        path = self.config.data_items_path
        if not path.is_file():
            raise FileNotFoundError(f"PANORAMA data-items manifest not found: {path}")
        data = pd.read_csv(path)
        required = {"npz_path", "label", "organ_label", "train", "val", "test"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        contract_columns = ["npz_path", "label", "organ_label", "train", "val", "test"]
        if data[contract_columns].isna().any().any():
            raise ValueError(f"{path} contains missing ROI, label, or split values.")
        for split_name in ("train", "val", "test"):
            split_values = data[split_name]
            if not pd.api.types.is_bool_dtype(split_values):
                if not set(split_values.unique()).issubset({0, 1}):
                    raise ValueError(
                        f"{path} column {split_name!r} must contain booleans or 0/1."
                    )
                data[split_name] = split_values.astype(bool)
        self._validate_integer_column(data, "label", path)
        self._validate_integer_column(data, "organ_label", path)
        if not data["organ_label"].equals(data["label"]):
            raise ValueError(f"PANORAMA manifest organ labels must equal class labels: {path}")
        selected = data.loc[data[self.split]].copy()
        if selected.empty:
            raise ValueError(f"PANORAMA manifest has no rows for split {self.split!r}: {path}")
        if not set(selected["label"]).issubset(self.config.valid_labels):
            raise ValueError(f"PANORAMA manifest contains unexpected class indices: {path}")
        if not set(selected["organ_label"]).issubset(self.config.valid_labels):
            raise ValueError(f"PANORAMA manifest contains unexpected ROI labels: {path}")
        return selected

    @staticmethod
    def _validate_integer_column(data, column, path):
        numeric = pd.to_numeric(data[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{path} column {column!r} must contain finite integers.")
        data[column] = numeric.astype(np.int64)
