"""CT-RATE volumes for frozen CARE linear-probe feature extraction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


ALL_CTRATE_CLASSNAMES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

DEFAULT_WINDOW = (-1150.0, 350.0)
DEFAULT_IMAGE_SIZE = (112, 336, 336)


def volume_name_to_path(root: str, volume_name: str) -> str:
    recon_base = volume_name.rsplit(".", 2)[0]
    return os.path.join(
        root,
        recon_base.rsplit("_", 2)[0],
        recon_base.rsplit("_", 1)[0],
        recon_base + ".pt",
    )


def volume_name_to_group_id(volume_name: str) -> str:
    recon_base = volume_name.rsplit(".", 2)[0]
    return recon_base.rsplit("_", 2)[0]


class DataSet(Dataset):
    """Load explicit CT-RATE labels and preprocessed ``.pt`` HU volumes."""

    def __init__(
        self,
        task: str = "ctrate",
        split: str = "validation",
        data_root: str = "",
        label_path: str = "",
        window_min: float = DEFAULT_WINDOW[0],
        window_max: float = DEFAULT_WINDOW[1],
        eval_image_size: Tuple[int, int, int] = DEFAULT_IMAGE_SIZE,
        transform_mode: str = "crop",
        **kwargs,
    ):
        del kwargs
        if task != "ctrate":
            raise ValueError(f"Unsupported CT-RATE task: {task}")
        if split == "valid":
            split = "validation"
        if split not in {"train", "validation"}:
            raise ValueError(f"Unsupported CT-RATE split: {split}")
        if not data_root or not label_path:
            raise ValueError("CT-RATE requires explicit data_root and label_path")
        if window_min >= window_max:
            raise ValueError("window_min must be smaller than window_max")
        if transform_mode not in {"crop", "z_raw"}:
            raise ValueError(f"Unsupported CT-RATE transform: {transform_mode}")

        self.split = split
        self.data_root = str(Path(data_root).expanduser().resolve())
        self.label_path = Path(label_path).expanduser().resolve()
        self.window_min = float(window_min)
        self.window_max = float(window_max)
        self.eval_image_size = tuple(int(value) for value in eval_image_size)
        self.transform_mode = transform_mode
        self.classnames = list(ALL_CTRATE_CLASSNAMES)

        if not self.label_path.is_file():
            raise FileNotFoundError(f"CT-RATE labels not found: {self.label_path}")
        if not Path(self.data_root).is_dir():
            raise FileNotFoundError(f"CT-RATE volume root not found: {self.data_root}")

        frame = pd.read_csv(self.label_path)
        required = ["VolumeName", *self.classnames]
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise ValueError(f"CT-RATE label columns missing: {missing}")

        self.samples = []
        for row in frame[required].itertuples(index=False, name=None):
            volume_name = str(row[0])
            labels = [int(value) for value in row[1:]]
            self.samples.append(
                {
                    "case_id": volume_name.rsplit(".", 2)[0],
                    "group_id": volume_name_to_group_id(volume_name),
                    "image_path": volume_name_to_path(self.data_root, volume_name),
                    "labels": labels,
                }
            )

        self.linear_probe_preprocessing = {
            "window": [self.window_min, self.window_max],
            "size": list(self.eval_image_size),
            "transform_mode": self.transform_mode,
            "clamp": [0.0, 1.0],
        }

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_transform(self, image: torch.Tensor) -> torch.Tensor:
        if self.transform_mode == "z_raw":
            _, _, height, width = image.shape
            height_size, width_size = self.eval_image_size[1:]
            pad_height = max(height_size - height, 0)
            pad_width = max(width_size - width, 0)
            image = F.pad(
                image.unsqueeze(0),
                (
                    pad_width // 2,
                    pad_width - pad_width // 2,
                    pad_height // 2,
                    pad_height - pad_height // 2,
                    0,
                    0,
                ),
                value=0,
            ).squeeze(0)
            _, _, height, width = image.shape
            start_height = (height - height_size) // 2
            start_width = (width - width_size) // 2
            return image[
                :,
                :,
                start_height : start_height + height_size,
                start_width : start_width + width_size,
            ]

        depth_size, height_size, width_size = self.eval_image_size
        pad_depth = max(depth_size - image.shape[1], 0)
        pad_height = max(height_size - image.shape[2], 0)
        pad_width = max(width_size - image.shape[3], 0)
        image = F.pad(
            image.unsqueeze(0),
            (
                pad_width // 2,
                pad_width - pad_width // 2,
                pad_height // 2,
                pad_height - pad_height // 2,
                pad_depth // 2,
                pad_depth - pad_depth // 2,
            ),
            value=0,
        ).squeeze(0)
        _, depth, height, width = image.shape
        start_depth = (depth - depth_size) // 2
        start_height = (height - height_size) // 2
        start_width = (width - width_size) // 2
        return image[
            :,
            start_depth : start_depth + depth_size,
            start_height : start_height + height_size,
            start_width : start_width + width_size,
        ]

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = torch.load(sample["image_path"], weights_only=True)
        image = (image - self.window_min) / (self.window_max - self.window_min)
        image = image.clamp(0.0, 1.0).unsqueeze(0).float()
        image = self._apply_transform(image)
        return {
            "volume": image,
            "labels": torch.tensor(sample["labels"], dtype=torch.float32),
            "image_path": sample["image_path"],
            "case_id": sample["case_id"],
            "group_id": sample["group_id"],
        }

    def get_classnames(self) -> List[str]:
        return list(self.classnames)
