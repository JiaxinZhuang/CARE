"""Generic manifest-backed NIfTI dataset for CARE fine-tuning."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


REQUIRED_COLUMNS = ("case_id", "group_id", "volume_path")


class CTDataset(Dataset):
    """Load raw HU NIfTI volumes listed in an explicit CSV manifest."""

    _IMAGENET_MEAN = 0.449
    _IMAGENET_STD = 0.226

    def __init__(
        self,
        manifest_path: str | Path,
        label_columns: list[str],
        windows: list[list[float]] | list[tuple[float, float]],
        input_size: tuple[int, int, int] = (112, 336, 336),
        *,
        require_labels: bool = True,
        augment: bool = False,
        resize_mode: str = "aspect",
        normalization: str = "pretrained",
        **_: Any,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {self.manifest_path}")
        self.label_columns = list(label_columns)
        if not self.label_columns or len(set(self.label_columns)) != len(self.label_columns):
            raise ValueError("labels must be a non-empty ordered list without duplicates")
        self.CT_WINDOWS = [tuple(float(value) for value in window) for window in windows]
        if len(self.CT_WINDOWS) != 3 or any(len(window) != 2 or window[0] <= 0
                                            for window in self.CT_WINDOWS):
            raise ValueError("windows must contain exactly three positive-width [width, level] pairs")
        self.input_size = tuple(int(value) for value in input_size)
        if self.input_size != (112, 336, 336):
            raise ValueError("CARE ViT-H fine-tuning requires input_size [112, 336, 336]")
        if resize_mode not in {"aspect", "stretch", "physical"}:
            raise ValueError(f"Unknown resize_mode: {resize_mode!r}")
        if normalization not in {"pretrained", "unit"}:
            raise ValueError(f"Unknown normalization: {normalization!r}")
        self.resize_mode = resize_mode
        self.normalization = normalization
        self.augment = bool(augment)

        frame = pd.read_csv(
            self.manifest_path,
            dtype={column: str for column in REQUIRED_COLUMNS},
            keep_default_na=False,
        )
        missing = set(REQUIRED_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")
        for column in REQUIRED_COLUMNS:
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"Manifest column {column!r} contains empty values")
        if frame["case_id"].astype(str).duplicated().any():
            raise ValueError("Manifest case_id values must be unique")

        present = [column in frame.columns for column in self.label_columns]
        if any(present) and not all(present):
            absent = [column for column, exists in zip(self.label_columns, present) if not exists]
            raise ValueError(f"Manifest has only a subset of configured labels; missing: {absent}")
        self.has_labels = all(present)
        if require_labels and not self.has_labels:
            raise ValueError(f"Manifest must contain ordered label columns: {self.label_columns}")
        if self.has_labels:
            values = frame[self.label_columns].apply(pd.to_numeric, errors="coerce")
            if values.isna().any().any() or not values.isin((0, 1)).all().all():
                raise ValueError("All label values must be binary 0/1")
            frame[self.label_columns] = values.astype(np.int8)

        base = self.manifest_path.parent
        resolved = []
        for value in frame["volume_path"].astype(str):
            path = Path(value).expanduser()
            path = path.resolve() if path.is_absolute() else (base / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"NIfTI volume does not exist: {path}")
            if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
                raise ValueError(f"volume_path must reference a .nii or .nii.gz file: {path}")
            resolved.append(path)
        frame = frame.copy()
        frame["case_id"] = frame["case_id"].astype(str)
        frame["group_id"] = frame["group_id"].astype(str)
        frame["_resolved_volume_path"] = resolved
        self.df = frame.reset_index(drop=True)

    @property
    def num_classes(self) -> int:
        return len(self.label_columns)

    @property
    def cls_names(self) -> list[str]:
        return list(self.label_columns)

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def ct_window(img_hu: np.ndarray, width: float, level: float) -> np.ndarray:
        """Clip HU to [level-width/2, level+width/2] and scale to [0, 1]."""
        lo = level - width / 2.0
        hi = level + width / 2.0
        return np.clip((img_hu.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)

    def make_multi_window_image(self, img_hu: np.ndarray) -> np.ndarray:
        """Apply the configured CT windows and return [3, D, H, W]."""
        out = np.empty((len(self.CT_WINDOWS), *img_hu.shape), dtype=np.float32)
        src = img_hu if img_hu.dtype == np.float32 else img_hu.astype(np.float32)
        for index, (width, level) in enumerate(self.CT_WINDOWS):
            lo = level - width / 2.0
            hi = level + width / 2.0
            np.subtract(src, lo, out=out[index])
            np.divide(out[index], hi - lo, out=out[index])
            np.clip(out[index], 0.0, 1.0, out=out[index])
        return out

    def _make_multi_window_image_for_row(self, img_hu: np.ndarray, row: Any) -> np.ndarray:
        return self.make_multi_window_image(img_hu)

    @staticmethod
    def _load_nifti_hu_sitk(nifti_path: Path) -> np.ndarray:
        """Load NIfTI with SimpleITK and return float32 HU array shaped [z, y, x]."""
        try:
            import SimpleITK as sitk

            img = sitk.ReadImage(str(nifti_path))
            return sitk.GetArrayFromImage(img).astype(np.float32)
        except ImportError:
            import nibabel as nib

            return np.asanyarray(nib.load(nifti_path).dataobj).transpose(2, 1, 0).astype(np.float32)

    @staticmethod
    def _spacing_zyx(nifti_path: Path) -> tuple[float, float, float]:
        try:
            import SimpleITK as sitk

            spacing_xyz = sitk.ReadImage(str(nifti_path)).GetSpacing()
        except ImportError:
            import nibabel as nib

            spacing_xyz = nib.load(nifti_path).header.get_zooms()[:3]
        spacing = tuple(float(value) for value in reversed(spacing_xyz))
        if len(spacing) != 3 or not np.isfinite(spacing).all() or any(value <= 0 for value in spacing):
            raise ValueError(f"NIfTI spacing must contain three finite positive values: {nifti_path}")
        return spacing

    @staticmethod
    def _augment_multiwindow(image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            image = image.flip(-1)
        body = (image > 0).any(dim=0, keepdim=True).to(image.dtype)
        jitter = (torch.rand(image.shape[0], 1, 1, 1, device=image.device) - 0.5) * 0.04
        image = (image + jitter * body).clamp(0.0, 1.0)
        return image + torch.randn_like(image) * 0.05 * body

    @staticmethod
    def _resize_with_pad(
        tensor: torch.Tensor,
        target_size: tuple[int, int, int],
        spacing: tuple[float, float, float] | None = None,
    ) -> torch.Tensor:
        _, d, h, w = tensor.shape
        D, H, W = target_size
        new_d, new_h, new_w = CTDataset.fitted_shape((d, h, w), target_size, spacing)
        resized = F.interpolate(
            tensor.unsqueeze(0), size=(new_d, new_h, new_w),
            mode="trilinear", align_corners=False,
        ).squeeze(0)
        pad_d, pad_h, pad_w = D - new_d, H - new_h, W - new_w
        return F.pad(resized, (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_d // 2, pad_d - pad_d // 2,
        ))

    @staticmethod
    def fitted_shape(shape, target_size, spacing=None):
        units = shape if spacing is None else [n * s / r for n, s, r in zip(shape, spacing, (3., 1., 1.))]
        scale = min(t / n for t, n in zip(target_size, units))
        return tuple(min(t, max(1, round(n * scale))) for t, n in zip(target_size, units))

    def _resize_to_input(self, tensor: torch.Tensor, row=None) -> torch.Tensor:
        if self.resize_mode == "stretch":
            return F.interpolate(
                tensor.unsqueeze(0), size=self.input_size,
                mode="trilinear", align_corners=False,
            ).squeeze(0).contiguous()
        spacing = None
        if self.resize_mode == "physical":
            if row is None:
                raise ValueError("physical resize needs the volume spacing")
            spacing = tuple(float(row[f"spacing_{axis}"]) for axis in "zyx")
        return self._resize_with_pad(tensor, self.input_size, spacing).contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.df.iloc[index]
        path = Path(source["_resolved_volume_path"])
        hu_zyx = self._load_nifti_hu_sitk(path)
        if hu_zyx.ndim != 3 or not np.isfinite(hu_zyx).all():
            raise ValueError(f"Expected a finite 3D HU volume: {path}")
        row = source.to_dict()
        row["image_path"] = str(path)
        row.update({f"spacing_{axis}": value
                    for axis, value in zip("zyx", self._spacing_zyx(path))})
        image = torch.from_numpy(self._make_multi_window_image_for_row(hu_zyx, row))
        image = self._resize_to_input(image, row)
        if self.augment:
            image = self._augment_multiwindow(image)
        if self.normalization == "pretrained":
            image = (image - self._IMAGENET_MEAN) / self._IMAGENET_STD
        label = (torch.tensor([int(source[column]) for column in self.label_columns],
                              dtype=torch.float32)
                 if self.has_labels else torch.full((len(self.label_columns),), float("nan")))
        return {
            "image": image,
            "label": label,
            "has_labels": self.has_labels,
            "case_id": str(source["case_id"]),
            "group_id": str(source["group_id"]),
        }
