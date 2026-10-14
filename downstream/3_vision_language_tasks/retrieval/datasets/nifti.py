from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from monai.transforms.compose import Compose
from monai.transforms.croppad.array import CropForeground
from monai.transforms.intensity.array import ScaleIntensityRange
from monai.transforms.spatial.array import Resize
from torch.utils.data import Dataset


TARGET_SIZE = (112, 336, 336)
_REQUIRED_ROW_FIELDS = (
    "image_path",
    "text_en",
    "report_key",
    "body_part",
)

PREPROCESSING = {
    "ctrg": {
        "input_intensity": "pre-windowed 8-bit values",
        "orientation": "native NIfTI axes, permute (x,y,z) to (z,y,x)",
        "spacing": "native voxel grid; configured spacing is not applied",
        "intensity_range": (0.0, 255.0),
        "transforms": (
            "ScaleIntensityRange([0,255] -> [0,1], clip=True)",
            "CropForeground(select_fn=lambda x: x > 0)",
            "Resize(spatial_size=(112,336,336))",
        ),
        "output": "float tensor [1,112,336,336]",
    },
}


def preprocessing_spec(name: str) -> dict[str, Any]:
    try:
        return deepcopy(PREPROCESSING[name])
    except KeyError as exc:
        raise ValueError(
            f"Unknown preprocessing {name!r}; choose one of {sorted(PREPROCESSING)}"
        ) from exc


def _positive_select_fn(value: torch.Tensor) -> torch.Tensor:
    return value > 0


def _transform(intensity_range: tuple[float, float]) -> Compose:
    return Compose(
        [
            ScaleIntensityRange(
                a_min=intensity_range[0],
                a_max=intensity_range[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForeground(select_fn=_positive_select_fn),
            Resize(spatial_size=TARGET_SIZE),
        ]
    )


def _load_ctrg(path: Path) -> torch.Tensor:
    volume = torch.from_numpy(nib.load(str(path)).get_fdata()).float()
    return volume.permute(2, 1, 0).unsqueeze(0)


class NiftiRetrievalDataset(Dataset):
    """Read-only NIfTI dataset for normalized retrieval manifest rows."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], preprocessing: str) -> None:
        if preprocessing not in PREPROCESSING:
            raise ValueError(
                f"Unknown preprocessing {preprocessing!r}; "
                f"choose one of {sorted(PREPROCESSING)}"
            )
        self.preprocessing = preprocessing
        self.rows = []
        for index, source_row in enumerate(rows):
            row = dict(source_row)
            missing = [field for field in _REQUIRED_ROW_FIELDS if not row.get(field)]
            if missing:
                raise ValueError(f"Row {index} is missing required fields: {missing}")
            row["image_path"] = str(row["image_path"])
            if not row.get("pair_id"):
                row["pair_id"] = row["image_path"]
            self.rows.append(row)

        intensity_range = PREPROCESSING[preprocessing]["intensity_range"]
        self.transform = _transform(intensity_range)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = Path(self.rows[index]["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if not path.name.endswith(".nii.gz"):
            raise ValueError(f"Expected a .nii.gz NIfTI path, got {path}")

        volume = _load_ctrg(path)
        volume = self.transform(volume)
        volume = volume.as_tensor() if hasattr(volume, "as_tensor") else volume
        return {"volume": volume.float(), "index": int(index)}
