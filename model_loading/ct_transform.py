from os import PathLike
from typing import Callable

import nibabel as nib
import numpy as np
import torch
from monai.transforms import CropForeground, Resize, ScaleIntensityRange
from scipy.ndimage import zoom


def get_transform(
    target_spacing: tuple[float, float, float] = (3.0, 1.0, 1.0),
    target_size: tuple[int, int, int] = (112, 336, 336),
) -> Callable[[str | PathLike[str]], torch.Tensor]:
    """Convert a raw-HU NIfTI to CPU float32 [1,D,H,W] in S/A/L order.

    Spacing is in mm; HU [-1000, 1000] is clipped and mapped to [0, 1],
    then normalized with mean 0.449 and std 0.226.
    """
    spacing = np.asarray(target_spacing, dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("target_spacing must contain three positive finite values")
    size = np.asarray(target_size, dtype=np.float64)
    if (
        size.shape != (3,)
        or not np.isfinite(size).all()
        or (size <= 0).any()
        or (size != np.floor(size)).any()
    ):
        raise ValueError("target_size must contain three positive integers")
    target_size = tuple(int(value) for value in size)

    scale = ScaleIntensityRange(
        a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0,
        clip=True, dtype=np.float32,
    )
    crop = CropForeground(
        select_fn=lambda value: value > 0, channel_indices=None, margin=0,
        allow_smaller=False, return_coords=False, k_divisible=1,
        mode="constant", lazy=False,
    )
    resize = Resize(
        spatial_size=target_size, size_mode="all", mode="area",
        align_corners=None, anti_aliasing=False, dtype=torch.float32, lazy=False,
    )

    def transform(path: str | PathLike[str]) -> torch.Tensor:
        if not isinstance(path, (str, PathLike)):
            raise TypeError("Expected a HU NIfTI path, not an array or tensor")
        image = nib.load(str(path))
        if len(image.shape) != 3 or any(length <= 0 for length in image.shape):
            raise ValueError(f"Expected a nonempty 3D NIfTI, got {image.shape}")
        if image.header.get_xyzt_units()[0] not in ("mm", "unknown"):
            raise ValueError("NIfTI spatial units must be millimeters or unspecified")
        if not np.isfinite(image.affine).all() or np.linalg.matrix_rank(image.affine[:3, :3]) != 3:
            raise ValueError("NIfTI affine must be finite and have three spatial axes")
        orientation = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(image.affine),
            nib.orientations.axcodes2ornt(("L", "A", "S")),
        )
        image = image.as_reoriented(orientation)
        source_spacing = nib.affines.voxel_sizes(image.affine)[[2, 1, 0]]
        if not np.isfinite(source_spacing).all() or (source_spacing <= 0).any():
            raise ValueError("NIfTI spacing must contain three positive finite values")
        volume = image.get_fdata(dtype=np.float32).transpose(2, 1, 0)
        if not np.isfinite(volume).all():
            raise ValueError("NIfTI HU values must be finite")
        volume = zoom(volume, source_spacing / spacing, order=1, mode="constant", cval=0)
        if any(length <= 0 for length in volume.shape):
            raise ValueError("Spacing resampling produced an empty spatial dimension")
        tensor = torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0)
        tensor = scale(tensor)
        if bool(torch.any(tensor > 0)):
            tensor = crop(tensor)
        tensor = resize(tensor)
        tensor = tensor.as_tensor() if hasattr(tensor, "as_tensor") else tensor
        tensor = ((tensor - 0.449) / 0.226).to(dtype=torch.float32).contiguous()
        if tuple(tensor.shape) != (1, *target_size):
            raise ValueError(f"Unexpected retrieval tensor shape: {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError("Retrieval preprocessing produced non-finite values")
        return tensor

    return transform
