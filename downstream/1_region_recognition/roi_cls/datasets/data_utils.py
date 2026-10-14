"""Raw ROI preprocessing primitives retained for cache provenance."""

from typing import Dict, List

import numpy as np
from monai.transforms import SpatialCropd
from monai.transforms.transform import MapTransform


class OrganCropTransform(MapTransform):
    """Crop image and mask to the selected organ-label bounding box."""

    def __init__(self, keys: List[str], margin: int = 5, min_size: int = 8):
        super().__init__(keys)
        self.margin = margin
        self.min_size = min_size

    def __call__(self, data: Dict) -> Dict:
        result = dict(data)
        mask = result["mask"]
        organ_label = result["organ_label"]
        organ_mask = mask == organ_label
        if not organ_mask.any():
            raise ValueError(f"Organ label {organ_label} not found in mask")

        coordinates = np.where(organ_mask)
        start = [max(0, coordinate.min() - self.margin) for coordinate in coordinates[1:]]
        end = [
            min(size, coordinate.max() + self.margin + 1)
            for size, coordinate in zip(mask.shape[1:], coordinates[1:])
        ]
        for axis in range(len(start)):
            if end[axis] - start[axis] < self.min_size:
                center = (start[axis] + end[axis]) // 2
                start[axis] = max(0, center - self.min_size // 2)
                end[axis] = min(mask.shape[axis + 1], start[axis] + self.min_size)

        return SpatialCropd(keys=self.keys, roi_start=start, roi_end=end)(result)
