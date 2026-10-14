"""CARE visual adapter preserving the paper ROI-classification input path."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize


CARE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CARE_CHECKPOINT = CARE_ROOT / "ckpts" / "care.pth"


def _get_shared_loader():
    root = str(CARE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from model_loading.models import get_model

    return get_model


class OursWrapper(nn.Module):
    """Expose the L2-normalized pooled CARE visual feature as ``B x F``."""

    def __init__(self, pretrained_path=DEFAULT_CARE_CHECKPOINT, text_assets_path=None):
        super().__init__()
        get_model = _get_shared_loader()
        care_model = get_model(
            "cpu",
            str(Path(pretrained_path).expanduser()),
            text_assets_path=text_assets_path,
        )
        self.model = care_model.visual
        self.output_dim = self.model.output_dim
        del care_model
        self.normalizer = Normalize(
            torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(),
            torch.as_tensor(IMAGENET_DEFAULT_STD).mean(),
        )

    def forward(self, image=None, **kwargs):
        if image is None:
            raise ValueError("Image input is required.")
        if image.ndim != 5:
            raise ValueError(f"Expected B,C,D,H,W input, got shape {tuple(image.shape)}.")

        image = F.interpolate(
            image,
            size=(112, 336, 336),
            mode="trilinear",
            align_corners=False,
        )
        if image.shape[1] == 3:
            image = image.mean(dim=1, keepdim=True)
        elif image.shape[1] != 1:
            raise ValueError(f"Expected one or three input channels, got {image.shape[1]}.")
        image = self.normalizer(image)

        visual_output = self.model(image)
        pooled = visual_output[0] if isinstance(visual_output, tuple) else visual_output
        if pooled.ndim != 2 or pooled.shape[-1] != self.output_dim:
            raise ValueError(f"Unexpected CARE pooled feature shape: {tuple(pooled.shape)}.")
        return F.normalize(pooled, dim=-1)


def create_ours(**kwargs):
    return OursWrapper(**kwargs)
