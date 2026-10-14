from pathlib import Path
import sys
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize


CARE_ROOT = Path(__file__).resolve().parents[4]
if str(CARE_ROOT) not in sys.path:
    sys.path.insert(0, str(CARE_ROOT))

from model_loading.models import get_model, get_tokenizer


class OursWrapper(nn.Module):
    """Frozen CARE dual-encoder adapter for retrieval feature extraction."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        text_assets_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.model = get_model(
            device,
            str(checkpoint_path),
            text_assets_path=text_assets_path,
        )
        self.tokenizer = get_tokenizer(text_assets_path=text_assets_path)
        self.normalizer = Normalize(
            torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(),
            torch.as_tensor(IMAGENET_DEFAULT_STD).mean(),
        )
        self.model.requires_grad_(False)
        super().train(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "OursWrapper":
        """Keep the extraction wrapper and backbone in evaluation mode."""
        super().train(False)
        self.model.eval()
        return self

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5:
            raise ValueError(f"Expected image [B,C,D,H,W], got {tuple(image.shape)}")
        if image.shape[1] == 3:
            image = image.mean(dim=1, keepdim=True)
        elif image.shape[1] != 1:
            raise ValueError(f"Expected one or three image channels, got {image.shape[1]}")

        # Preserve the original retrieval wrapper order: normalize, then resize.
        image = self.normalizer(image)
        return F.interpolate(
            image,
            size=(112, 336, 336),
            mode="trilinear",
            align_corners=False,
        )

    @torch.inference_mode()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        visual_output = self.model.visual(self._prepare_image(image))
        if not isinstance(visual_output, tuple) or len(visual_output) != 2:
            raise RuntimeError("CARE visual encoder must return (pooled, tokens)")
        pooled, _ = visual_output
        return F.normalize(pooled, dim=-1)

    @torch.inference_mode()
    def encode_text(self, text: str | Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(text).to(next(self.model.parameters()).device)
        return self.model.encode_text(tokens, normalize=True)

    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor,
        text: str | Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(image), self.encode_text(text)
