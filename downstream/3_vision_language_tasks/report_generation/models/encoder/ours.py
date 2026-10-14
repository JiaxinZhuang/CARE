import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _care_root():
    configured = os.environ.get("CARE_CODE_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if (root / "model_loading" / "configs" / "care_vit_h.json").is_file():
            return root
        raise FileNotFoundError(f"CARE_CODE_ROOT does not contain model_loading: {root}")
    for root in Path(__file__).resolve().parents:
        if (root / "model_loading" / "configs" / "care_vit_h.json").is_file():
            return root
    raise FileNotFoundError(
        "Could not locate the shared CARE model_loading package from this release tree"
    )


CARE_ROOT = _care_root()
MODEL_LOADING_ROOT = CARE_ROOT / "model_loading"
CARE_CONFIG = MODEL_LOADING_ROOT / "configs" / "care_vit_h.json"
if str(CARE_ROOT) not in sys.path:
    sys.path.insert(0, str(CARE_ROOT))

# Build only the shared CARE visual architecture. Calling model_loading.get_model
# here would load care.pth even though every visual tensor is replaced by the
# strictly validated merged report checkpoint.
from model_loading import open_clip
from model_loading.open_clip.model import CLIPVisionCfg, _build_vision_tower


IMAGENET_MEAN = sum((0.485, 0.456, 0.406)) / 3.0
IMAGENET_STD = sum((0.229, 0.224, 0.225)) / 3.0


class _CAREVisualModel(nn.Module):
    """Checkpoint-compatible subset of CustomTextCLIP used by the VLM."""

    def __init__(self):
        super().__init__()
        open_clip.add_model_config(CARE_CONFIG)
        config = open_clip.get_model_config(CARE_CONFIG.stem)
        if config is None:
            raise RuntimeError(f"CARE model config was not registered: {CARE_CONFIG}")
        self.visual = _build_vision_tower(
            config["embed_dim"], CLIPVisionCfg(**config["vision_cfg"])
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))


class OursWrapper(nn.Module):
    """Shared CARE visual architecture with the original report VLM interface."""

    def __init__(
        self,
        input_size=(112, 336, 336),
        hidden_dim=1024,
        feature_mode="tokens_cls",
        token_grid=(2, 4, 4),
        in_channels=1,
        care_checkpoint_path=None,
        care_text_assets_path=None,
    ):
        super().__init__()
        if feature_mode != "tokens_cls":
            raise ValueError("Merged report checkpoints require tokens_cls features")
        if in_channels != 1:
            raise ValueError("Merged report checkpoints require one-channel CARE input")
        if hidden_dim != 1024:
            raise ValueError(f"CARE visual output must be 1024-D, got {hidden_dim}")

        self.input_size = tuple(input_size)
        self.feature_mode = feature_mode
        self.token_grid = tuple(token_grid)
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.model = _CAREVisualModel()

        visual = self.model.visual
        if tuple(visual.image_size) != self.input_size:
            raise ValueError(
                f"VLM input_size {self.input_size} does not match CARE {visual.image_size}"
            )
        if not visual.output_tokens:
            raise ValueError("CARE visual config must enable output_tokens")

        if care_checkpoint_path is not None:
            self.load_care_weights(care_checkpoint_path, care_text_assets_path)

    def load_care_weights(self, checkpoint_path, text_assets_path=None):
        """Strict-load the complete shared CARE checkpoint, then keep its vision tower."""
        from model_loading.models import get_model

        care_model = get_model(
            "cpu",
            checkpoint_path,
            text_assets_path=text_assets_path,
        )
        expected = self.model.visual.state_dict()
        loaded = care_model.visual.state_dict()
        if set(expected) != set(loaded):
            raise RuntimeError(
                "Strict CARE load succeeded, but visual architecture keys differ: "
                f"missing={sorted(set(expected) - set(loaded))[:10]}, "
                f"unexpected={sorted(set(loaded) - set(expected))[:10]}"
            )
        mismatched = {
            name: (tuple(expected[name].shape), tuple(loaded[name].shape))
            for name in expected
            if expected[name].shape != loaded[name].shape
        }
        if mismatched:
            raise RuntimeError(f"CARE visual tensor shapes differ: {mismatched}")

        self.model.visual = care_model.visual
        self.model.logit_scale = care_model.logit_scale
        del care_model

    def forward(self, image, mask=None, device="cuda"):
        if image is None:
            raise ValueError("Image input is required")
        if image.ndim != 5:
            raise ValueError(f"Expected image [B,C,D,H,W], got {tuple(image.shape)}")

        image = F.interpolate(
            image,
            size=self.input_size,
            mode="trilinear",
            align_corners=False,
        )
        if image.shape[1] == 3:
            image = image.mean(dim=1, keepdim=True)
        elif image.shape[1] != 1:
            raise ValueError(f"Expected one or three image channels, got {image.shape[1]}")
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        pooled, tokens = self.model.visual(image)
        proj = getattr(self.model.visual, "proj", None)
        if proj is not None and tokens.shape[-1] == proj.shape[0]:
            tokens = tokens @ proj

        batch, length, channels = tokens.shape
        grid = tuple(self.model.visual.grid_size)
        if math.prod(grid) != length:
            raise RuntimeError(f"CARE token grid {grid} does not match length {length}")
        tokens = tokens.transpose(1, 2).reshape(batch, channels, *grid)
        tokens = F.adaptive_avg_pool3d(tokens, self.token_grid)
        tokens = tokens.flatten(2).transpose(1, 2)
        image_features = torch.cat([pooled.unsqueeze(1), tokens], dim=1)
        if image_features.shape[1] != 33:
            raise RuntimeError(
                f"Expected CLS + 32 CARE tokens, got {image_features.shape[1]}"
            )
        return image_features

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def device(self):
        return next(self.model.parameters()).device


def create_ours(input_size=(112, 336, 336), **kwargs):
    return OursWrapper(input_size=input_size, **kwargs)
