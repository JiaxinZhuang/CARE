"""KiPA22 CARE-UNETR model implementation.

The MONAI UNETR architecture, LoRA namespace, and tensor shapes support strict
loading of the released KiPA22 checkpoint. CARE transfer populates the mapped
visual parameters. Parameters outside that mapping, including the MONAI patch
bias and LoRA-A, use the initializers defined by the runtime libraries. Fresh
initialization is not verified to be numerically identical to prior training
runs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNETR


PATCH_SIZE = (16, 16, 16)
CARE_PATCH_SIZE = (7, 21, 21)
HIDDEN_SIZE = 1280
MLP_DIM = 5120
NUM_HEADS = 16
NUM_BLOCKS = 12
CARE_ROOT = Path(__file__).resolve().parents[4]


def _get_shared_care_loader():
    loader_path = CARE_ROOT / "model_loading" / "models.py"
    if not loader_path.is_file():
        raise ImportError(f"shared CARE model loader not found: {loader_path}")
    root = str(CARE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from model_loading.models import get_model

    return get_model


def _copy_tensor(source: torch.Tensor, target: torch.Tensor, name: str) -> None:
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(
            f"{name} shape mismatch: source {tuple(source.shape)}, "
            f"target {tuple(target.shape)}"
        )
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))


def _copy_linear(source: nn.Linear, target: nn.Linear, name: str) -> None:
    _copy_tensor(source.weight, target.weight, f"{name}.weight")
    if (source.bias is None) != (target.bias is None):
        raise ValueError(f"{name} bias mismatch")
    if source.bias is not None:
        _copy_tensor(source.bias, target.bias, f"{name}.bias")


def _copy_attention_projection(source_attention, target_attention) -> None:
    """Copy an attention output projection across OpenCLIP/MONAI names."""
    source = getattr(source_attention, "out_proj", None)
    if source is None:
        source = getattr(source_attention, "proj", None)
    target = getattr(target_attention, "out_proj", None)
    if target is None:
        target = getattr(target_attention, "proj", None)
    if not isinstance(source, nn.Linear) or not isinstance(target, nn.Linear):
        raise AttributeError("attention output projection is missing")
    _copy_linear(source, target, "attention.out_proj")


def _resize_3d_positional_embedding(source, target, target_grid) -> None:
    """Remove the CARE class token and resize its cubic 3D patch grid."""
    if source.ndim == 2:
        source = source.unsqueeze(0)
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("positional embeddings must have shape [1, tokens, width]")
    if source.shape[0] != 1 or target.shape[0] != 1:
        raise ValueError("only one positional embedding table is supported")
    if source.shape[-1] != target.shape[-1]:
        raise ValueError("positional embedding widths do not match")

    patch_tokens = source.shape[1] - 1
    source_side = round(patch_tokens ** (1.0 / 3.0))
    if source_side ** 3 != patch_tokens:
        raise ValueError(
            f"cannot infer a cubic grid after removing one class token from "
            f"{source.shape[1]} tokens"
        )

    target_grid = tuple(int(value) for value in target_grid)
    if len(target_grid) != 3 or any(value <= 0 for value in target_grid):
        raise ValueError(f"invalid target grid: {target_grid}")
    if math.prod(target_grid) != target.shape[1]:
        raise ValueError(
            f"target grid {target_grid} does not match {target.shape[1]} tokens"
        )

    patches = source[:, 1:].reshape(
        1, source_side, source_side, source_side, source.shape[-1]
    )
    patches = patches.permute(0, 4, 1, 2, 3).float()
    resized = F.interpolate(
        patches, size=target_grid, mode="trilinear", align_corners=False
    )
    resized = resized.permute(0, 2, 3, 4, 1).reshape_as(target)
    _copy_tensor(resized, target, "position_embeddings")


class LoRALinear(nn.Module):
    """Frozen linear layer plus the checkpoint-matched low-rank update."""

    def __init__(self, base_layer, rank=8, alpha=16.0, dropout=0.05):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRA can only wrap nn.Linear")
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

        self.base_layer = base_layer
        self.scaling = float(alpha) / int(rank)
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base_layer.in_features, int(rank), bias=False)
        self.lora_b = nn.Linear(int(rank), base_layer.out_features, bias=False)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs):
        update = self.lora_b(self.lora_a(self.lora_dropout(inputs)))
        return self.base_layer(inputs) + update * self.scaling


class OursUNETR(UNETR):
    """Twelve-layer MONAI UNETR-H used by the KiPA22 task."""

    def __init__(self, in_channels, out_channels, img_size):
        img_size = tuple(int(value) for value in img_size)
        if len(img_size) != 3 or any(size % 16 for size in img_size):
            raise ValueError("each ROI dimension must be divisible by patch size 16")
        super().__init__(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            img_size=img_size,
            feature_size=16,
            hidden_size=HIDDEN_SIZE,
            mlp_dim=MLP_DIM,
            num_heads=NUM_HEADS,
            proj_type="conv",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
            qkv_bias=False,
        )
        self.target_img_size = img_size
        if len(self.vit.blocks) != NUM_BLOCKS:
            raise RuntimeError(f"expected {NUM_BLOCKS} MONAI blocks")

    def transfer_care_visual(self, visual) -> None:
        """Transfer CARE patch/position weights and visual blocks 0 through 11."""
        source_patch = visual.conv1
        target_patch = self.vit.patch_embedding.patch_embeddings
        if tuple(source_patch.kernel_size) != CARE_PATCH_SIZE:
            raise ValueError(f"expected CARE patch kernel {CARE_PATCH_SIZE}")
        if tuple(target_patch.kernel_size) != PATCH_SIZE:
            raise ValueError(f"expected UNETR patch kernel {PATCH_SIZE}")
        if source_patch.weight.shape[:2] != target_patch.weight.shape[:2]:
            raise ValueError("CARE and UNETR patch channels do not match")

        resized_kernel = F.interpolate(
            source_patch.weight.detach().float(),
            size=PATCH_SIZE,
            mode="trilinear",
            align_corners=False,
        )
        _copy_tensor(resized_kernel, target_patch.weight, "patch_embeddings.weight")
        # CARE conv1 has no bias, so CARE transfer does not populate the patch bias.

        source_blocks = visual.transformer.resblocks
        if len(source_blocks) < NUM_BLOCKS:
            raise ValueError(
                f"CARE visual tower has {len(source_blocks)} blocks, need {NUM_BLOCKS}"
            )
        for index, target_block in enumerate(self.vit.blocks):
            source_block = source_blocks[index]
            _copy_linear(source_block.ln_1, target_block.norm1, f"block{index}.norm1")

            source_attention = source_block.attn
            target_attention = target_block.attn
            if not hasattr(source_attention, "in_proj_weight"):
                raise AttributeError(f"CARE block {index} has no fused QKV weight")
            _copy_tensor(
                source_attention.in_proj_weight,
                target_attention.qkv.weight,
                f"block{index}.attn.qkv.weight",
            )
            if target_attention.qkv.bias is not None:
                if source_attention.in_proj_bias is None:
                    raise ValueError(f"CARE block {index} has no QKV bias")
                _copy_tensor(
                    source_attention.in_proj_bias,
                    target_attention.qkv.bias,
                    f"block{index}.attn.qkv.bias",
                )
            _copy_attention_projection(source_attention, target_attention)

            _copy_linear(source_block.ln_2, target_block.norm2, f"block{index}.norm2")
            _copy_linear(
                source_block.mlp.c_fc,
                target_block.mlp.linear1,
                f"block{index}.mlp.linear1",
            )
            _copy_linear(
                source_block.mlp.c_proj,
                target_block.mlp.linear2,
                f"block{index}.mlp.linear2",
            )

        source_position = visual.positional_embedding
        if tuple(source_position.shape) != (1 + 16 ** 3, HIDDEN_SIZE):
            raise ValueError(
                f"expected CARE positional table {(1 + 16 ** 3, HIDDEN_SIZE)}, "
                f"got {tuple(source_position.shape)}"
            )
        target_grid = tuple(size // 16 for size in self.target_img_size)
        _resize_3d_positional_embedding(
            source_position,
            self.vit.patch_embedding.position_embeddings,
            target_grid,
        )


def configure_lora_encoder(model, rank=8, alpha=16.0, dropout=0.05):
    """Apply the KiPA22 trainability policy and LoRA namespace."""
    if any(isinstance(block.attn.qkv, LoRALinear) for block in model.vit.blocks):
        raise RuntimeError("LoRA is already configured")

    for name, parameter in model.named_parameters():
        parameter.requires_grad = not name.startswith("vit.")
    patch_embedding = model.vit.patch_embedding
    for parameter in patch_embedding.patch_embeddings.parameters():
        parameter.requires_grad = True
    patch_embedding.position_embeddings.requires_grad = True
    for module in model.vit.modules():
        if isinstance(module, nn.LayerNorm):
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad = True

    for block in model.vit.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha, dropout)
        block.attn.out_proj = LoRALinear(
            block.attn.out_proj, rank, alpha, dropout
        )
    return model


def trainable_parameter_summary(model):
    """Return disjoint trainable counts matching the frozen training log."""
    counts = dict(lora=0, encoder_adaptation=0, patch_embedding=0, decoder=0)
    total = 0
    for name, parameter in model.named_parameters():
        name = name.removeprefix("module.")
        total += parameter.numel()
        if not parameter.requires_grad:
            continue
        if ".lora_" in name:
            counts["lora"] += parameter.numel()
        elif name.startswith("vit.patch_embedding.patch_embeddings."):
            counts["patch_embedding"] += parameter.numel()
        elif name.startswith("vit."):
            counts["encoder_adaptation"] += parameter.numel()
        else:
            counts["decoder"] += parameter.numel()
    counts["trainable"] = sum(counts.values())
    counts["total"] = total
    return counts


def build_optimizer_parameter_groups(model, args):
    """Build decoder/adaptation/patch groups with the configured learning rates."""
    grouped = dict(decoder=[], encoder_adaptation=[], patch_embedding=[])
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized = name.removeprefix("module.")
        if normalized.startswith("vit.patch_embedding.patch_embeddings."):
            group = "patch_embedding"
        elif normalized.startswith("vit."):
            group = "encoder_adaptation"
        else:
            group = "decoder"
        if id(parameter) in seen:
            raise RuntimeError(f"duplicate optimizer parameter: {normalized}")
        seen.add(id(parameter))
        grouped[group].append(parameter)

    missing = [name for name, parameters in grouped.items() if not parameters]
    if missing:
        raise RuntimeError(f"empty optimizer parameter groups: {missing}")
    learning_rates = dict(
        decoder=float(args.optim_lr),
        encoder_adaptation=float(args.encoder_lr),
        patch_embedding=float(args.patch_embed_lr),
    )
    return [
        dict(name=name, params=parameters, lr=learning_rates[name])
        for name, parameters in grouped.items()
    ]


def get_model(args, load_pretrained=True):
    """Construct fixed UNETR-H plus LoRA, optionally initialized from CARE."""
    model = OursUNETR(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        img_size=(args.roi_x, args.roi_y, args.roi_z),
    )
    if load_pretrained:
        pretrained_path = getattr(args, "pretrained_path", None)
        if not pretrained_path:
            raise ValueError("load_pretrained=True requires args.pretrained_path")
        checkpoint_path = Path(pretrained_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CARE checkpoint not found: {checkpoint_path}")
        get_care_model = _get_shared_care_loader()
        joint_model = get_care_model("cpu", checkpoint_path)
        try:
            visual = getattr(joint_model, "visual", None)
            if visual is None:
                raise AttributeError("shared CARE model has no visual tower")
            model.transfer_care_visual(visual)
        finally:
            del joint_model

    configure_lora_encoder(
        model,
        rank=getattr(args, "lora_rank", 8),
        alpha=getattr(args, "lora_alpha", 16.0),
        dropout=getattr(args, "lora_dropout", 0.05),
    )
    return model
