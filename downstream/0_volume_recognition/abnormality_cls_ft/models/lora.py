"""Minimal LoRA for the open_clip ViT visual tower (no peft dependency).

Only what the pan-cancer classifier needs: low-rank adapters on the MLP
(`c_fc`, `c_proj`) and on the attention (fused qkv `in_proj_weight`, and
`out_proj`) of a `ResidualAttentionBlock`.

Why attention needs its own wrapper: `nn.MultiheadAttention` hands
`in_proj_weight` and `out_proj.weight` to `F.multi_head_attention_forward`
as *raw tensors*, so replacing the `out_proj` submodule (or hooking it) has
no effect at all. `LoRAMultiheadAttention` therefore re-implements the
self-attention path the block actually uses -- self-attention, need_weights
False, attention dropout 0 -- on top of `scaled_dot_product_attention`, and
adds the low-rank terms in activation space (mathematically identical to
adding B@A*scale to the weight, but without materialising a
[3*width, width] delta per block on every forward).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_CHOICES = ("qkv", "out", "fc", "proj")


def _lora_params(r: int, fan_in: int, fan_out: int) -> tuple[nn.Parameter, nn.Parameter]:
    """A (r x fan_in) kaiming-init, B (fan_out x r) zero-init -> delta starts at 0."""
    a = nn.Parameter(torch.empty(r, fan_in))
    nn.init.kaiming_uniform_(a, a=math.sqrt(5))
    b = nn.Parameter(torch.zeros(fan_out, r))
    return a, b


class LoRALinear(nn.Module):
    """`nn.Linear` with a frozen base weight plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A, self.lora_B = _lora_params(self.r, base.in_features, base.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return out + delta * self.scaling

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:.3f}"


class LoRAMultiheadAttention(nn.Module):
    """Self-attention with LoRA on the fused qkv projection and/or `out_proj`.

    Drop-in for `ResidualAttentionBlock.attn`: called as
    `attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)` and expected
    to return a `(output, weights)` tuple.

    Supported: self-attention only (the block always passes the same tensor as
    q/k/v), `need_weights=False`, and `attn_mask` None or broadcastable to
    [B, H, L, S]. Anything else raises rather than silently changing semantics.
    """

    def __init__(self, attn: nn.MultiheadAttention, r: int, alpha: float, dropout: float,
                 qkv: bool = True, out: bool = True) -> None:
        super().__init__()
        if attn.in_proj_weight is None:
            raise RuntimeError("LoRAMultiheadAttention needs a fused in_proj_weight")
        if attn.bias_k is not None or attn.bias_v is not None or attn.add_zero_attn:
            raise RuntimeError("LoRAMultiheadAttention does not support bias_k/bias_v/add_zero_attn")
        self.attn = attn
        for p in self.attn.parameters():
            p.requires_grad = False
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.batch_first = attn.batch_first
        self.attn_dropout = float(attn.dropout)
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        embed_dim = attn.embed_dim
        if qkv:
            self.qkv_lora_A, self.qkv_lora_B = _lora_params(self.r, embed_dim, 3 * embed_dim)
        else:
            self.qkv_lora_A = self.qkv_lora_B = None
        if out:
            self.out_lora_A, self.out_lora_B = _lora_params(self.r, embed_dim, embed_dim)
        else:
            self.out_lora_A = self.out_lora_B = None

    def forward(self, query, key=None, value=None, need_weights: bool = False,
                attn_mask=None, **kwargs):
        if need_weights:
            raise RuntimeError("LoRAMultiheadAttention only supports need_weights=False")
        if (key is not None and key is not query) or (value is not None and value is not query):
            raise RuntimeError("LoRAMultiheadAttention only supports self-attention")

        x = query if self.batch_first else query.transpose(0, 1)
        bsz, seq, embed_dim = x.shape

        qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        if self.qkv_lora_A is not None:
            qkv = qkv + F.linear(
                F.linear(self.lora_dropout(x), self.qkv_lora_A), self.qkv_lora_B
            ) * self.scaling

        q, k, v = qkv.chunk(3, dim=-1)
        shape = (bsz, seq, self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)

        if attn_mask is not None and attn_mask.dim() == 3:
            raise RuntimeError("LoRAMultiheadAttention expects a None or 2D attn_mask")
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        o = o.transpose(1, 2).reshape(bsz, seq, embed_dim)

        out = F.linear(o, self.attn.out_proj.weight, self.attn.out_proj.bias)
        if self.out_lora_A is not None:
            out = out + F.linear(
                F.linear(self.lora_dropout(o), self.out_lora_A), self.out_lora_B
            ) * self.scaling

        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, None

    def extra_repr(self) -> str:
        targets = [n for n, p in (("qkv", self.qkv_lora_A), ("out", self.out_lora_A)) if p is not None]
        return f"r={self.r}, scaling={self.scaling:.3f}, targets={','.join(targets)}"


def parse_targets(spec: str) -> set[str]:
    """'qkv,out,fc,proj' -> {'qkv','out','fc','proj'}."""
    targets = {t.strip() for t in str(spec).replace(",", " ").split() if t.strip()}
    unknown = targets - set(TARGET_CHOICES)
    if unknown:
        raise ValueError(f"unknown --lora-targets {sorted(unknown)}; choose from {TARGET_CHOICES}")
    if not targets:
        raise ValueError("--lora-targets is empty")
    return targets


def parse_blocks(spec: str, total: int) -> list[int]:
    """'all' -> every block; 'last:N' -> the last N; '0,5,31' -> those (0-based)."""
    spec = str(spec).strip()
    if spec in ("", "all"):
        return list(range(total))
    if spec.startswith("last:"):
        n = int(spec.split(":", 1)[1])
        if n <= 0:
            raise ValueError(f"--lora-blocks {spec!r}: N must be > 0")
        return list(range(max(0, total - n), total))
    idx = [int(v) for v in spec.replace(",", " ").split()]
    for i in idx:
        if not 0 <= i < total:
            raise ValueError(f"--lora-blocks index {i} out of range 0..{total - 1}")
    return idx


def inject_lora(visual: nn.Module, blocks: list[int], targets: set[str],
                r: int, alpha: float, dropout: float) -> list[str]:
    """Replace the targeted submodules of `visual.transformer.resblocks` in place.

    Returns the list of injected module paths (for logging).
    """
    resblocks = visual.transformer.resblocks
    injected: list[str] = []
    want_qkv, want_out = "qkv" in targets, "out" in targets
    for i in blocks:
        blk = resblocks[i]
        if want_qkv or want_out:
            blk.attn = LoRAMultiheadAttention(blk.attn, r, alpha, dropout,
                                              qkv=want_qkv, out=want_out)
            if want_qkv:
                injected.append(f"resblocks.{i}.attn.qkv")
            if want_out:
                injected.append(f"resblocks.{i}.attn.out_proj")
        if "fc" in targets:
            blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, r, alpha, dropout)
            injected.append(f"resblocks.{i}.mlp.c_fc")
        if "proj" in targets:
            blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, r, alpha, dropout)
            injected.append(f"resblocks.{i}.mlp.c_proj")
    return injected


def is_lora_param_name(name: str) -> bool:
    """True for the trainable A/B parameters created above (used for param groups)."""
    return "lora_A" in name or "lora_B" in name
