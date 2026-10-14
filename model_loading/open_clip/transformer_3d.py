"""
Author: Jx Zhuang

3D Vision Transformer implementation for OpenCLIP

Extends the original VisionTransformer to support 3D data with different patch sizes in each dimension.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union, Callable, Dict, Any
from .transformer import LayerNorm

# ============================================================================
# STANDARD 3D POSITION EMBEDDING IMPLEMENTATION (PROVEN APPROACH)
# ============================================================================

def get_3d_sincos_pos_embed_unequal_division(embed_dim, grid):
    """
    Use unequal division for dimensions that don't divide evenly by 3.
    For 1024: use 342, 342, 340 (or other combinations)
    """
    # Calculate dimension splits
    base_dim = embed_dim // 3
    remainder = embed_dim % 3

    #! Hardcode
    if embed_dim == 1024:
        dim_d = 340
        dim_h = 342
        dim_w = 342
    elif embed_dim == 1280:
        dim_d = 424
        dim_h = 428
        dim_w = 428
    else:
        raise ValueError(f"Unsupported embed_dim: {embed_dim}")

    # Distribute remainder: give extra dimensions to depth and height
    # dim_d = base_dim + (1 if remainder >= 1 else 0)  # 342 for 1024
    # dim_h = base_dim + (1 if remainder >= 2 else 0)  # 342 for 1024
    # dim_w = base_dim                                 # 340 for 1024

    # Generate embeddings for each dimension
    emb_d = get_1d_sincos_pos_embed_from_grid_standard(dim_d, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid_standard(dim_h, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid_standard(dim_w, grid[2])

    # Concatenate in [width, height, depth] order
    emb = np.concatenate([emb_w, emb_h, emb_d], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid_standard(embed_dim, pos):
    """
    Standard 1D sinusoidal position embedding
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # Standard frequency calculation

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_3d_sincos_pos_embed_from_grid_standard(embed_dim, grid):
    """
    grid: 3D grid of shape [3, 1, D, H, W]
    return: position embeddings of shape [D*H*W, embed_dim]
    """
    assert embed_dim % 3 == 0

    # use one third of dimensions to encode each axis
    emb_d = get_1d_sincos_pos_embed_from_grid_standard(embed_dim // 3, grid[0])  # (D*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid_standard(embed_dim // 3, grid[1])  # (D*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid_standard(embed_dim // 3, grid[2])  # (D*H*W, D/3)

    # Concatenate in [width, height, depth] order for better intuition
    emb = np.concatenate([emb_w, emb_h, emb_d], axis=1)  # (D*H*W, D)
    return emb


def get_3d_sincos_pos_embed_standard(embed_dim, grid_sizes, cls_token=False, flatten=True):
    """
    Standard implementation following established practices
    grid_sizes: sequence of the grid depth, height, and width [D, H, W]
    return:
    pos_embed: [D*H*W, embed_dim] or [1+D*H*W, embed_dim] (w/ or w/o cls_token)
    """
    grid_d = np.arange(grid_sizes[0], dtype=np.float32)
    grid_h = np.arange(grid_sizes[1], dtype=np.float32)
    grid_w = np.arange(grid_sizes[2], dtype=np.float32)

    # Create 3D meshgrid with proper indexing
    grid = np.meshgrid(grid_d, grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0)  # [3, D, H, W]

    # Reshape for processing
    grid = grid.reshape([3, 1, grid_sizes[0], grid_sizes[1], grid_sizes[2]])
    try:
        pos_embed = get_3d_sincos_pos_embed_from_grid_standard(embed_dim, grid)
    except Exception:
        pos_embed = get_3d_sincos_pos_embed_unequal_division(embed_dim, grid)

    if flatten:
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    else:
        pos_embed = pos_embed.reshape([grid_sizes[0], grid_sizes[1], grid_sizes[2], embed_dim])
        if cls_token:
            # Add cls token as separate dimension for non-flattened case
            cls_embed = np.zeros([1, 1, 1, embed_dim])
            pos_embed = np.concatenate([cls_embed, pos_embed], axis=0)

    return pos_embed


# ============================================================================
# VISION TRANSFORMER 3D WITH UPDATED POSITION EMBEDDING
# ============================================================================

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class VisionTransformer3D(nn.Module):
    """
    3D Vision Transformer using the proven standard position embedding approach
    """

    def __init__(
            self,
            image_size: Union[Tuple[int, int, int], int],
            patch_size: Union[Tuple[int, int, int], int],
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'sin_cos_3d',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            drop_path_rate: float = 0.0,  # 新增参数：drop_path率
    ):
        super().__init__()

        self.output_tokens = output_tokens
        self.pos_embed_type = pos_embed_type

        # Convert image_size and patch_size to 3D tuples
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size, image_size)
        else:
            self.image_size = tuple(image_size)

        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size, patch_size)
        else:
            self.patch_size = tuple(patch_size)

        # Calculate grid size for 3D data
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],  # depth
            self.image_size[1] // self.patch_size[1],  # height
            self.image_size[2] // self.patch_size[2]   # width
        )

        self.output_dim = output_dim
        self.pool_type = pool_type
        self.final_ln_after_pool = final_ln_after_pool

        # 3D Convolutional patch embedding
        self.conv1 = nn.Conv3d(
            in_channels=1,  # Assuming single channel for medical images
            out_channels=width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # Class embedding
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        # 3D Position embedding using the standard approach
        if pos_embed_type == 'sin_cos_3d':
            pos_embed_np = get_3d_sincos_pos_embed_standard(
                width,
                self.grid_size,
                cls_token=True,
                flatten=True
            )
            self.positional_embedding = nn.Parameter(
                torch.from_numpy(pos_embed_np).float(),
                requires_grad=False
            )
        elif pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] * self.grid_size[2] + 1, width)
            )
        else:
            raise ValueError(f"Unsupported positional embedding type: {pos_embed_type}")

        # Patch dropout
        from .transformer import PatchDropout
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        # Pre-transformer layer norm
        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)

        # Transformer (you'll need to import this from your transformer.py)
        self.drop_path_rate = drop_path_rate
        from .transformer import Transformer
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path_rate=drop_path_rate,
        )

        # Attention pooling (if specified)
        if attentional_pool:
            from .transformer import AttentionalPooler
            self.attn_pool = AttentionalPooler(
                output_dim,
                width,
                n_head=attn_pooler_heads,
                n_queries=attn_pooler_queries,
            )
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width

        # Final layer norm and projection
        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.init_parameters()

    def interpolate_weights(self, state_dict: Dict[str, torch.Tensor]):
        """
        Interpolate weights from a checkpoint with different patch size or grid size.
        This method should be called from the loading script before `model.load_state_dict()`.
        """
        # Interpolate conv1.weight if patch sizes are different
        conv1_key = 'conv1.weight'
        if conv1_key in state_dict and self.conv1.weight.shape != state_dict[conv1_key].shape:
            ckpt_conv_weight = state_dict[conv1_key]
            current_conv_weight_shape = self.conv1.weight.shape

            # Use trilinear interpolation for the kernel
            interpolated_weight = torch.nn.functional.interpolate(
                ckpt_conv_weight,
                size=current_conv_weight_shape[2:], # Only interpolate spatial/temporal dimensions
                mode='trilinear',
                align_corners=False
            )
            state_dict[conv1_key] = interpolated_weight

        # Interpolate positional_embedding if grid sizes are different
        pos_embed_key = 'positional_embedding'
        if pos_embed_key in state_dict and self.positional_embedding.shape != state_dict[pos_embed_key].shape:
            ckpt_pos_embed = state_dict[pos_embed_key]
            # Separate class token and patch embeddings
            cls_token_ckpt = ckpt_pos_embed[:1, :]
            patch_embed_ckpt = ckpt_pos_embed[1:, :]

            # Get old grid size from checkpoint shape
            num_patches_ckpt = patch_embed_ckpt.shape[0]

            # Heuristic to find old grid_size. Assumes it was somewhat cubic.
            # This part may need adjustment if grid was highly non-uniform.
            d_h_w_prod_ckpt = num_patches_ckpt
            side_ckpt = round(d_h_w_prod_ckpt ** (1/3))

            # This is a guess. We need to know how the original model was structured.
            # For now, let's assume we can't perfectly know the old D,H,W ratio.
            # A simple interpolation of the flat sequence is more robust if we don't know old grid dimensions.
            patch_embed_ckpt = patch_embed_ckpt.permute(1, 0).unsqueeze(0) # [1, C, N_old]

            num_patches_current = self.positional_embedding.shape[0] - 1

            interpolated_patch_embed = torch.nn.functional.interpolate(
                patch_embed_ckpt,
                size=num_patches_current,
                mode='linear',
                align_corners=False
            )

            interpolated_patch_embed = interpolated_patch_embed.squeeze(0).permute(1, 0)

            # Combine with class token
            interpolated_pos_embed = torch.cat([cls_token_ckpt, interpolated_patch_embed], dim=0)
            state_dict[pos_embed_key] = interpolated_pos_embed

        return state_dict

    def init_parameters(self):
        # Initialize parameters following standard practices
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.transformer.grad_checkpointing = enable

    def resize_pos_embed(self, new_image_size: Tuple[int, int, int]):
        """
        Resize position embeddings to adapt to a new image resolution.
        """
        new_grid_size = (
            new_image_size[0] // self.patch_size[0],
            new_image_size[1] // self.patch_size[1],
            new_image_size[2] // self.patch_size[2],
        )

        if new_grid_size == self.grid_size:
            return

        if self.pos_embed_type == 'sin_cos_3d':
            # Interpolate sin-cos embeddings for the new grid size. This is more robust.
            cls_pos_embed = self.positional_embedding[:1, :]
            patch_pos_embed = self.positional_embedding[1:, :]

            old_D, old_H, old_W = self.grid_size
            width = self.positional_embedding.shape[-1]

            # Reshape to 3D grid for interpolation
            patch_pos_embed = patch_pos_embed.reshape(old_D, old_H, old_W, width).permute(3, 0, 1, 2)
            patch_pos_embed = patch_pos_embed.unsqueeze(0)  # Add batch dim

            new_D, new_H, new_W = new_grid_size

            # Interpolate using trilinear mode for 3D
            resized_patch_pos_embed = torch.nn.functional.interpolate(
                patch_pos_embed,
                size=(new_D, new_H, new_W),
                mode='trilinear',
                align_corners=False
            )

            # Reshape back and remove batch dim
            resized_patch_pos_embed = resized_patch_pos_embed.squeeze(0).permute(1, 2, 3, 0).reshape(-1, width)

            # Concatenate with class token
            new_pos_embed = torch.cat([cls_pos_embed, resized_patch_pos_embed], dim=0)
            self.positional_embedding = nn.Parameter(new_pos_embed, requires_grad=False)

        elif self.pos_embed_type == 'learnable':
            # Interpolate learnable embeddings
            cls_pos_embed = self.positional_embedding[:1, :]
            patch_pos_embed = self.positional_embedding[1:, :]

            old_D, old_H, old_W = self.grid_size
            width = self.positional_embedding.shape[-1]

            # Reshape to 3D grid for interpolation
            patch_pos_embed = patch_pos_embed.reshape(old_D, old_H, old_W, width).permute(3, 0, 1, 2)
            patch_pos_embed = patch_pos_embed.unsqueeze(0)  # Add batch dim

            new_D, new_H, new_W = new_grid_size

            # Interpolate using trilinear mode for 3D
            resized_patch_pos_embed = torch.nn.functional.interpolate(
                patch_pos_embed,
                size=(new_D, new_H, new_W),
                mode='trilinear',
                align_corners=False
            )

            # Reshape back and remove batch dim
            resized_patch_pos_embed = resized_patch_pos_embed.squeeze(0).permute(1, 2, 3, 0).reshape(-1, width)

            # Concatenate with class token
            new_pos_embed = torch.cat([cls_pos_embed, resized_patch_pos_embed], dim=0)
            self.positional_embedding = nn.Parameter(new_pos_embed)
        else:
            raise ValueError(f"Unsupported positional embedding type for resizing: {self.pos_embed_type}")

        # Update grid and image size attributes
        self.grid_size = new_grid_size
        self.image_size = new_image_size

    def _embeds(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 3D input to patch embeddings with position encoding"""
        # 3D convolution: x shape [B, C, D, H, W] -> [B, width, grid_d, grid_h, grid_w]
        x = self.conv1(x)

        # Flatten spatial dimensions: [B, width, grid_d * grid_h * grid_w]
        x = x.reshape(x.shape[0], x.shape[1], -1)

        # Transpose: [B, grid_d * grid_h * grid_w, width]
        x = x.permute(0, 2, 1)

        # Add class token
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

        # Add position embeddings
        x = x + self.positional_embedding.to(x.dtype)

        # Apply patch dropout
        x = self.patch_dropout(x)

        # Apply pre-transformer norm
        x = self.ln_pre(x)

        return x

    def _pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool the transformer output"""
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        return pooled, tokens

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Global pooling operation"""
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor):
        """Forward pass through the 3D Vision Transformer"""
        # Input: [B, C, D, H, W]
        x = self._embeds(x)  # [B, N+1, width] where N = grid_d * grid_h * grid_w
        x = self.transformer(x)  # [B, N+1, width]
        pooled, tokens = self._pool(x)  # [B, width], [B, N, width]

        # Final projection
        if self.proj is not None:
            pooled = pooled @ self.proj
            # Also apply projection to tokens when output_tokens is True
            #! Change here 250731.
            if self.output_tokens:
                tokens = tokens @ self.proj

        if self.output_tokens:
            return pooled, tokens

        return pooled

    #! From timm, by Jx
    def get_drop_path_info(self) -> Dict[str, Any]:
        """
        获取当前模型的drop_path信息
        """
        info = {
            'drop_path_rate': self.drop_path_rate,
            'num_layers': len(self.transformer.resblocks),
            'per_layer_rates': []
        }

        for i, block in enumerate(self.transformer.resblocks):
            if hasattr(block, 'drop_path_1') and hasattr(block.drop_path_1, 'drop_prob'):
                info['per_layer_rates'].append(block.drop_path_1.drop_prob)
            else:
                info['per_layer_rates'].append(0.0)

        return info

    def print_drop_path_config(self, detailed: bool = True):
        """
        格式化打印DropPath配置信息
        直接添加这个方法到你的VisionTransformer3D类中
        """
        info = self.get_drop_path_info()

        print("="*70)
        print(f"  3D Vision Transformer - DropPath Configuration")
        print("="*70)

        # 基本信息
        print(f"📊 Configuration:")
        print(f"   • Max DropPath Rate: {info['drop_path_rate']:.3f}")
        print(f"   • Number of Layers: {info['num_layers']}")

        if detailed and info['per_layer_rates']:
            rates = info['per_layer_rates']
            print(f"   • Min Rate: {min(rates):.3f}")
            print(f"   • Max Rate: {max(rates):.3f}")
            print(f"   • Mean Rate: {sum(rates)/len(rates):.3f}")

            print(f"\n📋 Layer-wise Rates:")
            print("-"*50)
            print(f"{'Layer':<8} {'Rate':<10} {'Percentage':<12} {'Status'}")
            print("-"*50)

            for i, rate in enumerate(rates):
                percentage = f"{rate*100:.1f}%"
                status = "Active" if rate > 0 else "Disabled"
                print(f"{i+1:<8} {rate:<10.3f} {percentage:<12} {status}")

            print("-"*50)

            # 简单的效果分析
            active_layers = sum(1 for rate in rates if rate > 0)
            print(f"\n💡 Analysis:")
            print(f"   • {active_layers}/{len(rates)} layers have DropPath enabled")

            if info['drop_path_rate'] > 0:
                print(f"   • Training: Randomly skip blocks for regularization")
                print(f"   • Inference: All blocks active (DropPath disabled)")
            else:
                print(f"   • DropPath is disabled - no stochastic depth")

        print("="*70)

