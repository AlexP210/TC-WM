"""
V-JEPA visual encoder for TC-WM.

Loads the V-JEPA ViT encoder (target_encoder) from an official checkpoint
and adapts it for single-frame image encoding. V-JEPA is trained on video
with 3D patch embeddings (tubelet_size=2), so we collapse the temporal dim
to produce a standard 2D patch embedding for single images.

Output: (B, N, D) patch tokens, matching the DINOv2 encoder interface.

Reference:
    Bardes et al., "Revisiting Feature Prediction for Learning Visual
    Representations from Video", 2024.
    https://github.com/facebookresearch/jepa
"""

import math
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseEncoder


# ---------------------------------------------------------------------------
# Lightweight ViT components (from V-JEPA source, simplified for inference)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        y, _ = self.attn(self.norm1(x), mask=mask)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed2D(nn.Module):
    """2D image patch embedding (standard ViT)."""
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sincos positional embeddings."""
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)
    emb_h = _get_1d_sincos(embed_dim // 2, grid_h)
    emb_w = _get_1d_sincos(embed_dim // 2, grid_w)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _get_1d_sincos(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ---------------------------------------------------------------------------
# V-JEPA Encoder wrapper
# ---------------------------------------------------------------------------

# Available V-JEPA model configs
VJEPA_CONFIGS = {
    "vjepa_vitl16": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "patch_size": 16,
        "tubelet_size": 2,
        "url": "https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar",
    },
    "vjepa_vith16": {
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "patch_size": 16,
        "tubelet_size": 2,
        "url": "https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar",
    },
}

# Default checkpoint directory
_CKPT_DIR = os.path.expanduser("~/.cache/vjepa")


class VJEPAEncoder(BaseEncoder):
    """
    V-JEPA visual encoder adapted for single-frame image encoding.

    Loads the frozen target_encoder from V-JEPA and converts the 3D
    patch embedding (used for video) to a 2D patch embedding for images.

    Args:
        name: Model config name (e.g., "vjepa_vitl16").
        img_size: Input image resolution (default 224).
        checkpoint_path: Path to .pth.tar checkpoint. If None, downloads
            from the official URL.
        encoder_key: Key in checkpoint dict ("target_encoder" or "encoder").
    """

    def __init__(
        self,
        name: str = "vjepa_vitl16",
        img_size: int = 224,
        checkpoint_path: str = None,
        encoder_key: str = "target_encoder",
    ):
        super().__init__()
        assert name in VJEPA_CONFIGS, (
            f"Unknown V-JEPA model: {name}. Choose from {list(VJEPA_CONFIGS.keys())}"
        )
        cfg = VJEPA_CONFIGS[name]

        self.name = name
        self.img_size = img_size
        self.patch_size = cfg["patch_size"]
        self.latent_ndim = 2  # patch tokens: (B, N, D)

        embed_dim = cfg["embed_dim"]
        depth = cfg["depth"]
        num_heads = cfg["num_heads"]
        tubelet_size = cfg["tubelet_size"]

        self.emb_dim = embed_dim  # exposed for downstream modules

        # ImageNet normalization (registered before loading weights so they
        # don't appear as "missing" during load_state_dict)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)

        # --- Build 2D ViT encoder ---
        self.patch_embed = PatchEmbed2D(
            patch_size=self.patch_size, in_chans=3, embed_dim=embed_dim
        )
        num_patches = (img_size // self.patch_size) ** 2
        self.num_patches = num_patches

        # 2D sincos positional embedding (fixed)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )
        sincos = get_2d_sincos_pos_embed(embed_dim, img_size // self.patch_size)
        self.pos_embed.data.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

        # Transformer blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                qkv_bias=True, norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # --- Load and adapt pretrained weights ---
        self._load_pretrained(cfg, checkpoint_path, encoder_key, tubelet_size)

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def _load_pretrained(self, cfg, checkpoint_path, encoder_key, tubelet_size):
        """Load V-JEPA checkpoint and adapt 3D conv to 2D conv."""
        if checkpoint_path is None:
            # Download to default cache directory
            os.makedirs(_CKPT_DIR, exist_ok=True)
            filename = cfg["url"].split("/")[-1]
            checkpoint_path = os.path.join(_CKPT_DIR, filename)
            if not os.path.exists(checkpoint_path):
                print(f"[V-JEPA] Downloading checkpoint to {checkpoint_path} ...")
                torch.hub.download_url_to_dest(cfg["url"], checkpoint_path)

        print(f"[V-JEPA] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if encoder_key not in ckpt:
            raise KeyError(
                f"Key '{encoder_key}' not found in checkpoint. "
                f"Available: {list(ckpt.keys())}"
            )

        raw_state = ckpt[encoder_key]

        # Strip "module.backbone." prefix from DDP checkpoint
        prefix = "module.backbone."
        state_dict = {}
        for k, v in raw_state.items():
            if k.startswith(prefix):
                state_dict[k[len(prefix):]] = v
            else:
                state_dict[k] = v

        # --- Convert 3D patch embedding to 2D ---
        # Original shape: (embed_dim, in_chans, tubelet_size, patch_size, patch_size)
        # Target shape:   (embed_dim, in_chans, patch_size, patch_size)
        key_pe_w = "patch_embed.proj.weight"
        if key_pe_w in state_dict and state_dict[key_pe_w].ndim == 5:
            w3d = state_dict[key_pe_w]  # (D, C, T, H, W)
            # Sum over temporal dimension to collapse tubelet into single frame
            w2d = w3d.sum(dim=2)  # (D, C, H, W)
            state_dict[key_pe_w] = w2d
            print(f"[V-JEPA] Converted patch_embed 3D->2D: "
                  f"{tuple(w3d.shape)} -> {tuple(w2d.shape)}")

        # --- Adapt positional embedding ---
        # Checkpoint pos_embed: (1, T*H*W, D) for video
        # We need:              (1, H*W, D) for single image
        key_pos = "pos_embed"
        if key_pos in state_dict:
            ckpt_pos = state_dict[key_pos]  # (1, N_video, D)
            n_video = ckpt_pos.shape[1]
            grid_size = self.img_size // self.patch_size
            n_spatial = grid_size * grid_size  # 196 for 224/16

            if n_video != n_spatial:
                # V-JEPA video pos_embed has shape (1, T*H*W, D)
                # Recompute clean 2D sincos instead of slicing (more reliable)
                print(f"[V-JEPA] Replacing video pos_embed ({n_video}) with "
                      f"2D sincos ({n_spatial})")
                del state_dict[key_pos]  # use our pre-initialized 2D sincos

        # Load weights (strict=False allows missing pos_embed)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            # pos_embed is expected to be missing (we use our own 2D sincos)
            expected_missing = {"pos_embed", "image_mean", "image_std"}
            non_pos_missing = [k for k in missing if k not in expected_missing]
            if non_pos_missing:
                print(f"[V-JEPA] WARNING: missing keys: {non_pos_missing}")
        if unexpected:
            print(f"[V-JEPA] Unexpected keys (ignored): {unexpected[:10]}")

        print(f"[V-JEPA] Loaded {self.name} encoder successfully "
              f"(emb_dim={self.emb_dim}, patches={self.num_patches})")

    def train(self, mode=True):
        """Override to keep model always in eval mode."""
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images into patch tokens.

        Args:
            x: (B, 3, H, W) input images, pixel values in [0, 1].

        Returns:
            (B, N, D) patch token embeddings where N = (H/patch_size)^2
            and D = embed_dim.
        """
        # Normalize with ImageNet stats
        if x.dtype != self.image_mean.dtype:
            x = x.to(dtype=self.image_mean.dtype)
        x = (x - self.image_mean) / self.image_std

        # Patch embedding + positional encoding
        x = self.patch_embed(x)  # (B, N, D)

        # Interpolate pos_embed if input size differs from training size
        pos_embed = self._interpolate_pos_encoding(x)
        x = x + pos_embed

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x  # (B, N, D)

    def _interpolate_pos_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate positional embeddings if spatial resolution changed."""
        N = x.shape[1]
        if N == self.num_patches:
            return self.pos_embed

        dim = self.pos_embed.shape[-1]
        n0 = int(math.sqrt(self.num_patches))
        n1_h = n1_w = int(math.sqrt(N))

        pos = self.pos_embed.reshape(1, n0, n0, dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(n1_h, n1_w), mode="bicubic",
                            align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return pos
