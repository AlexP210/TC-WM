import torch.nn as nn
from einops import rearrange
from models.base import BaseEncoder


class TDMPC2CNNEncoder(BaseEncoder):
    """
    TDMPC2-style trainable CNN encoder for 224x224 observations.

    Architecture: 4 Conv layers with stride-2 downsampling → 14x14 spatial grid
    → linear projection to emb_dim → (B, 196, emb_dim) patch sequence.

    Compatible with ViT predictor and ProjectedVQVAEDecoder.
    Reference: TD-MPC2 (Hansen et al., 2023).
    """

    def __init__(self, emb_dim: int = 256, num_channels: int = 32):
        super().__init__()
        self.name = "tdmpc2_cnn"
        self.emb_dim = emb_dim
        self.latent_ndim = 2      # patch sequence: (B, N, D)
        self.patch_size = 16      # 224 / 14 ≈ 16, for compatibility with decoder_scale=16

        self.cnn = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3),   # 224→112
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=5, stride=2, padding=2),  # 112→56
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1),  # 56→28
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1),  # 28→14
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(num_channels, emb_dim)

    def forward(self, x):
        """x: (B, 3, H, W) → (B, N, emb_dim) where N = (H/16)^2"""
        feat = self.cnn(x)                              # (B, C, 14, 14)
        feat = rearrange(feat, "b c h w -> b (h w) c")  # (B, 196, C)
        return self.proj(feat)                          # (B, 196, emb_dim)
