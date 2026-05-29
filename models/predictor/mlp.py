import torch.nn as nn
from models.base import BasePredictor


class MLPPredictor(BasePredictor):
    """
    TDMPC2-style MLP predictor.

    Applies a per-token MLP (no cross-token attention) to the input sequence.
    Compatible with ViTPredictor interface: takes (B, T*P, D) and returns (B, T*P, D).

    For vector latents (P=1) this is equivalent to the original TDMPC2 MLP transition model.
    For patch sequences (P>1) each patch is processed independently.

    Reference: TD-MPC2 (Hansen et al., 2023).
    """

    def __init__(self, *, num_patches, num_frames, dim, hidden_dim=512, num_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, T*P, D) — apply MLP per token with residual
        return x + self.mlp(self.norm(x))
