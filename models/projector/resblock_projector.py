import torch
import torch.nn as nn

from models.utils.resblocks import ResBlock1d


class ResBlockProjector(nn.Module):
    """
    Residual 1D-conv projector for token grids.

    Accepts either:
      - forward(x) where x is (B, T, P, C)
      - forward(visual_emb, proprio_emb) and concatenates on last dim
    """

    def __init__(
        self,
        in_features=None,
        out_features=None,
        visual_emb_dim=None,
        proprio_emb_dim=None,
        projected_dim=None,
        hidden_channels=256,
        num_blocks=2,
        kernel_size=3,
        dropout=0.0,
        activation=nn.ReLU,
        use_history=False,
        num_hist=3,
        act_embed_dim=None,
    ):
        super().__init__()
        if in_features is None:
            if visual_emb_dim is None or proprio_emb_dim is None:
                raise ValueError("Provide in_features or both visual_emb_dim and proprio_emb_dim.")
            in_features = visual_emb_dim + proprio_emb_dim
        if out_features is None:
            if projected_dim is None:
                raise ValueError("Provide out_features or projected_dim.")
            out_features = projected_dim

        self.in_features = in_features
        self.out_features = out_features
        self.projected_dim = out_features
        self.use_history = use_history
        self.num_hist = num_hist
        self.act_embed_dim = act_embed_dim

        self.in_proj = nn.Conv1d(in_features, self.projected_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                ResBlock1d(
                    channels=self.projected_dim,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_proj = (
            nn.Identity()
            if self.projected_dim == out_features
            else nn.Conv1d(self.projected_dim, out_features, kernel_size=1)
        )

    def forward(self, visual_emb, proprio_emb=None, act=None):
        if proprio_emb is None:
            x = visual_emb
        else:
            if visual_emb.shape[:-1] != proprio_emb.shape[:-1]:
                raise ValueError("visual_emb and proprio_emb must share batch/time/patch dims")
            if visual_emb.shape[-1] + proprio_emb.shape[-1] != self.in_features:
                raise ValueError(
                    f"Input feature mismatch: got {visual_emb.shape[-1]}+{proprio_emb.shape[-1]}, "
                    f"expected {self.in_features}."
                )
            x = torch.cat([visual_emb, proprio_emb], dim=-1)
        if x.dim() == 4:
            B, T, P, C = x.shape
            force_patch_dim = False
        elif x.dim() == 3:
            B, T, C = x.shape
            P = 1
            x = x.unsqueeze(2)
            force_patch_dim = True
        else:
            raise ValueError(f"Expected input with 3 or 4 dims, got {x.dim()}")

        if C != self.in_features:
            raise ValueError(f"Input feature mismatch: got {C}, expected {self.in_features}.")

        x = x.permute(0, 1, 3, 2).reshape(B * T, C, P)
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_proj(x)
        x = x.reshape(B, T, self.out_features, P).permute(0, 1, 3, 2)

        return x
