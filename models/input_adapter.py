"""
InputAdapter — reusable shape-normalisation layer.

Placed at the entry of any loss function or module that needs to handle
multiple input shapes.  The core computation only sees one canonical format.

Modes
-----
as_is             pass through unchanged
mean_pool         (B, N, D) → mean over N → (B, D)
flatten           (B, N, D) → (B, N*D)
per_patch         keep (B, N, D); loss iterates over N independently
reshape_to_spatial(B, N, D) → (B, D, H, W) using meta.n_tokens (must be square)
tile_to_spatial   (B, D)    → (B, D, H, W) by tiling (for vector → spatial)

An optional ProjectionHead can be appended after mode handling.
"""

from __future__ import annotations
from typing import Optional, List

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from models.meta import ModuleMeta


# ---------------------------------------------------------------------------
# Projection head (used inside InputAdapter)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Configurable projection / MLP head applied after shape normalisation.

    Works on both (B, D) and (B, N, D) inputs — broadcasts over N when needed.

    Parameters
    ----------
    input_dim:   input feature dim (inferred from meta if None at build time)
    hidden_dim:  hidden dim; None → single linear layer (no hidden)
    output_dim:  output feature dim; None → same as input_dim
    num_layers:  1 = linear, 2+ = MLP with hidden layers
    activation:  activation class (default: ReLU)
    dropout:     dropout probability
    normalize:   L2-normalise the output (useful for contrastive losses)
    """

    def __init__(
        self,
        input_dim:  int,
        output_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        activation: type = nn.ReLU,
        dropout:    float = 0.0,
        normalize:  bool = False,
    ):
        super().__init__()
        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or output_dim
        self.normalize = normalize

        layers: List[nn.Module] = []
        prev = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, D) or (B, N, D) — nn.Linear broadcasts over leading dims
        out = self.net(x)
        if self.normalize:
            out = nn.functional.normalize(out, dim=-1)
        return out


# ---------------------------------------------------------------------------
# InputAdapter
# ---------------------------------------------------------------------------

VALID_MODES = frozenset({
    "as_is",
    "mean_pool",
    "flatten",
    "per_patch",
    "reshape_to_spatial",
    "tile_to_spatial",
})


class InputAdapter(nn.Module):
    """
    Shape-normalisation wrapper.

    Parameters
    ----------
    mode:       one of VALID_MODES (see module docstring)
    projection: optional dict of kwargs for ProjectionHead;
                set projection=None (or omit) for no projection
    """

    def __init__(
        self,
        mode: str = "as_is",
        projection: Optional[dict] = None,
    ):
        super().__init__()
        assert mode in VALID_MODES, (
            f"InputAdapter: unknown mode '{mode}'. "
            f"Valid modes: {sorted(VALID_MODES)}"
        )
        self.mode = mode
        self.proj: Optional[ProjectionHead] = None
        if projection is not None and projection.get("enabled", True):
            kw = {k: v for k, v in projection.items() if k != "enabled"}
            self.proj = ProjectionHead(**kw)

    # ------------------------------------------------------------------

    def forward(self, x: Tensor, meta: ModuleMeta) -> Tensor:
        """
        Apply shape normalisation then optional projection.

        Parameters
        ----------
        x:    input tensor
        meta: ModuleMeta from the previous module

        Returns
        -------
        Tensor in the canonical shape expected by the downstream loss/module.
        """
        x = self._apply_mode(x, meta)
        if self.proj is not None:
            x = self.proj(x)
        return x

    # ------------------------------------------------------------------

    def _apply_mode(self, x: Tensor, meta: ModuleMeta) -> Tensor:
        if self.mode == "as_is":
            return x

        elif self.mode == "mean_pool":
            assert x.ndim == 3, (
                f"mean_pool expects (B, N, D), got {tuple(x.shape)}"
            )
            return x.mean(dim=1)

        elif self.mode == "flatten":
            assert x.ndim == 3, (
                f"flatten expects (B, N, D), got {tuple(x.shape)}"
            )
            return x.flatten(1)

        elif self.mode == "per_patch":
            assert x.ndim == 3, (
                f"per_patch expects (B, N, D), got {tuple(x.shape)}"
            )
            # keep shape — downstream iterates over patch dim
            return x

        elif self.mode == "reshape_to_spatial":
            assert x.ndim == 3, (
                f"reshape_to_spatial expects (B, N, D), got {tuple(x.shape)}"
            )
            assert meta.n_tokens is not None, (
                "reshape_to_spatial requires meta.n_tokens to be set"
            )
            H, W = meta.spatial_size()   # asserts square, raises if not
            assert x.shape[1] == meta.n_tokens, (
                f"n_tokens mismatch: meta={meta.n_tokens}, tensor N={x.shape[1]}"
            )
            return rearrange(x, "b (h w) d -> b d h w", h=H, w=W)

        elif self.mode == "tile_to_spatial":
            raise NotImplementedError(
                "tile_to_spatial is not yet implemented"
            )

        raise RuntimeError(f"Unhandled InputAdapter mode: {self.mode}")
