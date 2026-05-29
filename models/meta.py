"""
ModuleMeta and LatentBundle — framework-level data contracts.

ModuleMeta flows alongside tensors through the pipeline so each module
knows the format of what it receives (patch_sequence / vector / spatial).

LatentBundle is assembled in WorldModel.forward() and passed to
TaskObjective / loss functions.  Each loss picks the keys it needs and
ignores the rest.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import torch
from torch import Tensor


@dataclass
class ModuleMeta:
    """
    Carries shape/format information alongside tensors in the pipeline.

    shape_type:
        "patch_sequence"  — tensor is (B, N, D)
        "vector"          — tensor is (B, D)
        "spatial"         — tensor is (B, D, H, W)
        "unknown"         — not yet set (initial state)

    n_tokens: number of patch tokens; None when shape_type == "vector".
    emb_dim:  last dimension of the tensor.
    extra:    arbitrary module-specific metadata for custom pass-through.
    """
    shape_type: str = "unknown"
    emb_dim:    int = 0
    n_tokens:   Optional[int] = None
    extra:      dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def is_patch(self) -> bool:
        return self.shape_type == "patch_sequence"

    def is_vector(self) -> bool:
        return self.shape_type == "vector"

    def is_spatial(self) -> bool:
        return self.shape_type == "spatial"

    def spatial_size(self) -> Tuple[int, int]:
        """Return (H, W) assuming square patch grid."""
        assert self.n_tokens is not None, "n_tokens is None — not a patch tensor"
        H = W = int(self.n_tokens ** 0.5)
        assert H * W == self.n_tokens, (
            f"n_tokens={self.n_tokens} is not a perfect square"
        )
        return H, W

    @classmethod
    def from_tensor(cls, z: Tensor) -> "ModuleMeta":
        """Infer meta from tensor shape alone (best-effort)."""
        if z.ndim == 2:
            return cls(shape_type="vector", emb_dim=z.shape[-1])
        elif z.ndim == 3:
            return cls(shape_type="patch_sequence",
                       emb_dim=z.shape[-1], n_tokens=z.shape[1])
        elif z.ndim == 4:
            return cls(shape_type="spatial", emb_dim=z.shape[1])
        return cls()


@dataclass
class LatentBundle:
    """
    Standardised input dict for all loss functions and task objectives.

    Assembled in WorldModel.forward() after all modules have run.
    Each loss function picks only the keys it needs; missing Optional
    fields are None and should be checked (or asserted) inside the loss.

    Key naming convention:
        z_enc    — encoder output, before projection        (B, N, D_enc)
        z_proj   — projector output, compressed             (B, N, D_proj) or (B, D_proj)
        z_pred   — predictor's predicted next-frame latent  same shape as z_proj
        z_target — actual next frame, encoded + projected   same shape as z_proj
        obs      — raw observation tensor                   (B, C, H, W)
        recon    — decoder reconstruction                   (B, C, H, W)
        proprio  — proprioceptive state from dataset         (B, proprio_dim) [optional]
        reward   — reward from dataset                      (B,)             [optional]
        value    — value from dataset                       (B,)             [optional]
        action   — action tensor                            (B, action_dim)
    """
    action:   Tensor

    z_enc:    Optional[Tensor] = None
    z_proj:   Optional[Tensor] = None
    z_pred:   Optional[Tensor] = None
    z_target: Optional[Tensor] = None

    obs:      Optional[Tensor] = None
    recon:    Optional[Tensor] = None

    proprio:  Optional[Tensor] = None
    reward:   Optional[Tensor] = None
    value:    Optional[Tensor] = None

    # Optional per-sample weight (B,): set by MTJEPAWorldModel to valid-action fraction.
    # ConsistencyLossObjective uses it to down-weight samples with padded action dims.
    mask:     Optional[Tensor] = None

    def to(self, device) -> "LatentBundle":
        """Move all tensors to device."""
        def _maybe(t):
            return t.to(device) if t is not None else None
        return LatentBundle(
            action   = self.action.to(device),
            z_enc    = _maybe(self.z_enc),
            z_proj   = _maybe(self.z_proj),
            z_pred   = _maybe(self.z_pred),
            z_target = _maybe(self.z_target),
            obs      = _maybe(self.obs),
            recon    = _maybe(self.recon),
            proprio  = _maybe(self.proprio),
            reward   = _maybe(self.reward),
            value    = _maybe(self.value),
            mask     = _maybe(self.mask),
        )
