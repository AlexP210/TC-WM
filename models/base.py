"""
Base abstract classes for the Universal Latent Dynamics WM Framework.

Design principle
----------------
Existing implementations keep their original forward(x) signatures unchanged.
Each ABC adds a higher-level method (encode / project / predict / decode /
compute) that wraps the existing forward and returns the (output, meta, aux_losses)
triple expected by the new framework.

This means:
- Old code that calls encoder.forward(x) continues to work unmodified.
- New framework code calls encoder.encode(x, meta) for the full interface.
- Existing subclasses just add `class Foo(BaseEncoder)` — no other change needed.

Forward signature conventions
------------------------------
All higher-level methods return a triple:
    (output: Tensor, meta: ModuleMeta, aux_losses: Dict[str, Tensor])

aux_losses is {} for deterministic modules; stochastic modules (e.g. RSSM)
return {"kl": tensor, ...} computed internally.  The trainer sums all
aux_losses with configured weights — it does not need to know whether a
module is stochastic.

Predictor input
---------------
ViTPredictor (and future predictors) receive the full history window
already flattened to (B, T*N, D).  Callers are responsible for packing
history frames before calling predict().
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import replace

from models.meta import ModuleMeta, LatentBundle


AuxLosses = Dict[str, Tensor]


# ---------------------------------------------------------------------------
# BaseEncoder
# ---------------------------------------------------------------------------

class BaseEncoder(nn.Module, ABC):
    """
    Visual (or state) encoder.

    Subclasses must implement forward(x) → Tensor.
    The base class provides encode() which wraps forward() with meta.

    Attributes expected on subclasses
    ----------------------------------
    emb_dim:      int   — output feature dimension
    latent_ndim:  int   — 1 for CLS token (B,1,D), 2 for patch sequence (B,N,D)
    name:         str   — encoder identifier used in identity checks
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Encode observations.  x: (B, C, H, W).  Returns (B, N, D) or (B, 1, D)."""

    def encode(self, x: Tensor, meta: ModuleMeta) -> Tuple[Tensor, ModuleMeta, AuxLosses]:
        """
        Full framework interface.  Calls forward() and updates meta.

        Returns
        -------
        z:        encoded tensor   (B, N, D)
        new_meta: updated ModuleMeta
        {}:       encoders have no aux losses
        """
        z = self.forward(x)
        new_meta = self._make_meta(z)
        return z, new_meta, {}

    def _make_meta(self, z: Tensor) -> ModuleMeta:
        assert z.ndim == 3, f"Encoder output must be (B, N, D), got {tuple(z.shape)}"
        n_tokens = z.shape[1]
        shape_type = "patch_sequence" if n_tokens > 1 else "vector"
        return ModuleMeta(shape_type=shape_type, emb_dim=z.shape[-1], n_tokens=n_tokens)


# ---------------------------------------------------------------------------
# BaseProjector
# ---------------------------------------------------------------------------

class BaseProjector(nn.Module, ABC):
    """
    Projects concatenated (visual + proprio) features to the latent space.

    Subclasses must implement forward(x, proprio=None) → Tensor.

    Special case — torch.nn.Identity:
        Identity is used as the DINO-WM baseline (no compression).
        It is NOT a subclass of BaseProjector but is handled explicitly
        in init_models() via a _target_ check.
    """

    @abstractmethod
    def forward(self, x: Tensor, proprio: Optional[Tensor] = None) -> Tensor:
        """
        x:      visual features   (B, N, D_vis)  or  (B, D_vis)
        proprio: proprio embedding (B, N, D_prop) or  (B, D_prop)  [optional]
        Returns projected tensor with same spatial structure as x.
        """

    def project(
        self,
        x: Tensor,
        proprio: Optional[Tensor],
        meta: ModuleMeta,
    ) -> Tuple[Tensor, ModuleMeta, AuxLosses]:
        """Full framework interface."""
        z = self.forward(x, proprio)
        new_meta = self._make_meta(z, meta)
        return z, new_meta, {}

    def _make_meta(self, z: Tensor, prev_meta: ModuleMeta) -> ModuleMeta:
        """
        Update meta from projector output.
        If output collapses the patch dim (B, D), switch to "vector".
        """
        if z.ndim == 2:
            return replace(prev_meta, shape_type="vector",
                           emb_dim=z.shape[-1], n_tokens=None)
        elif z.ndim == 3:
            return replace(prev_meta, emb_dim=z.shape[-1], n_tokens=z.shape[1])
        return prev_meta


# ---------------------------------------------------------------------------
# BasePredictor
# ---------------------------------------------------------------------------

class BasePredictor(nn.Module, ABC):
    """
    Temporal dynamics predictor.

    Input: full history window packed as (B, T*N, D) — T frames, N tokens each.
    Output: predicted next-frame tokens (B, T*N, D) with same packing.

    Stochastic predictors (e.g. RSSM) compute KL internally and return it
    in aux_losses.  Deterministic predictors return {}.

    Subclasses must implement forward(x) → Tensor.
    For stochastic models, override forward_with_losses() instead.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T*N, D).  Returns (B, T*N, D)."""

    def forward_with_losses(self, x: Tensor) -> Tuple[Tensor, AuxLosses]:
        """
        Deterministic default: calls forward(), no aux losses.
        Stochastic subclasses override this to return {"kl": tensor, ...}.
        """
        return self.forward(x), {}

    def predict(
        self,
        x: Tensor,
        meta: ModuleMeta,
    ) -> Tuple[Tensor, ModuleMeta, AuxLosses]:
        """Full framework interface."""
        z, aux = self.forward_with_losses(x)
        # meta shape_type / n_tokens unchanged — predictor keeps the format
        new_meta = replace(meta, emb_dim=z.shape[-1])
        return z, new_meta, aux


# ---------------------------------------------------------------------------
# BaseTaskObjective
# ---------------------------------------------------------------------------

class BaseTaskObjective(nn.Module, ABC):
    """
    Task-centric loss module (InfoNCE alignment, reward prediction, etc.).

    Receives a LatentBundle and ModuleMeta; returns only aux_losses.
    No tensor output — the objective only contributes to the loss.

    Class-level attributes (override in subclasses):
        loss_key    — key in the returned AuxLosses dict that VWorldModel adds
                      to the total loss (weighted by loss_weight).
        loss_weight — scalar multiplier for the primary loss.

    The bundle may contain None fields; each objective asserts the keys
    it actually needs at the top of compute().
    """

    loss_key: str = "task_loss"   # override in each subclass
    loss_weight: float = 1.0      # override in __init__ of each subclass

    @abstractmethod
    def compute(
        self,
        bundle: LatentBundle,
        meta: ModuleMeta,
    ) -> AuxLosses:
        """
        Compute task-centric losses from the latent bundle.

        Returns
        -------
        Dict[str, Tensor] — loss name → scalar tensor.
        Names become keys in the global loss dict, e.g. "infonce", "reward_pred".
        """

    def forward(
        self,
        bundle: LatentBundle,
        meta: ModuleMeta,
    ) -> Tuple[None, ModuleMeta, AuxLosses]:
        """Thin wrapper so TaskObjective fits the (output, meta, aux) pattern."""
        return None, meta, self.compute(bundle, meta)


# ---------------------------------------------------------------------------
# BaseDecoder
# ---------------------------------------------------------------------------

class BaseDecoder(nn.Module, ABC):
    """
    Optional image / feature decoder.

    Design principle
    ----------------
    The decoder is a black box from VWorldModel's perspective.
    Caller passes `z` (the latent) and an optional `targets` dict;
    the decoder owns all internal stages, stop-gradient placement,
    and auxiliary loss computation.  VWorldModel never inspects
    decoder internals — it just sums the returned loss dict.

    Implementing a new decoder
    --------------------------
    Implement forward_with_losses(z, targets=None):
      z:        (b, t, p, d) input latent
      targets:  optional dict for auxiliary supervision; unknown keys
                are silently ignored
      Returns:  (decoded_image (b*t, 3, H, W), loss_dict)
                loss_dict keys are free-form scalars; VWorldModel
                adds them all to the total loss with weight 1.0.
    """

    @abstractmethod
    def forward_with_losses(
        self,
        z: Tensor,
        targets: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, AuxLosses]:
        """
        z:       input latent
        targets: optional supervision targets
        Returns: (decoded_image, loss_dict)
        """

    def decode(
        self,
        x: Tensor,
        meta: ModuleMeta,
    ) -> Tuple[Tensor, ModuleMeta, AuxLosses]:
        """Full framework interface."""
        recon, aux = self.forward_with_losses(x)
        return recon, meta, aux
