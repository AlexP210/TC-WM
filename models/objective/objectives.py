"""
Dynamics objectives for the Universal Latent Dynamics WM Framework.

Each objective implements BaseTaskObjective and returns aux_losses from compute().

Implemented here
----------------
  ConsistencyLossObjective    — MSE(z_pred_proj, z_target_proj)  [TCWM + DINO-WM]
  RewardPredictionObjective   — MLP head: latent -> reward       [DreamerV3]
  ContinuationObjective       — MLP head: latent -> done (BCE)   [DreamerV3]

Task-centric objectives (supervised by external labels) live in:
  models.objective.task.objectives
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseTaskObjective, AuxLosses
from models.meta import LatentBundle, ModuleMeta
from models.utils.criterion import symlog


# ---------------------------------------------------------------------------
# ConsistencyLossObjective  (TCWM + DINO-WM)
# ---------------------------------------------------------------------------

class ConsistencyLossObjective(BaseTaskObjective):
    """
    Latent dynamics consistency loss: MSE between the predictor's output and the
    true next-frame encoded+projected features.

    Replaces the inline ``z_loss`` in VWorldModel.forward().

    Both TCWM and DINO-WM use this as their primary dynamics prediction objective.

    Parameters
    ----------
    projected_dim : int
        Dimension of the projected features.  Used to slice the first
        ``projected_dim`` channels from z_pred and z_target (which may have
        additional action/proprio channels appended along the feature axis).
    loss_weight : float
        Scalar multiplier applied to the consistency loss before adding to total.
    """

    loss_key = "consistency_loss"

    def __init__(
        self,
        projected_dim: int = None,
        loss_weight: float = 1.0,
        stop_grad_target: bool = False,
    ):
        super().__init__()
        self.projected_dim = projected_dim
        self.loss_weight = loss_weight
        self.stop_grad_target = stop_grad_target

    def compute(self, bundle: LatentBundle, meta: ModuleMeta) -> AuxLosses:
        assert bundle.z_pred   is not None, "ConsistencyLossObjective requires bundle.z_pred"
        assert bundle.z_target is not None, "ConsistencyLossObjective requires bundle.z_target"

        # projected_dim=None: _post_bundle already sliced bundle to the supervise range
        if self.projected_dim is None:
            z_pred_proj = bundle.z_pred
            z_tgt_proj  = bundle.z_target
        else:
            z_pred_proj = bundle.z_pred[:, :, :, :self.projected_dim]    # (B, T, P, D_proj)
            z_tgt_proj  = bundle.z_target[:, :, :, :self.projected_dim]  # (B, T, P, D_proj)
        if self.stop_grad_target:
            z_tgt_proj = z_tgt_proj.detach()

        if bundle.mask is not None:
            # bundle.mask: (B,) — per-sample valid-action fraction (0..1].
            # Weight each sample's MSE by its valid fraction so tasks with
            # heavily zero-padded actions don't under-contribute to the loss.
            per_sample = F.mse_loss(z_pred_proj, z_tgt_proj, reduction="none")  # (B, T, P, D)
            w = bundle.mask.to(per_sample.dtype).view(-1, 1, 1, 1)              # (B, 1, 1, 1)
            loss = (per_sample * w).mean()
        else:
            loss = F.mse_loss(z_pred_proj, z_tgt_proj)

        return {"consistency_loss": loss}


# ---------------------------------------------------------------------------
# RewardPredictionObjective  (DreamerV3)
# ---------------------------------------------------------------------------

class RewardPredictionObjective(BaseTaskObjective):
    """
    Predict scalar reward from latent features.

    Reads bundle.z_pred (B, T, P, D) and bundle.reward (B, T).
    Pools patches, projects to scalar, computes symlog MSE against reward target.

    Only listed in method config when dataset has reward; if not listed, never instantiated.

    Parameters
    ----------
    projected_dim : int
        First projected_dim channels of z_pred are used (excludes action/proprio).
    hidden_dim : int
        Hidden dimension for the MLP head.
    num_layers : int
        Number of hidden layers.
    loss_weight : float
        Scalar multiplier for the reward loss.
    use_symlog : bool
        Apply symlog transform to target before MSE.
    """

    loss_key = "reward_pred_loss"

    def __init__(
        self,
        projected_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        loss_weight: float = 1.0,
        use_symlog: bool = True,
    ):
        super().__init__()
        self.projected_dim = projected_dim
        self.loss_weight = loss_weight
        self.use_symlog = use_symlog

        layers = []
        d = projected_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.head = nn.Sequential(*layers)

    def compute(self, bundle: LatentBundle, meta: ModuleMeta) -> AuxLosses:
        assert bundle.z_pred is not None, "RewardPredictionObjective requires bundle.z_pred"
        assert bundle.reward is not None, "RewardPredictionObjective requires bundle.reward"

        # z_pred: (B, T, P, D_full) -> slice projected -> pool patches -> (B, T, D_proj)
        z = bundle.z_pred[:, :, :, :self.projected_dim].mean(dim=2)
        pred = self.head(z).squeeze(-1)  # (B, T)

        target = bundle.reward
        if self.use_symlog:
            target = symlog(target)
            pred = symlog(pred)

        loss = F.mse_loss(pred, target)
        return {"reward_pred_loss": loss}


# ---------------------------------------------------------------------------
# ContinuationObjective  (DreamerV3)
# ---------------------------------------------------------------------------

class ContinuationObjective(BaseTaskObjective):
    """
    Predict continuation (1 - done) from latent features using BCE.

    Reads bundle.z_pred (B, T, P, D) and bundle.reward is used to infer
    episode boundaries (or a dedicated 'done' field if available).

    Only listed in method config when dataset has done/continuation signal.

    Parameters
    ----------
    projected_dim : int
        First projected_dim channels of z_pred are used.
    hidden_dim : int
        Hidden dimension for the MLP head.
    num_layers : int
        Number of hidden layers.
    loss_weight : float
        Scalar multiplier for the continuation loss.
    """

    loss_key = "continuation_loss"

    def __init__(
        self,
        projected_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.projected_dim = projected_dim
        self.loss_weight = loss_weight

        layers = []
        d = projected_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.head = nn.Sequential(*layers)

    def compute(self, bundle: LatentBundle, meta: ModuleMeta) -> AuxLosses:
        assert bundle.z_pred is not None, "ContinuationObjective requires bundle.z_pred"
        # continuation must be provided via a custom bundle field or reward-based heuristic
        # For now, require it in bundle.value as a (B, T) binary tensor
        # (the trainer sets bundle.value = continuation when this objective is active)
        assert bundle.value is not None, "ContinuationObjective requires bundle.value (continuation signal)"

        z = bundle.z_pred[:, :, :, :self.projected_dim].mean(dim=2)
        logits = self.head(z).squeeze(-1)  # (B, T)

        target = bundle.value  # (B, T), float 0/1
        loss = F.binary_cross_entropy_with_logits(logits, target)
        return {"continuation_loss": loss}
