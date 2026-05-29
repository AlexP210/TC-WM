"""
Task-centric objectives: supervised by external signals (state, reward, etc.).

These objectives encode domain knowledge about the task — they require labels
from the dataset (state, reward, …) and are typically method-specific.

Implemented
-----------
  InfoNCEAlignmentObjective   — InfoNCE: projected latent ↔ state  [TCWM]
  RewardPredictionObjective   — MLP: z_pred → reward scalar         [TDMPC2-style, optional]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from models.base import BaseTaskObjective, AuxLosses
from models.meta import LatentBundle, ModuleMeta


# ---------------------------------------------------------------------------
# InfoNCEAlignmentObjective  (TCWM)
# ---------------------------------------------------------------------------

class InfoNCEAlignmentObjective(BaseTaskObjective):
    """
    TCWM task-centric objective: align projected latent features with ground-truth
    state via InfoNCE contrastive loss.

    The alignment_projection (nn.Linear) is a learnable submodule owned by this
    objective.  Callers access it via ``objective.alignment_projection`` to set
    up an optimizer and checkpoint it separately if needed.

    Parameters
    ----------
    alignment_dim : int
        Number of leading dims from projected features used for alignment.
    proprio_dim : int
        Dimension of the proprioceptive state (read from dataset at init time).
    projected_dim : int
        Total projected feature dimension; used to slice visual dims from z_pred.
    concat_dim : int
        0 → token-axis concat (z_pred has extra proprio/action tokens at end);
        1 → feature-axis concat (first projected_dim dims are visual+proprio).
    num_pred : int
        Prediction horizon; used to align z_pred time axis with proprio targets.
    temperature : float
        InfoNCE softmax temperature.
    alignment_regularization : float
        L2 regularization coefficient on alignment_projection weights.
    loss_weight : float
        Scalar multiplier applied to state_consistency_loss before adding to total.
    """

    loss_key = "state_consistency_loss"

    def __init__(
        self,
        alignment_dim: int,
        proprio_dim: int,
        projected_dim: int,
        concat_dim: int,
        num_pred: int,
        temperature: float = 0.1,
        alignment_regularization: float = 1e-4,
        loss_weight: float = 1.0,
        # Legacy param name kept for backward compat
        state_consistency_loss_weight: Optional[float] = None,
        open_alignment: bool = True,  # kept for compat; use loss_weight=0 instead
    ):
        super().__init__()
        self.alignment_projection = nn.Linear(alignment_dim, proprio_dim)
        self.alignment_dim = alignment_dim
        self.projected_dim = projected_dim
        self.concat_dim = concat_dim
        self.num_pred = num_pred
        self.temperature = temperature
        self.alignment_regularization = alignment_regularization
        # Resolve loss_weight: prefer explicit loss_weight, fallback to legacy param
        self.loss_weight = (
            state_consistency_loss_weight if state_consistency_loss_weight is not None
            else loss_weight
        )
        self.open_alignment = open_alignment  # if False, weight effectively 0

    @property
    def effective_loss_weight(self) -> float:
        return self.loss_weight if self.open_alignment else 0.0

    @staticmethod
    def _infonce_loss(z_aligned: Tensor, z_target: Tensor, temperature: float) -> Tensor:
        z_aligned_flat = z_aligned.reshape(-1, z_aligned.shape[-1])
        z_target_flat  = z_target.reshape(-1,  z_target.shape[-1])
        z_aligned_norm = F.normalize(z_aligned_flat, dim=1)
        z_target_norm  = F.normalize(z_target_flat,  dim=1)
        logits = torch.matmul(z_aligned_norm, z_target_norm.T) / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

    def compute(self, bundle: LatentBundle, meta: ModuleMeta) -> AuxLosses:
        assert bundle.z_pred  is not None, "InfoNCEAlignmentObjective requires bundle.z_pred"
        assert bundle.proprio is not None, "InfoNCEAlignmentObjective requires bundle.proprio"

        z_pred = bundle.z_pred   # (B, T, P, D) — full predictor output
        proprio = bundle.proprio  # (B, T_total, proprio_dim)

        # Slice projected visual features depending on concatenation scheme
        if self.concat_dim == 0:
            z_visual = z_pred[:, :, :-2, :]                      # exclude extra tokens
        else:
            z_visual = z_pred[:, :, :, :self.projected_dim]       # first proj_dim features

        z_avg        = z_visual.mean(dim=2)                        # (B, T, D)
        z_align_dims = z_avg[:, :, :self.alignment_dim]           # (B, T, alignment_dim)
        proprio_tgt  = proprio[:, self.num_pred:, :]              # (B, T, proprio_dim)

        b, t, align_dim = z_align_dims.shape
        z_projected = self.alignment_projection(z_align_dims.reshape(-1, align_dim))
        z_projected = z_projected.reshape(b, t, -1)               # (B, T, proprio_dim)

        alignment_loss = self._infonce_loss(z_projected, proprio_tgt, self.temperature)
        w_reg = self.alignment_regularization * torch.sum(self.alignment_projection.weight ** 2)
        state_consistency_loss = alignment_loss + w_reg

        return {
            "state_consistency_loss": state_consistency_loss,
            "alignment_loss":         alignment_loss,
            "w_regularization":       w_reg,
        }


# ---------------------------------------------------------------------------
# RewardPredictionObjective  (TDMPC2-style, data-driven)
# ---------------------------------------------------------------------------

class RewardPredictionObjective(BaseTaskObjective):
    """
    Reward prediction head: MLP from projected latent + action → scalar reward.

    Only active when the dataset provides reward data (bundle.reward is not None).
    If reward is absent the objective returns an empty dict and adds nothing to loss.

    Parameters
    ----------
    projected_dim : int
        Projected feature dimension (input to the head after spatial averaging).
    action_dim : int
        Raw action embedding dimension.
    hidden_dim : int
        Hidden width of the 2-layer MLP reward head.
    loss_weight : float
        Scalar multiplier for the reward prediction loss.
    """

    loss_key = "reward_loss"

    def __init__(
        self,
        projected_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        loss_weight: float = 0.1,
    ):
        super().__init__()
        in_dim = projected_dim + action_dim
        self.reward_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.loss_weight = loss_weight
        self.projected_dim = projected_dim

    def compute(self, bundle: LatentBundle, meta: ModuleMeta) -> AuxLosses:
        if bundle.reward is None:
            # Dataset has no reward data — skip silently
            return {}
        assert bundle.z_pred  is not None, "RewardPredictionObjective requires bundle.z_pred"
        assert bundle.action  is not None, "RewardPredictionObjective requires bundle.action"

        # Spatial average of projected features
        z_proj = bundle.z_pred[:, :, :, :self.projected_dim]  # (B, T, P, D_proj)
        z_avg  = z_proj.mean(dim=2)                           # (B, T, D_proj)

        # Action: already (B, T, action_dim) or needs slicing from encoded action
        act = bundle.action                                    # (B, T, action_dim)

        inp = torch.cat([z_avg, act], dim=-1)                 # (B, T, D_proj + action_dim)
        B, T, D = inp.shape
        r_pred = self.reward_head(inp.reshape(B * T, D)).reshape(B, T, 1)

        r_true = bundle.reward                                 # (B, T) or (B, T, 1)
        if r_true.ndim == 2:
            r_true = r_true.unsqueeze(-1)

        reward_loss = F.mse_loss(r_pred, r_true)
        return {"reward_loss": reward_loss}
