"""
MTJEPAWorldModel — Multi-Task JEPA world model.

Subclasses VWorldModel with overrides of encode() and _post_bundle():
  1. Calls super().encode() to get the standard predictor input z
  2. Looks up frozen CLIP text embedding from obs["task_id"]
  3. Projects 512D → task_proj_dim and CONCATENATEs to z (along feature dim)
  4. Applies per-task action mask to raw actions before encoding, so zero-padded
     action dimensions (from multi-task heterogeneous action spaces) don't bias
     the action encoder or predictor.

Action masking (Newt-style):
  - `_action_masks` buffer: (num_tasks, max_action_dim), 1.0 for valid dims
  - Applied to `act` in encode() before super().encode()
  - Valid fraction per sample propagated through LatentBundle.mask
  - ConsistencyLossObjective uses bundle.mask to weight per-sample MSE

obs["task_id"] is a scalar LongTensor set by MultiTaskTrajSlicerDataset.
Single-task datasets don't set this key → graceful fallback (no masking, no
task embedding).
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.meta import LatentBundle
from models.visual_world_model import VWorldModel


def _load_embeddings(task_emb_path: str) -> torch.Tensor:
    """Load CLIP text embeddings from tasks JSON.

    Returns a tensor of shape (num_tasks, emb_dim) ordered by task insertion
    order. Task order must match the task_id integers assigned by
    MultiTaskRobomimicDataset (0, 1, 2, ...).
    """
    with open(task_emb_path, "r") as f:
        tasks = json.load(f)

    embeddings = []
    for task_name, info in tasks.items():
        if "text_embedding" not in info:
            raise ValueError(
                f"Task '{task_name}' has no text_embedding. "
                f"Run: python create_task_embeddings.py"
            )
        embeddings.append(info["text_embedding"])

    return torch.tensor(embeddings, dtype=torch.float32)  # (num_tasks, 512)


class MTJEPAWorldModel(VWorldModel):
    """
    Multi-Task JEPA world model with frozen CLIP task conditioning and
    per-task action masking for heterogeneous action spaces.

    Extra __init__ args (all optional, falls back to plain VWorldModel):
        task_emb_path     (str)        path to robomimic_tasks.json with text_embeddings
        task_proj_dim     (int)        projection output dim for task embedding
        action_dims       (List[int])  per-task valid action dimensions, e.g. [7, 7, 7, 14, 7]
                                       used to build _action_masks buffer
        supervise_proprio (bool)       if True, include proprio in consistency loss target;
                                       if False, supervise visual features only (default: True)
    """

    def __init__(
        self,
        task_emb_path: str = None,
        task_proj_dim: int = None,
        action_dims: Optional[List[int]] = None,
        supervise_proprio: bool = True,
        **kwargs,
    ):
        self.supervise_proprio = supervise_proprio
        super().__init__(**kwargs)

        self.task_proj = None
        self.task_proj_dim = task_proj_dim or 0

        # --- CLIP task embeddings ---
        if task_emb_path and os.path.exists(task_emb_path):
            if not task_proj_dim:
                raise ValueError("task_proj_dim must be set when task_emb_path is provided")
            emb = _load_embeddings(task_emb_path)          # (num_tasks, 512)
            self.register_buffer("task_emb_weight", emb)   # frozen, not a parameter
            self.task_proj = nn.Linear(512, task_proj_dim)
            print(
                f"[MTJEPAWorldModel] Loaded {emb.shape[0]} task embeddings "
                f"(CLIP 512D → {task_proj_dim}D concat conditioning)"
            )
        elif task_emb_path:
            print(
                f"[MTJEPAWorldModel] Warning: task_emb_path '{task_emb_path}' not found. "
                f"Run create_task_embeddings.py first. Falling back to no task conditioning."
            )

        # --- Action masks (Newt-style) ---
        # _action_masks[t, d] = 1.0  if dim d is a valid (non-padded) action dim for task t
        #                       0.0  if dim d is zero-padded
        if action_dims is not None:
            max_dim = max(action_dims)
            masks = torch.zeros(len(action_dims), max_dim)
            for t, dim in enumerate(action_dims):
                masks[t, :dim] = 1.0
            self.register_buffer("_action_masks", masks)
            print(
                f"[MTJEPAWorldModel] Action masks: {len(action_dims)} tasks, "
                f"max_action_dim={max_dim}"
            )
            print(f"  Per-task valid dims: {action_dims}")

    # ------------------------------------------------------------------
    # encode: apply action mask → super().encode() → append task emb
    # ------------------------------------------------------------------

    def encode(self, obs, act):
        """Apply action mask → standard encode → concat task conditioning."""
        task_id = obs.get("task_id", None)

        # Action masking: zero out padded action dims before encoding.
        # act shape: (B, T, max_action_dim * frameskip) — actions are chunked per frame.
        # _action_masks shape: (num_tasks, max_action_dim).
        # We tile the per-dim mask across the frameskip factor so each action
        # in the chunk gets the same mask pattern.
        if hasattr(self, "_action_masks") and task_id is not None:
            tid = task_id if task_id.dim() > 0 else task_id.unsqueeze(0)
            mask = self._action_masks[tid]          # (B, max_action_dim)
            max_action_dim = mask.shape[1]
            act_chunk_dim = act.shape[2]            # max_action_dim * frameskip
            frameskip = act_chunk_dim // max_action_dim
            mask_tiled = mask.repeat(1, frameskip)  # (B, max_action_dim * frameskip)
            act = act * mask_tiled.unsqueeze(1)     # (B, T, max_action_dim * frameskip)

        z, z_dct = super().encode(obs, act)

        # Task embedding concat
        if not hasattr(self, "task_emb_weight"):
            return z, z_dct

        B, T, P, _ = z.shape

        if task_id is None:
            # rollout / inference without task_id — concat zeros to maintain dim
            pad = torch.zeros(B, T, P, self.task_proj_dim, device=z.device, dtype=z.dtype)
            z = torch.cat([z, pad], dim=-1)
            return z, z_dct

        tid = task_id if task_id.dim() > 0 else task_id.unsqueeze(0)
        raw_emb  = self.task_emb_weight[tid]        # (B, 512)
        task_emb = self.task_proj(raw_emb)          # (B, task_proj_dim)

        task_emb_tiled = task_emb.unsqueeze(1).unsqueeze(1).expand(B, T, P, -1)
        z = torch.cat([z, task_emb_tiled], dim=-1)  # (B, T, P, D + task_proj_dim)
        return z, z_dct

    # ------------------------------------------------------------------
    # _post_bundle: inject valid-action fraction into bundle.mask
    # ------------------------------------------------------------------

    def _post_bundle(self, bundle: LatentBundle, obs: dict) -> LatentBundle:
        """Slice bundle to supervised dims; add valid-action fraction mask.

        Feature layout: [visual(enc_dim) | proprio(proprio_dim) | action(act_dim) | task(task_proj_dim)]
        With supervise_proprio=True : supervise visual+proprio dims.
        With supervise_proprio=False: supervise visual dims only (proprio is input-only).
        """
        if self.supervise_proprio:
            supervised_dim = self.encoder.emb_dim + self.proprio_dim // self.num_proprio_repeat
        else:
            supervised_dim = self.encoder.emb_dim  # 384 visual features only
        bundle.z_pred   = bundle.z_pred[:, :, :, :supervised_dim]
        bundle.z_target = bundle.z_target[:, :, :, :supervised_dim]

        if not hasattr(self, "_action_masks"):
            return bundle
        task_id = obs.get("task_id", None)
        if task_id is None:
            return bundle
        tid = task_id if task_id.dim() > 0 else task_id.unsqueeze(0)
        valid_frac = self._action_masks[tid].mean(dim=-1)  # (B,)
        bundle.mask = valid_frac
        return bundle


class MTJEPATokenWorldModel(VWorldModel):
    """
    MT-JEPA variant: action and task_emb as independent tokens in the sequence.

    Token layout per frame: [visual(0:n_visual) | action_token(1) | task_token(1)]
    All tokens share feature dim = encoder_emb_dim (384D).

    Contrast with MTJEPAWorldModel which tiles action/task into the feature dim
    of every visual patch. Here they are dedicated sequence-level tokens with their
    own position in attention, so visual patches can explicitly attend to them.

    Extra __init__ args (same signature as MTJEPAWorldModel):
        task_emb_path  (str)        path to robomimic_tasks.json with text_embeddings
        task_proj_dim  (int)        unused (kept for API compat); projection is always → encoder_emb_dim
        action_dims    (List[int])  per-task valid action dimensions for masking
    """

    def __init__(
        self,
        task_emb_path: str = None,
        task_proj_dim: int = None,   # kept for API compat, not used
        action_dims: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Action token projection: raw action_emb_dim → encoder_emb_dim
        action_emb_raw_dim = self.action_dim // self.num_action_repeat
        self.action_token_proj = nn.Linear(action_emb_raw_dim, self.encoder.emb_dim)

        # Proprio token projection: proprio_emb_dim → encoder_emb_dim
        # Used both at encode time (input) and as prediction target (same space, no extra head)
        proprio_emb_raw_dim = self.proprio_dim // self.num_proprio_repeat
        self.proprio_token_proj = nn.Linear(proprio_emb_raw_dim, self.encoder.emb_dim)

        # Task token projection: CLIP 512D → encoder_emb_dim (384D)
        if task_emb_path and os.path.exists(task_emb_path):
            emb = _load_embeddings(task_emb_path)
            self.register_buffer("task_emb_weight", emb)
            self.task_proj = nn.Linear(512, self.encoder.emb_dim)
            print(
                f"[MTJEPATokenWorldModel] Loaded {emb.shape[0]} task embeddings "
                f"(CLIP 512D → {self.encoder.emb_dim}D token)"
            )
        elif task_emb_path:
            print(
                f"[MTJEPATokenWorldModel] Warning: task_emb_path '{task_emb_path}' not found."
            )

        # Action masks (Newt-style, same as MTJEPAWorldModel)
        if action_dims is not None:
            max_dim = max(action_dims)
            masks = torch.zeros(len(action_dims), max_dim)
            for t, dim in enumerate(action_dims):
                masks[t, :dim] = 1.0
            self.register_buffer("_action_masks", masks)
            print(
                f"[MTJEPATokenWorldModel] Action masks: {len(action_dims)} tasks, "
                f"max_action_dim={max_dim}"
            )
            print(f"  Per-task valid dims: {action_dims}")

        # n_visual is set on first encode() call
        self.n_visual = None

    # ------------------------------------------------------------------
    # encode: produce [visual(196) | action_token(1) | task_token(1)]
    # ------------------------------------------------------------------

    def encode(self, obs, act):
        """Encode to token sequence: [visual | action_token | task_token]."""
        task_id = obs.get("task_id", None)

        # Action masking (same logic as MTJEPAWorldModel)
        if hasattr(self, "_action_masks") and task_id is not None:
            tid = task_id if task_id.dim() > 0 else task_id.unsqueeze(0)
            mask = self._action_masks[tid]
            max_action_dim = mask.shape[1]
            act_chunk_dim = act.shape[2]
            frameskip = act_chunk_dim // max_action_dim
            mask_tiled = mask.repeat(1, frameskip)
            act = act * mask_tiled.unsqueeze(1)

        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)   # (B, T, action_emb_dim)

        B, T, P, D = z_dct['visual'].shape
        self.n_visual = P  # 196 for DINO ViT-S/14 @ 224px

        # Action token: project to encoder_emb_dim, add sequence dim
        act_token = self.action_token_proj(act_emb).unsqueeze(2)  # (B, T, 1, D)

        # Proprio token: project encoded proprio to encoder_emb_dim
        proprio_token = self.proprio_token_proj(z_dct['proprio']).unsqueeze(2)  # (B, T, 1, D)

        # Task token
        if hasattr(self, "task_emb_weight") and task_id is not None:
            tid = task_id if task_id.dim() > 0 else task_id.unsqueeze(0)
            raw_emb  = self.task_emb_weight[tid]         # (B, 512)
            task_emb = self.task_proj(raw_emb)           # (B, D)
            task_token = task_emb.unsqueeze(1).unsqueeze(2).expand(B, T, 1, -1)
            # Cache detached task_emb for TaskConditionedDirectVQVAEDecoder (no-op otherwise)
            if hasattr(self.decoder, 'cache_task_emb'):
                self.decoder.cache_task_emb(task_emb.detach())
        else:
            task_token = torch.zeros(B, T, 1, D, device=z_dct['visual'].device,
                                     dtype=z_dct['visual'].dtype)
            if hasattr(self.decoder, 'cache_task_emb'):
                self.decoder.cache_task_emb(None)

        # Cache task token for rollout refreshing (same embedding across all steps, (B,1,1,D))
        self._cached_task_token = task_token[:, :1, :, :].detach()

        # Concat on patch/token dim: (B, T, P+3, D)
        # layout: [visual×P | action_token | task_token | proprio_token]
        z = torch.cat([z_dct['visual'], act_token, task_token, proprio_token], dim=2)
        return z, z_dct

    # ------------------------------------------------------------------
    # separate_s_a_p: visual tokens are first n_visual, all D dims
    # ------------------------------------------------------------------

    def separate_s_a_p(self, z):
        n = self.n_visual
        # layout: [visual×n | action_token | task_token | proprio_token]
        # proprio=None: base class skips raw-space proprio loss;
        # proprio prediction in 384D token space is handled by extra_losses()
        return {"projected": z[:, :, :n, :], "action": None, "proprio": None}

    # ------------------------------------------------------------------
    # replace_actions_from_z: swap the action token in rollout
    # ------------------------------------------------------------------

    def replace_actions_from_z(self, z, act):
        act_emb   = self.encode_act(act)
        act_token = self.action_token_proj(act_emb).unsqueeze(2)  # (B, T, 1, D)
        z = z.clone()
        z[:, :, self.n_visual : self.n_visual + 1, :] = act_token
        # proprio token (n_visual+2) stays from predictor output — it evolves each step
        return z

    # ------------------------------------------------------------------
    # refresh_special_tokens: restore task token from cache each rollout step
    # ------------------------------------------------------------------

    def refresh_special_tokens(self, z_new: torch.Tensor) -> torch.Tensor:
        """Restore the task token from the cached CLIP embedding after each rollout step.
        Prevents task conditioning drift when predictor output is fed back as context."""
        if not hasattr(self, "_cached_task_token") or self._cached_task_token is None:
            return z_new
        n = self.n_visual
        z_new = z_new.clone()
        z_new[:, :, n + 1 : n + 2, :] = self._cached_task_token
        return z_new

    # ------------------------------------------------------------------
    # separate_emb: no action/proprio in feature dim
    # ------------------------------------------------------------------

    def separate_emb(self, z):
        return {"visual": z[:, :, :self.n_visual, :], "proprio": None}

    # ------------------------------------------------------------------
    # _post_bundle: slice to visual tokens + valid-action fraction mask
    # ------------------------------------------------------------------

    def _post_bundle(self, bundle: LatentBundle, obs: dict) -> LatentBundle:
        """Slice bundle to visual tokens only; add valid-action fraction mask."""
        n = self.n_visual
        # Cache full z_pred (P+3 tokens) before slicing for extra_losses()
        self._z_pred_full = bundle.z_pred  # (B, T, P+3, D)
        # ConsistencyLossObjective will see (B, T, n_visual, D) not (B, T, n_visual+3, D)
        bundle.z_pred   = bundle.z_pred[:, :, :n, :]
        bundle.z_target = bundle.z_target[:, :, :n, :]

        if not hasattr(self, "_action_masks"):
            return bundle
        task_id = obs.get("task_id", None)
        if task_id is None:
            return bundle
        tid = task_id if task_id.dim() > 0 else task_id.unsqueeze(0)
        bundle.mask = self._action_masks[tid].mean(dim=-1)  # (B,)
        return bundle

    # ------------------------------------------------------------------
    # extra_losses: proprio token prediction in joint 384D space
    # ------------------------------------------------------------------

    def extra_losses(self, bundle, obs) -> dict:
        """MSE between predicted proprio token and encoded next-frame proprio token.

        Both sides live in encoder_emb_dim (384D) via proprio_token_proj.
        Target is detached so gradients only flow through z_pred (predictor path).
        """
        weight = getattr(self, "proprio_pred_loss_weight", 0.0)
        if weight == 0.0 or not hasattr(self, "_z_pred_full"):
            return {}

        n = self.n_visual
        # Predicted proprio token: position n+2 in the full (P+3) z_pred
        z_pred_proprio = self._z_pred_full[:, :, n + 2 : n + 3, :]  # (B, T, 1, D)

        # Target: encode next-frame proprio and project to the same 384D space
        proprio_next = obs["proprio"][:, self.num_pred :, :]  # (B, T, raw_proprio_dim)
        proprio_emb_next = self.encode_proprio(proprio_next)  # (B, T, proprio_emb_dim)
        proprio_target = self.proprio_token_proj(proprio_emb_next).unsqueeze(2).detach()  # (B, T, 1, D)

        proprio_loss = F.mse_loss(z_pred_proprio, proprio_target)
        return {"proprio_pred_loss": weight * proprio_loss}
