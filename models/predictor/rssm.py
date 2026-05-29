"""
RSSM Predictor for DreamerV3-style world models.

Implements a Recurrent State Space Model with categorical latent variables.
Fits the BasePredictor interface: forward_with_losses() returns (z_pred, aux_losses).

Latent state = (deter, stoch):
  - deter: deterministic GRU hidden state, shape (B, deter_dim)
  - stoch: categorical stochastic variable, shape (B, num_categories * num_classes)

The predictor manages its own recurrent carry internally during forward().
Input: (B, T*N, D) packed history — same interface as ViTPredictor.
Output: (B, T*N, D) predicted features — same interface as ViTPredictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict, Optional
from einops import rearrange, repeat

from models.base import BasePredictor, AuxLosses


class RSSMPredictor(BasePredictor):
    """
    RSSM predictor with categorical latent variables (DreamerV3 style).

    Parameters
    ----------
    num_patches : int
        Number of spatial patches per frame (e.g. 196 for 14x14).
    num_frames : int
        Number of history frames (num_hist).
    dim : int
        Input feature dimension per patch (projected_dim + action_dim + proprio_dim).
    deter_dim : int
        Deterministic GRU hidden dimension.
    num_categories : int
        Number of categorical distributions in the stochastic variable.
    num_classes : int
        Number of classes per categorical distribution.
    hidden_dim : int
        Hidden dimension for MLPs.
    num_layers : int
        Number of hidden layers in MLPs.
    free_nats : float
        KL is not penalized below this threshold.
    kl_balance : float
        Weight on the dynamics KL vs representation KL.
        loss = kl_balance * KL(sg(post) || prior) + (1 - kl_balance) * KL(post || sg(prior))
    unimix : float
        Fraction of uniform mixture added to categorical for exploration.
    """

    def __init__(
        self,
        *,
        num_patches: int,
        num_frames: int,
        dim: int,
        deter_dim: int = 512,
        num_categories: int = 32,
        num_classes: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 2,
        free_nats: float = 1.0,
        kl_balance: float = 0.8,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.input_dim = dim
        self.deter_dim = deter_dim
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.stoch_dim = num_categories * num_classes
        self.hidden_dim = hidden_dim
        self.free_nats = free_nats
        self.kl_balance = kl_balance
        self.unimix = unimix

        # The RSSM operates on pooled patch features (B, T, D_input)
        # We pool patches before feeding into the recurrent model,
        # then tile back to (B, T, N, D) for output.

        # Input projection: pool patches then project to hidden
        self.input_proj = self._build_mlp(dim, hidden_dim, hidden_dim, num_layers)

        # GRU transition: h_{t+1} = GRU(h_t, [stoch_t, input_t])
        self.gru = nn.GRUCell(hidden_dim + self.stoch_dim, deter_dim)

        # Prior: p(z_t | h_t) — predict stoch from deter alone
        self.prior_mlp = self._build_mlp(deter_dim, hidden_dim, num_categories * num_classes, num_layers)

        # Posterior: q(z_t | h_t, o_t) — predict stoch from deter + observation
        self.posterior_mlp = self._build_mlp(deter_dim + hidden_dim, hidden_dim, num_categories * num_classes, num_layers)

        # Output projection: (deter + stoch) -> input_dim
        # Output is per-frame (B, D), tiled to (B, N, D) at runtime
        self.output_proj = self._build_mlp(
            deter_dim + self.stoch_dim, hidden_dim, dim, num_layers
        )

    def _build_mlp(self, in_dim, hidden_dim, out_dim, num_layers):
        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        return nn.Sequential(*layers)

    def _categorical_sample(self, logits: Tensor) -> Tensor:
        """
        Sample from categorical with straight-through gradient and optional uniform mixture.

        Args:
            logits: (B, num_categories * num_classes)
        Returns:
            stoch: (B, num_categories * num_classes) — one-hot flattened
        """
        B = logits.shape[0]
        logits = logits.view(B, self.num_categories, self.num_classes)

        # Uniform mixture for exploration
        if self.unimix > 0:
            probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / self.num_classes
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = torch.log(probs + 1e-8)

        # Straight-through Gumbel softmax (hard sample, soft gradient)
        if self.training:
            stoch = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        else:
            # Argmax at eval time
            idx = logits.argmax(dim=-1)
            stoch = F.one_hot(idx, self.num_classes).float()

        return stoch.view(B, self.stoch_dim)

    def _kl_categorical(self, post_logits: Tensor, prior_logits: Tensor) -> Tensor:
        """
        KL divergence between two categorical distributions with free nats and balancing.

        Args:
            post_logits: (B, T, num_categories * num_classes)
            prior_logits: (B, T, num_categories * num_classes)
        Returns:
            kl_loss: scalar
        """
        B, T = post_logits.shape[:2]
        post_logits = post_logits.view(B * T, self.num_categories, self.num_classes)
        prior_logits = prior_logits.view(B * T, self.num_categories, self.num_classes)

        post_probs = F.softmax(post_logits, dim=-1)
        prior_probs = F.softmax(prior_logits, dim=-1)

        # Add unimix
        if self.unimix > 0:
            uniform = torch.ones_like(post_probs) / self.num_classes
            post_probs = (1 - self.unimix) * post_probs + self.unimix * uniform
            prior_probs = (1 - self.unimix) * prior_probs + self.unimix * uniform

        post_log = torch.log(post_probs + 1e-8)
        prior_log = torch.log(prior_probs + 1e-8)

        # KL(post || prior) per category, summed over classes
        kl_per_cat = (post_probs * (post_log - prior_log)).sum(dim=-1)  # (B*T, num_categories)
        kl = kl_per_cat.sum(dim=-1)  # (B*T,)

        # Free nats: don't penalize KL below threshold
        kl = torch.clamp(kl, min=self.free_nats)

        return kl.view(B, T)

    def forward(self, x: Tensor) -> Tensor:
        """Deterministic forward (no KL loss). Used by BasePredictor.predict()."""
        z, _ = self.forward_with_losses(x)
        return z

    def forward_with_losses(self, x: Tensor) -> Tuple[Tensor, AuxLosses]:
        """
        Full forward with KL losses.

        Args:
            x: (B, T*N, D) packed history — T frames, N patches each.
        Returns:
            z_out: (B, T*N, D) predicted features (same shape as input).
            aux_losses: {"kl_loss": scalar} — KL between prior and posterior.
        """
        B = x.shape[0]
        D = self.input_dim
        N = self.num_patches
        T = x.shape[1] // N  # infer num_frames from input (may be < self.num_frames during rollout)
        assert x.shape[1] == T * N, f"Input tokens {x.shape[1]} not divisible by num_patches {N}"

        # Unpack: (B, T*N, D) -> (B, T, N, D)
        x = x.view(B, T, N, D)

        # Pool patches to get per-frame features: (B, T, D)
        x_pooled = x.mean(dim=2)

        # Project input: (B, T, D) -> (B, T, hidden_dim)
        x_proj = self.input_proj(x_pooled)

        # Initialize recurrent state
        h = torch.zeros(B, self.deter_dim, device=x.device)
        stoch = torch.zeros(B, self.stoch_dim, device=x.device)

        outputs = []
        post_logits_all = []
        prior_logits_all = []

        for t in range(T):
            # 1. GRU transition: h_t = GRU(h_{t-1}, [stoch_{t-1}, input_t])
            gru_input = torch.cat([stoch, x_proj[:, t]], dim=-1)
            h = self.gru(gru_input, h)

            # 2. Prior: p(z_t | h_t)
            prior_logits = self.prior_mlp(h)

            # 3. Posterior: q(z_t | h_t, o_t) — uses observation
            post_input = torch.cat([h, x_proj[:, t]], dim=-1)
            post_logits = self.posterior_mlp(post_input)

            # 4. Sample from posterior during training (teacher forcing)
            stoch = self._categorical_sample(post_logits)

            # 5. Output: project (deter, stoch) -> (B, D), tile to (B, N, D)
            feat = torch.cat([h, stoch], dim=-1)
            out = self.output_proj(feat)  # (B, D)
            out = out.unsqueeze(1).expand(B, N, D)  # tile across patches

            outputs.append(out)
            post_logits_all.append(post_logits)
            prior_logits_all.append(prior_logits)

        # Stack outputs: (B, T, N, D) -> (B, T*N, D)
        z_out = torch.stack(outputs, dim=1)  # (B, T, N, D)
        z_out = z_out.view(B, T * N, D)

        # Compute KL loss
        post_logits_all = torch.stack(post_logits_all, dim=1)   # (B, T, stoch_dim)
        prior_logits_all = torch.stack(prior_logits_all, dim=1)  # (B, T, stoch_dim)

        # KL balancing: dynamics loss + representation loss
        kl_dyn = self._kl_categorical(post_logits_all.detach(), prior_logits_all)  # sg(post)
        kl_rep = self._kl_categorical(post_logits_all, prior_logits_all.detach())  # sg(prior)
        kl_loss = self.kl_balance * kl_dyn.mean() + (1 - self.kl_balance) * kl_rep.mean()

        aux_losses = {"kl_loss": kl_loss}
        return z_out, aux_losses
