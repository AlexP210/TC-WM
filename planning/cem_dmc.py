"""
CEM planner for dm_control continuous-action environments (reacher, cheetah, hopper, etc.).

Key differences from cem.py:
  - Action clamping: samples clamped to normalized action bounds (derived from dataset)
  - Sigma clamping: min_std prevents premature convergence, max_std caps exploration
  - Momentum: mu update uses exponential moving average for smoother optimization
  - use_env_reward: score candidates by real-env cumulative reward (for locomotion tasks)
"""

import time
import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device


class CEMDMCPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        # DMC-specific
        action_clamp=2.5,   # clamp normalized actions to [-c, c]; ~99% of N(0,1)
        min_std=0.05,        # prevent sigma collapse
        max_std=2.0,         # cap exploration range
        momentum=0.1,        # mu update momentum (0 = no momentum, pure CEM)
        use_env_reward=False,  # use real-env reward as CEM objective (for locomotion)
        warm_start_dset=None,  # trajectory dataset for warm-starting CEM mu
        warm_start_std=0.5,    # initial std when warm-starting (smaller = exploit more)
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.action_clamp = action_clamp
        self.min_std = min_std
        self.max_std = max_std
        self.momentum = momentum
        self.use_env_reward = use_env_reward
        self.warm_start_dset = warm_start_dset
        self.warm_start_std = warm_start_std
        self._frameskip = action_dim // (action_dim // (evaluator.frameskip if evaluator else 1))

    def _sample_warm_actions(self, n_evals):
        """Sample frameskip-packed action sequences from dataset for CEM warm-start."""
        dset = self.warm_start_dset
        frameskip = self.evaluator.frameskip
        raw_action_dim = self.action_dim // frameskip
        n_raw_steps = self.horizon * frameskip  # raw timesteps needed

        mu = torch.zeros(n_evals, self.horizon, self.action_dim)
        for i in range(n_evals):
            traj_id = torch.randint(0, len(dset), (1,)).item()
            _, act, _, _ = dset[traj_id]  # act: (T, raw_action_dim), already normalized
            max_offset = act.shape[0] - n_raw_steps
            if max_offset <= 0:
                offset = 0
            else:
                offset = torch.randint(0, max_offset, (1,)).item()
            raw_act = act[offset:offset + n_raw_steps]  # (n_raw_steps, raw_action_dim)
            # Pack into frameskip-packed format: (horizon, raw_action_dim * frameskip)
            packed = raw_act.reshape(self.horizon, frameskip * raw_action_dim)
            mu[i] = packed
        return mu

    def init_mu_sigma(self, obs_0, actions=None):
        n_evals = obs_0["visual"].shape[0]

        if self.warm_start_dset is not None:
            # Always warm-start from dataset: ignore memo_actions
            mu = self._sample_warm_actions(n_evals)
            sigma = self.warm_start_std * torch.ones([n_evals, self.horizon, self.action_dim])
            return mu, sigma

        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def _score_with_env_reward(self, action, traj_idx):
        """Score action candidates by real-env cumulative reward.

        Args:
            action: (num_samples, horizon, action_dim) normalized, frameskip-packed
            traj_idx: which trajectory (env index) to use
        Returns:
            loss: (num_samples,) tensor — negative cumulative reward (CEM minimizes)
        """
        frameskip = self.evaluator.frameskip
        # Denormalize and unpack frameskip
        exec_actions = rearrange(
            action.cpu(), "n t (f d) -> n (t f) d", f=frameskip
        )
        exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()

        init_state = self.evaluator.state_0[traj_idx]
        env = self.evaluator.env.envs[traj_idx]

        rewards = np.zeros(action.shape[0])
        for s in range(action.shape[0]):
            rewards[s] = env.rollout_reward_only(init_state, exec_actions[s])

        # CEM minimizes loss, so negate reward
        return torch.tensor(-rewards, dtype=action.dtype, device=action.device)

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs_projected(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]
        self.opt_steps = self.opt_steps if self.opt_steps is not None else 20

        for i in range(self.opt_steps):
            iter_start = time.perf_counter()
            losses = []
            total_candidates = 0
            for traj in range(n_evals):
                # Sample actions and clamp to normalized bounds
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action = action.clamp(-self.action_clamp, self.action_clamp)

                total_candidates += action.shape[0]
                action[0] = mu[traj]  # include current best

                if self.use_env_reward:
                    # Score by real-env cumulative reward
                    loss = self._score_with_env_reward(action, traj)
                else:
                    # Score by latent-space objective (goal-conditioned)
                    cur_trans_obs_0 = {
                        key: repeat(
                            arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                        )
                        for key, arr in trans_obs_0.items()
                    }
                    cur_z_obs_g = {
                        key: repeat(
                            arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                        )
                        for key, arr in z_obs_g.items()
                    }
                    with torch.no_grad():
                        i_z_obses, i_zs = self.wm.rollout(
                            obs_0=cur_trans_obs_0,
                            act=action,
                        )
                    loss = self.objective_fn(i_z_obses, cur_z_obs_g)

                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())

                # Update mu with momentum
                new_mu = topk_action.mean(dim=0)
                mu[traj] = (1 - self.momentum) * new_mu + self.momentum * mu[traj]
                # Update sigma with clamping
                sigma[traj] = topk_action.std(dim=0).clamp(self.min_std, self.max_std)

            iter_duration = time.perf_counter() - iter_start
            candidates_per_sec = (
                total_candidates / iter_duration if iter_duration > 0 else float("inf")
            )
            speed_logs = {
                f"{self.logging_prefix}/candidates_per_sec": candidates_per_sec,
                f"{self.logging_prefix}/iter_duration": iter_duration,
            }
            base_log = {
                f"{self.logging_prefix}/loss": np.mean(losses),
                "step": i + 1,
            }
            base_log.update(speed_logs)
            if self.wandb_run is not None:
                self.wandb_run.log(base_log)
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update(speed_logs)
                logs.update({"step": i + 1})
                if self.wandb_run is not None:
                    self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break

        return mu, np.full(n_evals, np.inf)
