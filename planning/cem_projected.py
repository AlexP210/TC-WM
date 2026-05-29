"""
CEM Planner for Projected Latent Representation
Specialized version that ensures goal encoding uses the same 64D projected features as rollout.
"""
import torch
import numpy as np
from einops import repeat
from planning.cem import CEMPlanner
from utils import move_to_device


class CEMProjectedPlanner(CEMPlanner):
    """
    CEM Planner modified for projected latent representation.
    Key difference: Goal encoding uses projected 64D features instead of raw 384D DINO features.
    """
    
    def __init__(self, horizon, topk, num_samples, var_scale, opt_steps, eval_every, 
                 wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run, **kwargs):
        # Call parent constructor (ignore extra kwargs like 'target')
        super().__init__(horizon, topk, num_samples, var_scale, opt_steps, eval_every, 
                        wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run)
    
    def encode_goal_projected(self, trans_obs_g):
        """
        Encode goal observation using only 64D projected visual+proprio features.
        No actions needed for goal state since we only compare projected features.
        """
        # Get raw encodings (384D visual + 32D proprio)
        z_dct = self.wm.encode_obs(trans_obs_g)
        
        # Apply projection to get 64D compressed features (visual + proprio only)
        # Follow same projection logic as in encode() for concat_dim == 1
        from einops import repeat
        proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
        z_visual_proprio = torch.cat([z_dct['visual'], proprio_tiled], dim=3)  # (b, t, patches, 384+32=416)
        z_projected = self.wm.post_concat_projection(z_visual_proprio)  # (b, t, patches, 64)
        
        # Return in dict format with correct semantic naming
        z_obs_g = {"projected": z_projected}  # 64D projected features only (no actions)
        return z_obs_g

    def plan(self, obs_0, obs_g, actions=None):
        """
        Modified plan() method that uses projected goal encoding.
        Everything else remains the same as original CEM planner.
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        
        z_obs_g = self.encode_goal_projected(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            # optimize individual instances
            for traj in range(n_evals):
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
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )
                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

        return mu, np.full(n_evals, np.inf)  # all actions are valid