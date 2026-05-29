import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat
from models.utils.criterion import SymlogMSELoss
from models.meta import LatentBundle, ModuleMeta
from typing import List, Optional

class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        emb_decoder,
        post_concat_projection,
        predictor,
        alignment_projection=None,   # legacy; ignored when task_objectives provided
        task_objective=None,         # single objective (legacy; prefer task_objectives)
        task_objectives=None,        # List[BaseTaskObjective] — canonical interface
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        train_decoder_grad=True,
        train_emb_decoder_grad=True,
        projected_dim=64,
        predictor_dim=None,  # Will be computed as projected_dim + action_dim
        raw_proprio_dim=0,
        alignment_dim=None,  # Number of dimensions for InfoNCE alignment (hyperparameter)
        proprio_pred_loss_type="mse",
        proprio_pred_space="raw",
        proprio_infonce_temp=0.1,
        open_alignment=False,
        action_dims=None,  # ignored in base class; consumed by MTJEPAWorldModel
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.emb_decoder = emb_decoder
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.train_decoder_grad = train_decoder_grad
        self.train_emb_decoder_grad = train_emb_decoder_grad
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.raw_proprio_dim = raw_proprio_dim
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.alignment_dim = min(self.proprio_dim, projected_dim)
        self.action_dim = action_dim * num_action_repeat 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)
        
        # Calculate concatenated dimension based on concat_dim
        if self.concat_dim == 1:
            # Feature concatenation: visual(384) + proprio(32) + action(16) = 432
            self.concat_emb_dim = self.encoder.emb_dim + self.proprio_dim + self.action_dim
        else:
            # Token concatenation: all tokens have same visual dimension (384)
            self.concat_emb_dim = self.encoder.emb_dim
        
        # Add projection layer after concatenation (configurable target dimension)  
        self.projected_dim = projected_dim
        # Projection input excludes action dimensions (visual + proprio only)
        self.projection_input_dim = self.encoder.emb_dim + self.proprio_dim  # 384 + 32 = 416
        self.post_concat_projection = post_concat_projection
        self.identity_projector = isinstance(self.post_concat_projection, nn.Identity)
        
        # Compute predictor dimension dynamically  
        self.predictor_dim = self.projected_dim + self.action_dim + self.raw_proprio_dim
        
        print(f"Concat emb_dim: {self.concat_emb_dim} (visual:{self.encoder.emb_dim} + proprio:{self.proprio_dim} + action:{self.action_dim})")
        print(f"Projection input: {self.projection_input_dim}D (visual:{self.encoder.emb_dim} + proprio:{self.proprio_dim})")
        print(f"After projection: {self.projected_dim}D (compressed visual+proprio)")
        print(f"Final predictor input: {self.predictor_dim}D ({self.projected_dim}D projected + {self.action_dim}D action + {self.raw_proprio_dim}D proprio)")
        print(f"InfoNCE alignment: {self.alignment_dim}D (first {self.alignment_dim} dims of projected features → proprio_dim from data)")

        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * self.encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.emb_criterion = nn.MSELoss()

        # Build canonical task_objectives ModuleList
        if task_objectives is not None:
            _objs = list(task_objectives) if not isinstance(task_objectives, nn.ModuleList) else list(task_objectives)
        elif task_objective is not None:
            _objs = [task_objective]
        else:
            _objs = []
        self.task_objectives = nn.ModuleList(_objs)

        # Backward-compat pointers
        self.task_objective = self.task_objectives[0] if self.task_objectives else None
        # alignment_projection: find it in whichever objective owns it
        self.alignment_projection = next(
            (getattr(o, "alignment_projection", None) for o in self.task_objectives
             if hasattr(o, "alignment_projection")),
            None,
        )
        self.state_consistency_loss_weight = 1.0
        self.alignment_regularization = 1e-4  # only used in legacy path
        self.proprio_pred_loss_weight = 1.0
        self.proprio_pred_loss_type = str(proprio_pred_loss_type).lower()
        self.proprio_pred_space = str(proprio_pred_space).lower()
        self.proprio_infonce_temp = proprio_infonce_temp
        self.emb_recon_weight = 1.0  # can be overridden via config
        self.open_alignment = open_alignment
        

    def _post_bundle(self, bundle: "LatentBundle", obs: dict) -> "LatentBundle":
        """Hook for subclasses to inject extra info (e.g. action masks) into the bundle."""
        return bundle

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        if self.proprio_encoder is not None:
            self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.emb_decoder.train(mode)
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        if self.proprio_encoder is not None:
            self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        raw_proprio = obs["proprio"]
        if self.concat_dim == 0:
            z_concat = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, 384)
            # Apply projection only to visual + proprio (exclude action): 384D -> 64D
            z_visual_proprio = torch.cat([z_dct['visual'], z_dct['proprio'].unsqueeze(2)], dim=2)
            z_projected = self.post_concat_projection(z_concat)  # (b, num_frames, num_patches + 2, 64)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_projected.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            raw_prop_tiled = repeat(
                raw_proprio.unsqueeze(2), "b t 1 a -> b t f a", f=z_projected.shape[2]
            )
            z = torch.cat([z_projected, act_repeated, raw_prop_tiled], dim=3)
            return z, z_dct
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            if self.identity_projector:
                z_projected = z_dct['visual']
                z = torch.cat([z_projected, act_repeated, proprio_repeated], dim=3)
            else:
                raw_prop_tiled = repeat(
                    raw_proprio.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct["visual"].shape[2]
                )
                z_projected = self.post_concat_projection(z_dct['visual'], proprio_repeated)
                z = torch.cat([z_projected, act_repeated, raw_prop_tiled], dim=3)
        return z, z_dct
    
    def encode_to_projected(self, obs, act):
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z_visual_proprio = torch.cat([z_dct['visual'], z_dct['proprio'].unsqueeze(2)], dim=2)
            z_projected = self.post_concat_projection(z_visual_proprio) 
            return {"projected": z_projected}
        elif self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            if self.identity_projector:
                z_projected = z_dct['visual']
                return {"projected": z_projected}
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z_projected = self.post_concat_projection(z_dct['visual'], proprio_repeated)
            z = torch.cat([z_projected, act_repeated], dim=3)
            return {"projected": z_projected}
        
    def encode_obs_projected(self, obs):
        z_dct = self.encode_obs(obs)
        # Return proprio in the SAME space as what rollout's separate_s_a_p gives:
        #   identity_projector (DINO-WM): z stores encoded proprio → return encoded
        #   MLP projector (TCWM): z stores raw proprio → return raw
        if self.identity_projector:
            proprio = z_dct['proprio']          # (B, T, encoded_proprio_dim)
        else:
            proprio = obs['proprio']            # (B, T, raw_proprio_dim)
        if self.concat_dim == 0:
            z_visual_proprio = torch.cat([z_dct['visual'], z_dct['proprio'].unsqueeze(2)], dim=2)
            z_projected = self.post_concat_projection(z_visual_proprio)
            return {"projected": z_projected, "proprio": proprio}
        elif self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            if self.identity_projector:
                z_projected = z_dct['visual']
                return {"projected": z_projected, "proprio": proprio}
            z_projected = self.post_concat_projection(z_dct['visual'], proprio_repeated)
            return {"projected": z_projected, "proprio": proprio}
        
    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs, project=False):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
                predictor_aux_losses: dict (empty for deterministic predictors, {"kl_loss": ...} for RSSM)
        """
        z_projected = z[:, :, :, :self.projected_dim]
        z_action = z[:, :, :, self.projected_dim:]
        T = z.shape[1]
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z, predictor_aux_losses = self.predictor.forward_with_losses(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z, predictor_aux_losses

    # def decode(self, z_concat):
    #     """
    #     input :   z: (b, num_frames, num_patches, emb_dim) - should be 64D projected features
    #     output: obs: (b, num_frames, 3, img_size, img_size)
        
    #     """
    #     b, num_frames, num_patches, concat_dim = z_concat.shape
        
    #     z_projected = z_concat[:, :, :, :self.projected_dim]
    #     z_action = z_concat[:, :, :, self.projected_dim:]
        
    #     visual, diff = self.decoder(z_projected) 
    #     visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
    #     obs = {
    #         "visual": visual,
    #         "action": z_action
    #     }
    
    #     return obs, diff
    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs = {"visual": z}
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    # def decode_obs(self, z_obs):
    #     """
    #     input :   z: (b, num_frames, num_patches, emb_dim) - 80D features [64D projected + 16D action]
    #     output: obs: (b, num_frames, 3, img_size, img_size)
    #     """
    #     b, num_frames, num_patches, emb_dim = z_obs["projected"].shape
    #     visual, diff = self.decoder(z_obs["projected"])  
    #     visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
    #     obs = {
    #         "visual": visual
    #     }
    #     return obs, diff
    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            # "proprio": z_obs["proprio"], 
        }
        return obs, diff
    
    # def separate_emb(self, z):
    #     """
    #     input: z (tensor)
    #     output: z_obs (dict), z_act (tensor)
    #     """
    #     if self.concat_dim == 0:
    #         z_projected, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
    #     elif self.concat_dim == 1:
    #         z_projected, z_act = z[..., :-self.action_dim], z[..., -self.action_dim:]
    #         # remove tiled dimensions
    #         z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
    #     z_obs = {"projected": z_projected}
    #     return z_obs, z_act
    # def separate_emb(self, z):
    #     """
    #     input: z (tensor)
    #     output: z_obs (dict), z_act (tensor)
    #     """
    #     if self.concat_dim == 0:
    #         z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
    #     elif self.concat_dim == 1:
    #         z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
    #                                      z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
    #                                      z[..., -self.action_dim:]
    #         # remove tiled dimensions
    #         z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
    #         z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
    #     z_obs = {"visual": z_visual, "proprio": z_proprio}
    #     return z_obs, z_act
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), 
        """
        if self.concat_dim == 0:
            z_visual, z_proprio = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio = z[..., :-self.proprio_dim], \
                                         z[..., -self.proprio_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs
    
    def separate_s_a_p(self, z):
        """
        input: z_pred (tensor)
        output: state_pred (tensor), action_pred (tensor), proprio_pred (tensor)
        """
        state_pred = z[:, :, :, :self.projected_dim]
        action_end = self.projected_dim + self.action_dim
        action_pred = z[:, :, :, self.projected_dim:action_end]
        proprio_pred = z[:, :, :, action_end:action_end + self.raw_proprio_dim]
        return {"projected": state_pred, "action": action_pred, "proprio": proprio_pred}

    def infonce_loss(self, z_aligned, z_target, temperature=0.1):
        z_aligned_flat = z_aligned.reshape(-1, z_aligned.shape[-1])  # (b*num_hist, 7)
        z_target_flat = z_target.reshape(-1, z_target.shape[-1])    # (b*num_hist, 7)
        z_aligned_norm = torch.nn.functional.normalize(z_aligned_flat, dim=1)
        z_target_norm = torch.nn.functional.normalize(z_target_flat, dim=1)
        logits = torch.matmul(z_aligned_norm, z_target_norm.T) / temperature  # (N, N)
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        loss_ce = torch.nn.functional.cross_entropy(logits, labels)
        return loss_ce
                
    def forward(self, obs, act, state=None):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
                state: (b, num_frames, proprio_dim) - proprioceptive state for alignment loss
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        
        z, z_dct = self.encode(obs, act)  
        
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, 64)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, 64)
        
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)
        
        if self.predictor is not None:
            z_pred, predictor_aux_losses = self.predict(z_src)
            for k, v in predictor_aux_losses.items():
                loss = loss + v
                loss_components[k] = v
            if self.decoder is not None:
                b, num_frames, num_patches, emb_dim = z_pred.shape
                z_projected = self.separate_s_a_p(z_pred)['projected']
                visual_emb_tgt = z_dct['visual'][:, self.num_pred :, ...]

                # Single unified call — decoder owns all loss computation.
                dec_out, dec_losses = self.decoder.forward_with_losses(
                    z_projected,
                    targets={"visual_emb": visual_emb_tgt, "image": visual_tgt, "action": act[:, :self.num_hist]},
                )
                visual_pred = rearrange(dec_out, "(b t) c h w -> b t c h w", b=b)

                recon_loss_pred = dec_losses.get("recon_loss", torch.zeros(1, device=z_projected.device))
                emb_recon_loss_pred = dec_losses.get("emb_recon_loss", torch.zeros(1, device=z_projected.device))
                vq_loss_pred = dec_losses.get("vq_loss", torch.zeros(1, device=z_projected.device))

                decoder_loss_pred = recon_loss_pred + self.emb_recon_weight * emb_recon_loss_pred + vq_loss_pred
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["emb_decoder_recon_loss_pred"] = emb_recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = vq_loss_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # --- Objectives loop (ConsistencyLoss + any others) ---
            bundle = LatentBundle(
                action=act,
                z_pred=z_pred,
                z_target=z_tgt,
                proprio=state,
            )
            bundle = self._post_bundle(bundle, obs)
            _meta = ModuleMeta()
            for obj in self.task_objectives:
                obj_losses = obj.compute(bundle, _meta)
                for k, v in obj_losses.items():
                    loss_components[k] = v
                # Primary loss: use obj.effective_loss_weight if available, else obj.loss_weight
                primary_key = obj.loss_key
                if primary_key in obj_losses:
                    w = getattr(obj, "effective_loss_weight", obj.loss_weight)
                    loss = loss + w * obj_losses[primary_key]

            # Legacy fallback: compute z_loss inline if no objectives defined
            if not self.task_objectives:
                z_pred_projected = self.separate_s_a_p(z_pred)['projected']
                z_tgt_projected  = self.separate_s_a_p(z_tgt)['projected']
                z_loss = self.emb_criterion(z_pred_projected, z_tgt_projected)
                loss = loss + z_loss
                loss_components["z_predicted_loss"] = z_loss

            loss = loss + decoder_loss_pred if self.decoder is not None else loss

            # Model-specific extra losses (e.g. proprio prediction for token models)
            for k, v in self.extra_losses(bundle, obs).items():
                loss = loss + v
                loss_components[k] = v

            if self.raw_proprio_dim > 0 and self.proprio_pred_loss_weight > 0:
                z_pred_parts = self.separate_s_a_p(z_pred)
                proprio_pred = z_pred_parts["proprio"]
                if proprio_pred is None:
                    pass  # token models handle proprio via extra_losses()
                else:
                    # Target: proprio embedding from z_tgt (same position as z_pred)
                    z_tgt_parts = self.separate_s_a_p(z_tgt)
                    proprio_tgt = z_tgt_parts["proprio"]
                    if proprio_tgt is not None:
                        if self.concat_dim == 0:
                            proprio_pred = proprio_pred[:, :, -2, :]
                            proprio_tgt = proprio_tgt[:, :, -2, :]
                        else:
                            proprio_pred = proprio_pred.mean(dim=2)
                            proprio_tgt = proprio_tgt.mean(dim=2)
                        proprio_tgt = proprio_tgt.detach()
                        if self.proprio_pred_loss_type == "infonce":
                            proprio_loss = self.infonce_loss(
                                proprio_pred, proprio_tgt, temperature=self.proprio_infonce_temp
                            )
                        else:
                            proprio_loss = self.emb_criterion(proprio_pred, proprio_tgt)
                        loss = loss + self.proprio_pred_loss_weight * proprio_loss
                        loss_components["proprio_pred_loss"] = proprio_loss
            
            # latent_dynamics_loss (legacy extra term, kept if configured)
            if hasattr(self, 'latent_dynamics_loss_weight') and self.latent_dynamics_loss_weight > 0:
                if self.concat_dim == 0:
                    z_visual_src = z_src[:, :, :-2, :self.projected_dim]
                else:
                    z_visual_src = z_src[:, :, :, :self.projected_dim]
                z_visual_pred = z_pred[:, :, :, :self.projected_dim]
                z_src_avg  = z_visual_src.mean(dim=2)
                z_pred_avg = z_visual_pred.mean(dim=2)
                dynamics_loss = torch.mean((z_pred_avg - z_src_avg) ** 2)
                loss = loss + self.latent_dynamics_loss_weight * dynamics_loss
                loss_components["dynamics_projected_loss"] = dynamics_loss

            
            # DINO feature reconstruction loss (384D -> 128D -> 384D)
            if hasattr(self.encoder, 'recon_dino_loss') and self.encoder.recon_dino_loss: # not used
                # Get DINO reconstruction loss from encoder
                dino_recon_loss = getattr(self.encoder, 'last_recon_loss', None)
                if dino_recon_loss is not None and hasattr(self, 'dino_recon_loss_weight'):
                    loss = loss + self.dino_recon_loss_weight * dino_recon_loss
                    loss_components["dino_recon_loss"] = dino_recon_loss
            
        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None:
            b_recon = z.shape[0]
            z_proj_recon = self.separate_s_a_p(z)['projected']
            dec_out_r, dec_aux_r = self.decoder.forward_with_losses(
                z_proj_recon,
                targets={"visual_emb": z_dct['visual'], "image": obs['visual'], "action": act},
            )
            visual_reconstructed = rearrange(dec_out_r, "(b t) c h w -> b t c h w", b=b_recon)
            recon_loss_reconstructed = dec_aux_r.get("recon_loss", torch.zeros(1, device=z_proj_recon.device))
            emb_recon_loss_reconstructed = dec_aux_r.get("emb_recon_loss", torch.zeros(1, device=z_proj_recon.device))
            vq_loss_reconstructed = dec_aux_r.get("vq_loss", torch.zeros(1, device=z_proj_recon.device))

            decoder_loss_reconstructed = recon_loss_reconstructed + self.emb_recon_weight * emb_recon_loss_reconstructed + vq_loss_reconstructed
            loss_components["decoder_recon_loss_reconstructed"] = recon_loss_reconstructed
            loss_components["emb_decoder_recon_loss_reconstructed"] = emb_recon_loss_reconstructed
            loss_components["decoder_vq_loss_reconstructed"] = vq_loss_reconstructed
            loss_components["decoder_loss_reconstructed"] = decoder_loss_reconstructed
            loss = loss + decoder_loss_reconstructed
            if hasattr(self.post_concat_projection, 'kl_loss'):
                kl = self.post_concat_projection.kl_loss  # scalar mean KL
                loss = loss + 1 * kl
                loss_components["projector_kl_loss"] = kl
        else:
            visual_reconstructed = None
            
        # Add visualization data to loss_components for three-way reconstruction comparison
        loss_components["visual_tgt"] = visual_tgt
        loss_components["visual_pred"] = visual_pred  
        loss_components["visual_reconstructed"] = visual_reconstructed
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def refresh_special_tokens(self, z_new: torch.Tensor) -> torch.Tensor:
        """Hook for subclasses to refresh fixed special tokens (e.g. task token) each rollout step.
        Default: no-op. Override in subclasses without touching VWorldModel rollout logic."""
        return z_new

    def extra_losses(self, bundle, obs) -> dict:
        """Hook for model-specific losses not covered by task_objectives.
        Returns {loss_name: weighted_loss_value}. Default: no-op."""
        return {}

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            action_end = self.projected_dim + self.action_dim
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[:, :, :, self.projected_dim:action_end] = act_repeated
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            action_end = self.projected_dim + self.action_dim
            z[..., self.projected_dim:action_end] = act_repeated
        return z


    def rollout_original(self, obs_0, act):
        """
        Original rollout for concatenated (not mixed) features.
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z, _ = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred, _ = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred, _ = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses = self.separate_emb(z)
        return z_obses, z

    def rollout(self, obs_0, act):
        """
        Rollout following original DINO WM strategy with 80D features and action replacement.
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                z_obses: dict with full 80D features for compatibility
                z: (b, t+n+1, num_patches, emb_dim) - 80D features [64D projected + 16D action]
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z, _ = self.encode(obs_0, act_0)  # Returns 80D features [64D projected + 16D action]
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred, _ = self.predict(z[:, -self.num_hist :])  # 80D → 80D prediction
            z_new = z_pred[:, -inc:, ...]
            # Following original DINO WM: replace action portion with planned actions
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z_new = self.refresh_special_tokens(z_new)  # hook: no-op by default
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred, _ = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_dct = self.separate_s_a_p(z)

        z_emb = self.emb_decoder(z_dct['projected'])

        # Return dict with 'visual' key so decode_obs() and evaluator metrics work;
        # also keep 'projected' for planners that use it (e.g. CEM objective).
        # 'proprio' (de-tiled, one patch) for planning objectives that use alpha.
        proprio = z_dct['proprio'][:, :, 0, :] if z_dct['proprio'] is not None else None
        return {'visual': z_emb, 'projected': z_dct['projected'], 'proprio': proprio}, z
