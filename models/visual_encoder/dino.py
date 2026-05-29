import torch
import torch.nn as nn
import torch.nn.functional as F

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

from models.base import BaseEncoder

class DinoV2Encoder(BaseEncoder):
    def __init__(self, name, feature_key, projected_dim=128):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.base_emb_dim = self.base_model.num_features
        self.projected_dim = projected_dim
        
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size
        
        # Keep old projection for backward compatibility
        self.projection = nn.Linear(self.base_emb_dim, self.projected_dim)
        
        self.emb_dim = self.base_emb_dim  # Use original 384D DINO embedding dimension

    def forward(self, x):
        emb_original = self.base_model.forward_features(x)[self.feature_key]
        
        if self.latent_ndim == 1:
            emb_original = emb_original.unsqueeze(1)  # dummy patch dim
        
        # Return original 384D DINO features (like original DINO-WM)
        return emb_original