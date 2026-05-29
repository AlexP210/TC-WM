import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


class CosmosCIEncoder(nn.Module):
    def __init__(
        self,
        model_id="nvidia/Cosmos-0.1-Tokenizer-CI8x8",
        image_size=112,
        torch_dtype="float16",
        patch_size=None,
        emb_dim=16,
        checkpoint_name="encoder.jit",
    ):
        super().__init__()
        self.name = "cosmos_ci"
        self.model_id = model_id
        self.image_size = image_size
        self.torch_dtype = torch_dtype
        self.base_emb_dim = emb_dim
        self.emb_dim = emb_dim
        self.checkpoint_name = checkpoint_name

        if patch_size is None:
            if "CI8x8" in model_id:
                patch_size = 8
            elif "CI16x16" in model_id:
                patch_size = 16
        self.patch_size = patch_size

        ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
        self.base_model = torch.jit.load(ckpt_path)
        self.base_model.eval()
        self.latent_ndim = 2

    def forward(self, x):
        if self.image_size is not None and x.shape[-2:] != (
            self.image_size,
            self.image_size,
        ):
            x = F.interpolate(
                x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
        if x.dtype != getattr(torch, self.torch_dtype):
            x = x.to(dtype=getattr(torch, self.torch_dtype))

        if hasattr(self.base_model, "encode"):
            latents = self.base_model.encode(x)
        else:
            latents = self.base_model(x)
        if isinstance(latents, (tuple, list)):
            latents = latents[0]
        if latents.dim() != 4:
            raise ValueError(f"Expected (B, C, H, W) latents, got shape {latents.shape}")

        b, c, h, w = latents.shape
        if c != self.emb_dim:
            raise ValueError(f"Latent channel mismatch: got {c}, expected {self.emb_dim}.")
        tokens = latents.permute(0, 2, 3, 1).reshape(b, h * w, c).float()
        return tokens
