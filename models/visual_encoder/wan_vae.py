import torch
import torch.nn as nn


class WanVAEEncoder(nn.Module):
    def __init__(
        self,
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="vae",
        torch_dtype="float16",
        sample_mode="mean",
        normalize="minus_one_to_one",
        output_dtype="float32",
        patch_size=None,
    ):
        super().__init__()
        self.name = "wan_vae"
        self.model_id = model_id
        self.subfolder = subfolder
        self.sample_mode = sample_mode
        self.normalize = normalize
        self.output_dtype = output_dtype
        self.patch_size = patch_size
        self.latent_ndim = 2

        try:
            from diffusers import AutoencoderKLWan
        except Exception as exc:
            raise ImportError(
                "WanVAEEncoder requires diffusers with AutoencoderKLWan. "
                "Install/upgrade diffusers before using this encoder."
            ) from exc

        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self.base_model = AutoencoderKLWan.from_pretrained(
            model_id, subfolder=subfolder, torch_dtype=dtype
        )
        self.base_model.eval()

        latent_channels = getattr(self.base_model.config, "latent_channels", None)
        if latent_channels is None:
            latent_channels = getattr(self.base_model.config, "z_dim", None)
        self.emb_dim = latent_channels
        self.scaling_factor = getattr(self.base_model.config, "scaling_factor", 1.0)
        if self.patch_size is None:
            self.patch_size = getattr(self.base_model.config, "scale_factor_spatial", None)

    def _normalize_input(self, x):
        if self.normalize == "minus_one_to_one":
            return x * 2.0 - 1.0
        if self.normalize == "none":
            return x
        raise ValueError(f"Unknown normalize mode: {self.normalize}")

    def _select_latent(self, latent_out):
        if hasattr(latent_out, "latent_dist"):
            dist = latent_out.latent_dist
            if self.sample_mode == "sample":
                return dist.sample()
            if self.sample_mode in ("mean", "mode"):
                return dist.mean if self.sample_mode == "mean" else dist.mode()
        if isinstance(latent_out, (tuple, list)):
            return latent_out[0]
        return latent_out

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(2)
        x = self._normalize_input(x)
        if any(p.device.type != "meta" for p in self.base_model.parameters()):
            ref = next(self.base_model.parameters())
            x = x.to(device=ref.device, dtype=ref.dtype)
        latent_out = self.base_model.encode(x)
        latents = self._select_latent(latent_out)
        if self.scaling_factor is not None:
            latents = latents * float(self.scaling_factor)
        if latents.dim() == 5 and latents.shape[2] == 1:
            latents = latents.squeeze(2)
        if latents.dim() != 4:
            raise ValueError(f"Expected latents (B, C, H, W), got {latents.shape}")

        b, c, h, w = latents.shape
        if self.emb_dim is None:
            self.emb_dim = c
        if self.patch_size is None and x.shape[-1] % w == 0:
            self.patch_size = x.shape[-1] // w

        tokens = latents.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if self.output_dtype is not None:
            out_dtype = getattr(torch, self.output_dtype) if isinstance(self.output_dtype, str) else self.output_dtype
            tokens = tokens.to(dtype=out_dtype)
        return tokens
