import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class DinoV3Encoder(nn.Module):
    def __init__(self, model_id="facebook/dinov3-vitb16-pretrain-lvd1689m"):
        super().__init__()
        self.name = "dinov3"
        self.model_id = model_id

        self.processor = None
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
        except Exception:
            self.processor = None

        self.base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.base_model.eval()

        self.patch_size = getattr(self.base_model.config, "patch_size", 14)
        self.emb_dim = self.base_model.config.hidden_size
        self.latent_ndim = 2

        if self.processor is not None:
            image_mean = self.processor.image_mean
            image_std = self.processor.image_std
        else:
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
        mean = torch.tensor(image_mean).view(1, 3, 1, 1)
        std = torch.tensor(image_std).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)

    def forward(self, x):
        if x.dtype != self.image_mean.dtype:
            x = x.to(dtype=self.image_mean.dtype)
        x = (x - self.image_mean) / self.image_std
        outputs = self.base_model(pixel_values=x)
        tokens = outputs.last_hidden_state
        # Drop CLS + register tokens to keep patch tokens only.
        num_register_tokens = getattr(self.base_model.config, "num_register_tokens", 0)
        tokens = tokens[:, 1 + num_register_tokens :, :]
        return tokens
