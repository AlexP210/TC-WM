import torch
from torch import nn
from torch.nn import functional as F

import sys
sys.path.append('..')
import distributed_fn as dist_fn
from einops import rearrange

from models.utils.resblocks import ResBlock

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 8:
            # 8x upsampling: stride=2, stride=2, stride=2 (2x2x2 = 8)
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 2, channel // 4, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 4, out_channel, 4, stride=2, padding=1),
                ]
            )
        elif stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

from models.base import BaseDecoder

class VQVAE(BaseDecoder):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        emb_dim=64,
        n_embed=512,
        decay=0.99,
        quantize=True,
        upsample_stride=4,  # 可配置的上采样rate
        decode_stride=4,    # 可配置的decode rate
    ):
        super().__init__()

        self.quantize = quantize
        self.quantize_b = Quantize(emb_dim, n_embed)

        if not quantize:
            for param in self.quantize_b.parameters():
                param.requires_grad = False

        self.upsample_b = Decoder(emb_dim, emb_dim, channel, n_res_block, n_res_channel, stride=upsample_stride)
        self.dec = Decoder(
            emb_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=decode_stride,
        )
        self.info = f"in_channel: {in_channel}, channel: {channel}, n_res_block: {n_res_block}, n_res_channel: {n_res_channel}, emb_dim: {emb_dim}, n_embed: {n_embed}, decay: {decay}"

    def forward(self, input):
        '''
            input: (b, t, num_patches, emb_dim)
        '''
        num_patches = input.shape[2]
        num_side_patches = int(num_patches ** 0.5)    
        input = rearrange(input, "b t (h w) e -> (b t) h w e", h=num_side_patches, w=num_side_patches)

        if self.quantize:
            quant_b, diff_b, id_b = self.quantize_b(input)
        else:
            quant_b, diff_b = input, torch.zeros(1).to(input.device)

        quant_b = quant_b.permute(0, 3, 1, 2).contiguous()
        diff_b = diff_b.unsqueeze(0)
        dec = self.decode(quant_b)
        return dec, diff_b # diff is 0 if no quantization

    def decode(self, quant_b):
        quant_b = quant_b.contiguous()
        # Hopper GPUs occasionally hit cuDNN GET kernel selection failures for these shapes.
        # Temporarily disabling cuDNN here keeps inference stable with minimal overhead.
        with torch.backends.cudnn.flags(enabled=False):
            upsample_b = self.upsample_b(quant_b)
            upsample_b = upsample_b.contiguous()
            dec = self.dec(upsample_b) # quant: (128, 64, 64)
        return dec

    def forward_with_losses(self, z, targets=None):
        dec, diff = self.forward(z)
        losses = {}
        if self.quantize:
            losses["vq_loss"] = diff
        return dec, losses

    def decode_code(self, code_b): # not used (only used in sample.py in original repo)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        dec = self.decode(quant_b)
        return dec

from models.decoder.emb_decoder_mlp import EmbDecoderMLP, ActionConditionedEmbDecoderMLP


class DirectVQVAEDecoder(VQVAE):
    """
    DINO-WM style: z(emb_dim) → [optional sg] → VQVAE → image.

    sg_input=True (default): detach z before VQVAE (frozen encoder path).
    sg_input=False: full gradient flow (trainable encoder path).
    """

    def __init__(self, sg_input: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.sg_input = sg_input

    def forward_with_losses(self, z, targets=None):
        import torch.nn.functional as F
        z_in = z.detach() if self.sg_input else z
        dec, diff = super().forward(z_in)
        losses = {"vq_loss": diff}
        if targets is not None and "image" in targets:
            img_tgt = targets["image"]
            if img_tgt.ndim == 5:                          # (b, t, c, h, w)
                b, t, c, h, w = img_tgt.shape
                img_tgt = img_tgt.reshape(b * t, c, h, w)
            losses["recon_loss"] = F.mse_loss(dec, img_tgt)
        return dec, losses

    def forward(self, z):
        dec, aux = self.forward_with_losses(z)
        return dec, aux.get("vq_loss", torch.zeros(1, device=z.device))


class ProjectedVQVAEDecoder(BaseDecoder):
    """
    TCWM/TDMPC2 style: z_proj(in_dim) → [optional sg] → emb_decoder(MLP) → [optional sg] → VQVAE → image.

    sg_before_emb_decoder (default False):
      False (current TCWM): emb_recon_loss flows back to predictor through emb_decoder.
      True  (old-style TCWM): z detached before emb_decoder — NO decoder grads reach predictor.
    sg_before_vqvae (default True):
      True: pixel loss stays inside VQVAE — emb_decoder trained only via emb_recon_loss.
      False (TDMPC2): full grad; pixel loss drives emb_decoder and beyond.

    forward_with_losses returns z_emb in aux so VWorldModel can compute emb_recon_loss.
    """

    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 384,
        sg_before_vqvae: bool = True,
        sg_before_emb_decoder: bool = False,
        hidden_dim: int = 512,
        num_layers: int = 2,
        channel: int = 64,
        n_res_block: int = 4,
        n_res_channel: int = 128,
        n_embed: int = 2048,
        quantize: bool = False,
    ):
        super().__init__()
        self.emb_decoder_mlp = EmbDecoderMLP(in_dim, emb_dim, hidden_dim, num_layers)
        self.vqvae = VQVAE(
            emb_dim=emb_dim,
            channel=channel,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            n_embed=n_embed,
            quantize=quantize,
        )
        self.sg_before_vqvae = sg_before_vqvae
        self.sg_before_emb_decoder = sg_before_emb_decoder

    def forward_with_losses(self, z, targets=None):
        """
        z:       (b, t, p, in_dim)
        targets: optional dict with any of:
            "visual_emb": (b, t, p, emb_dim) — emb_recon_loss (z_emb vs DINO features)
            "image":      (b, t, 3, H, W)    — recon_loss (decoded image vs target image)
        Returns: (dec (b*t, 3, H, W), loss_dict)
          loss_dict keys:
            "vq_loss"        — VQ-VAE commitment / codebook loss (only if quantize=True)
            "emb_recon_loss" — z_emb vs DINO target  (if targets["visual_emb"] provided)
            "recon_loss"     — decoded image vs target image (if targets["image"] provided)
        """
        z_in = z.detach() if self.sg_before_emb_decoder else z
        z_emb = self.emb_decoder_mlp(z_in)
        z_vqvae = z_emb.detach() if self.sg_before_vqvae else z_emb
        dec, diff = self.vqvae(z_vqvae)
        losses = {}
        if self.vqvae.quantize:
            losses["vq_loss"] = diff
        if targets is not None:
            if "visual_emb" in targets:
                losses["emb_recon_loss"] = F.mse_loss(z_emb, targets["visual_emb"])
            if "image" in targets:
                target_flat = rearrange(targets["image"], "b t c h w -> (b t) c h w")
                losses["recon_loss"] = F.mse_loss(dec, target_flat)
        return dec, losses

    def forward(self, z):
        dec, aux = self.forward_with_losses(z)
        return dec, aux.get("vq_loss", torch.zeros(1, device=z.device))


class ActionConditionedProjectedVQVAEDecoder(BaseDecoder):
    """
    ACG-WM decoder: z_proj(in_dim) + action → ActionConditionedEmbDecoderMLP → [sg] → VQVAE → image.

    Same as ProjectedVQVAEDecoder but the emb_decoder is conditioned on action.
    targets dict must include "action": (b, t, action_dim).
    """

    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        emb_dim: int = 384,
        sg_before_vqvae: bool = True,
        sg_before_emb_decoder: bool = False,
        hidden_dim: int = 512,
        num_layers: int = 2,
        channel: int = 64,
        n_res_block: int = 4,
        n_res_channel: int = 128,
        n_embed: int = 2048,
        quantize: bool = False,
        shift_action: bool = True,
    ):
        super().__init__()
        self.emb_decoder_mlp = ActionConditionedEmbDecoderMLP(
            in_dim=in_dim, action_dim=action_dim, out_dim=emb_dim,
            hidden_dim=hidden_dim, num_layers=num_layers,
            shift_action=shift_action,
        )
        self.vqvae = VQVAE(
            emb_dim=emb_dim,
            channel=channel,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            n_embed=n_embed,
            quantize=quantize,
        )
        self.sg_before_vqvae = sg_before_vqvae
        self.sg_before_emb_decoder = sg_before_emb_decoder

    def forward_with_losses(self, z, targets=None):
        """
        z:       (b, t, p, in_dim)
        targets: optional dict, decoder picks what it needs:
            "action":     (b, t, action_dim)  — used for action-conditioned emb decode;
                          falls back to zero vector if absent (e.g. rollout)
            "visual_emb": (b, t, p, emb_dim) — emb_recon_loss
            "image":      (b, t, 3, H, W)    — recon_loss
        """
        targets = targets or {}
        b, t = z.shape[:2]
        if "action" in targets:
            action = targets["action"]
        else:
            action = torch.zeros(b, t, self.emb_decoder_mlp.action_dim,
                                 device=z.device, dtype=z.dtype)

        z_in = z.detach() if self.sg_before_emb_decoder else z
        z_emb = self.emb_decoder_mlp(z_in, action)
        z_vqvae = z_emb.detach() if self.sg_before_vqvae else z_emb
        dec, diff = self.vqvae(z_vqvae)
        losses = {}
        if self.vqvae.quantize:
            losses["vq_loss"] = diff
        if "visual_emb" in targets:
            losses["emb_recon_loss"] = F.mse_loss(z_emb, targets["visual_emb"])
        if "image" in targets:
            target_flat = rearrange(targets["image"], "b t c h w -> (b t) c h w")
            losses["recon_loss"] = F.mse_loss(dec, target_flat)
        return dec, losses

    def forward(self, z, action=None):
        if action is None:
            # Fallback for decode_obs / rollout visualization: use zero action
            b, t, p, _ = z.shape
            action_dim = self.emb_decoder_mlp.action_dim
            action = torch.zeros(b, t, action_dim, device=z.device, dtype=z.dtype)
        dec, aux = self.forward_with_losses(z, targets={"action": action})
        return dec, aux.get("vq_loss", torch.zeros(1, device=z.device))


class TaskConditionedDirectVQVAEDecoder(DirectVQVAEDecoder):
    """
    MTJEPAToken setting only: DirectVQVAEDecoder + additive task embedding conditioning.

    MTJEPATokenWorldModel.encode() calls self.decoder.cache_task_emb(task_emb)
    with task_emb of shape (B, emb_dim), detached from the predictor graph.
    On each forward_with_losses() call the cached emb is projected by task_fusion and
    added to z (after sg_input detach), then the standard VQVAE pipeline runs.

    Gradient flow:
      recon_loss → dec → z_conditioned → task_cond → task_fusion  (decoder-side params)
      predictor features are stop-grad'd via sg_input — no decoder grads reach predictor.
      task_proj in MTJEPATokenWorldModel is detached at cache time → clean separation.
    """

    def __init__(self, emb_dim: int = 384, **kwargs):
        super().__init__(emb_dim=emb_dim, **kwargs)
        self.task_fusion = nn.Linear(emb_dim, emb_dim)
        self._cached_task_emb = None  # (B, emb_dim) or None; set by cache_task_emb()

    def cache_task_emb(self, task_emb):
        """Called by MTJEPATokenWorldModel.encode() before the decoder is invoked."""
        self._cached_task_emb = task_emb  # None clears the cache (no task_id branch)

    def forward_with_losses(self, z, targets=None):
        import torch.nn.functional as F
        # Apply stop-grad on predictor features (same semantics as parent sg_input)
        z_in = z.detach() if self.sg_input else z

        # Additive task conditioning (after detach so task_fusion gets recon gradients)
        if self._cached_task_emb is not None:
            B, T, P, D = z_in.shape
            task_cond = self.task_fusion(self._cached_task_emb)         # (B, D)
            z_in = z_in + task_cond.unsqueeze(1).unsqueeze(2).expand(B, T, P, D)

        # Run VQVAE directly (bypass parent forward_with_losses to avoid double-detach)
        dec, diff = VQVAE.forward(self, z_in)
        losses = {"vq_loss": diff}
        if targets is not None and "image" in targets:
            img_tgt = targets["image"]
            if img_tgt.ndim == 5:
                b, t, c, h, w = img_tgt.shape
                img_tgt = img_tgt.reshape(b * t, c, h, w)
            losses["recon_loss"] = F.mse_loss(dec, img_tgt)
        return dec, losses


class Emb_Decoder(nn.Module):
    """
    Decode patch embeddings (b, t, p, dim_in) to higher-res patch embeddings
    (b, t, n_patches_out, dim_out), where both the spatial grid (patch count)
    and embedding dim are "upsampled".

    Args:
        in_dim:      输入 embedding 维度 dim_in
        out_dim:     输出 embedding 维度 dim_out
        channel:     中间通道数（跟你原来 Decoder 的 channel 一样用法）
        n_res_block: 残差块个数
        n_res_channel: 残差块里间接通道数
        stride:      空间上采样倍率（2 / 4 / 8），决定 patch 数量放大倍数
                     n_patches_out = n_patches_in * (stride ** 2)
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        stride=4,
        keep_num_patches=False,
    ):
        super().__init__()

        self.stride = stride
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.keep_num_patches = keep_num_patches

        # 直接复用你上面定义好的 Decoder：
        # 输入通道 in_dim，输出通道 out_dim，空间上采样由 stride 决定
        self.dec = Decoder(
            in_channel=in_dim,
            out_channel=out_dim,
            channel=channel,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            stride=stride,
        )

    def forward(self, x):
        """
        x: (b, t, n_patches_in, in_dim)
        return: (b, t, n_patches_out, out_dim)
        其中 n_patches_out = (h_in * stride) * (w_in * stride)
                          = n_patches_in * (stride ** 2)
        """
        b, t, n_patches, dim = x.shape
        assert dim == self.in_dim, f"Expected in_dim={self.in_dim}, got {dim}"

        # 假设 patch 是正方形网格：n_patches = h * w
        num_side = int(n_patches ** 0.5)
        assert num_side * num_side == n_patches, \
            f"n_patches={n_patches} must be square, like 8x8=64"

        # (b, t, (h w), c) -> (b*t, c, h, w)，方便走 Conv/ConvTranspose
        x = rearrange(x, "b t (h w) c -> (b t) c h w", h=num_side, w=num_side)

        with torch.backends.cudnn.flags(enabled=False):
            y = self.dec(x)  # (b*t, out_dim, h_out, w_out)

        _, c_out, h_out, w_out = y.shape
        assert c_out == self.out_dim

        if self.keep_num_patches and (h_out != num_side or w_out != num_side):
            y = F.interpolate(
                y, size=(num_side, num_side), mode="bilinear", align_corners=False
            )
            h_out = w_out = num_side

        # 展平成新的 patch 序列：(b*t, c_out, h_out, w_out) -> (b, t, (h_out*w_out), c_out)
        y = rearrange(y, "(b t) c h w -> b t (h w) c", b=b, t=t)

        return y
    
class linear_decoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        return self.linear(x)
