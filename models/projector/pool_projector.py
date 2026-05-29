import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedProjectorWithPooling(nn.Module):
    """
    输入:
        z_visual:  (B, T, N, Dv)
        z_proprio: (B, T, N, Dp)   # proprio 已经 repeat 到 N 了
    输出:
        z_out:     (B, T, P, D_proj)

    逻辑:
        - 从 z_proprio 恢复 per-frame proprio（取第 0 个 patch）
        - 分别编码 visual / proprio
        - 用 proprio 做 gate，per-patch 融合
        - 在 patch 维度上池化成 per-frame latent
        - 从 per-frame latent 生成 P 个 projected patches
    """
    def __init__(self,
                 visual_emb_dim: int,
                 proprio_emb_dim: int,
                 hidden_dim: int = 256,
                 projected_dim: int = 256,
                 num_projected_patches: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projected_dim = projected_dim
        self.num_projected_patches = num_projected_patches

        # ---- visual encoder: Dv -> H (per patch) ----
        self.vis_ln = nn.LayerNorm(visual_emb_dim)
        self.vis_mlp = nn.Sequential(
            nn.Linear(visual_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ---- proprio encoder: Dp -> H (per frame) ----
        self.prop_ln = nn.LayerNorm(proprio_emb_dim)
        self.prop_mlp = nn.Sequential(
            nn.Linear(proprio_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ---- gate: from proprio (broadcast 后) ----
        self.gate = nn.Linear(hidden_dim, hidden_dim)

        # ---- frame-level head: H -> P * D_proj ----
        self.frame_head_ln = nn.LayerNorm(hidden_dim)
        self.frame_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_projected_patches * projected_dim),
        )

    def forward(self, z_visual, z_proprio):
        """
        z_visual:  (B, T, N, Dv)
        z_proprio: (B, T, N, Dp)   # 已 repeat
        """
        B, T, N, Dv = z_visual.shape
        _, _, N2, Dp = z_proprio.shape
        assert N == N2, "z_visual 和 z_proprio 的 num_patches 必须一致"

        # ---- 1) 视觉 per-patch 编码 ----
        h_vis = self.vis_mlp(self.vis_ln(z_visual))      # (B, T, N, H)

        # ---- 2) proprio per-frame 编码 ----
        # 从重复的 N 个 patch 中取一个即可（它们是一样的）
        z_prop_frame = z_proprio[:, :, 0, :]             # (B, T, Dp)
        h_prop_frame = self.prop_mlp(self.prop_ln(z_prop_frame))  # (B, T, H)

        # 广播到 N 个 patch
        h_prop = h_prop_frame.unsqueeze(2).expand(B, T, N, self.hidden_dim)  # (B, T, N, H)

        # ---- 3) gate from proprio, per-patch 融合 ----
        g = torch.sigmoid(self.gate(h_prop))             # (B, T, N, H)
        h_fuse_patches = g * h_prop + (1.0 - g) * h_vis  # (B, T, N, H)

        # ---- 4) patch 维池化，得到 per-frame latent ----
        h_frame = h_fuse_patches.mean(dim=2)             # (B, T, H)

        # ---- 5) 从 per-frame latent 生成 P 个 projected patches ----
        h_frame_norm = self.frame_head_ln(h_frame)       # (B, T, H)
        frame_tokens_flat = self.frame_head(h_frame_norm)  # (B, T, P * D_proj)

        # reshape: (B, T, P, D_proj)
        z_out = frame_tokens_flat.view(B, T, self.num_projected_patches, self.projected_dim)

        return z_out