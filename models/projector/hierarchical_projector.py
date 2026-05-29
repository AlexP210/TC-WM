import torch
import torch.nn as nn
import math
from einops import rearrange

class HierarchicalProjector(nn.Module):
    """
    分层学习的 patch 压缩器
    196 patches -> 49 patches，384 dim -> 64 dim
    """
    def __init__(
        self,
        in_features,
        out_features,
        input_patches=196,  # 14x14
        output_patches=49,  # 7x7
        num_layers=3,
        hidden_dim=256,
        use_cross_attention=True
    ):
        super().__init__()

        self.input_patches = input_patches
        self.output_patches = output_patches
        self.in_h = self.in_w = int(math.sqrt(input_patches))  # 14
        self.out_h = self.out_w = int(math.sqrt(output_patches))  # 7

        # 输入投影
        self.input_proj = nn.Linear(in_features, hidden_dim)

        # 分层下采样编码器
        self.encoder_layers = nn.ModuleList()

        # 第一层: 14x14 -> 7x7 (4:1 reduction)
        self.encoder_layers.append(
            PatchDownsampleBlock(
                hidden_dim, hidden_dim,
                downsample_ratio=2,  # 2x2 -> 1
                use_attention=use_cross_attention
            )
        )

        # 中间层：特征细化
        for _ in range(num_layers - 1):
            self.encoder_layers.append(
                PatchRefinementBlock(
                    hidden_dim, hidden_dim,
                    use_attention=use_cross_attention
                )
            )

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, out_features)

        # Layer norm
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        """
        x: (b, t, input_patches, in_features)
        -> (b, t, output_patches, out_features)
        """
        b, t, n_in, d_in = x.shape

        # 输入投影
        x = self.input_proj(x)  # (b, t, n_in, hidden_dim)

        # Reshape to spatial format
        x = rearrange(x, 'b t (h w) d -> b t h w d', h=self.in_h, w=self.in_w)

        # 通过编码器层
        for layer in self.encoder_layers:
            x = layer(x)

        # Reshape back to patches
        x = rearrange(x, 'b t h w d -> b t (h w) d')

        # 输出投影和归一化
        x = self.output_proj(x)
        x = self.norm(x)

        return x


class PatchDownsampleBlock(nn.Module):
    """下采样 block: 减少空间分辨率"""
    def __init__(self, in_dim, out_dim, downsample_ratio=2, use_attention=True):
        super().__init__()
        self.downsample_ratio = downsample_ratio

        if use_attention:
            # 使用注意力池化下采样
            self.downsample = AttentionPooling2D(
                in_dim, out_dim,
                kernel_size=downsample_ratio,
                stride=downsample_ratio
            )
        else:
            # 使用卷积下采样
            self.downsample = nn.Conv2d(
                in_dim, out_dim,
                kernel_size=downsample_ratio,
                stride=downsample_ratio
            )

        # 特征细化
        self.refine = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        """
        x: (b, t, h, w, d)
        """
        b, t, h, w, d = x.shape

        # Reshape for conv/attention
        x = rearrange(x, 'b t h w d -> (b t) d h w')

        # 下采样
        x = self.downsample(x)  # (bt, d, h//r, w//r)

        # Reshape back
        x = rearrange(x, '(b t) d h w -> b t h w d', b=b)

        # 特征细化
        x = self.refine(x)

        return x


class PatchRefinementBlock(nn.Module):
    """特征细化 block: 保持空间尺寸，改善特征"""
    def __init__(self, dim, hidden_dim=None, use_attention=True):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2

        if use_attention:
            self.attn = SpatialSelfAttention(dim)
        else:
            self.attn = None

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """x: (b, t, h, w, d)"""

        # Self attention
        if self.attn:
            x = x + self.attn(self.norm1(x))

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class AttentionPooling2D(nn.Module):
    """2D 注意力池化层"""
    def __init__(self, in_dim, out_dim, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        # Query, Key, Value projections
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)

        self.scale = out_dim ** -0.5

    def forward(self, x):
        """
        x: (bt, in_dim, h, w)
        -> (bt, out_dim, h//stride, w//stride)
        """
        bt, d, h, w = x.shape

        # 转换为 patch format
        x = rearrange(x, 'bt d h w -> bt (h w) d')

        # 分组为 kernel_size x kernel_size 的窗口
        # 简化处理：使用平均池化的窗口
        x = rearrange(x, 'bt (h w) d -> bt h w d', h=h, w=w)

        # 按 stride 分块
        x = rearrange(x, 'bt (nh kh) (nw kw) d -> bt (nh nw) (kh kw) d',
                     kh=self.kernel_size, kw=self.kernel_size)

        # 计算注意力
        q = self.q_proj(x.mean(dim=2, keepdim=True))  # (bt, patches, 1, out_dim)
        k = self.k_proj(x)  # (bt, patches, kernel_size^2, out_dim)
        v = self.v_proj(x)  # (bt, patches, kernel_size^2, out_dim)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).squeeze(2)  # (bt, patches, out_dim)

        # Reshape to spatial
        nh = nw = h // self.stride
        out = rearrange(out, 'bt (nh nw) d -> bt d nh nw', nh=nh, nw=nw)

        return out


class SpatialSelfAttention(nn.Module):
    """空间自注意力"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """x: (b, t, h, w, d)"""
        b, t, h, w, d = x.shape

        # Reshape for attention
        x = rearrange(x, 'b t h w d -> (b t) (h w) d')

        # QKV
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'bt n (h d) -> bt h n d', h=self.num_heads), qkv)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'bt h n d -> bt n (h d)')
        out = self.proj(out)

        # Reshape back
        out = rearrange(out, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)

        return out


class HierarchicalDecoder(nn.Module):
    """
    分层上采样解码器: 49 patches -> 196 patches
    """
    def __init__(
        self,
        in_features,
        out_features,
        input_patches=49,
        output_patches=196,
        hidden_dim=256
    ):
        super().__init__()

        self.input_patches = input_patches
        self.output_patches = output_patches
        self.in_h = self.in_w = int(math.sqrt(input_patches))  # 7
        self.out_h = self.out_w = int(math.sqrt(output_patches))  # 14

        # 输入投影
        self.input_proj = nn.Linear(in_features, hidden_dim)

        # 上采样层
        self.upsample = PatchUpsampleBlock(
            hidden_dim, hidden_dim,
            upsample_ratio=2
        )

        # 特征细化
        self.refine = PatchRefinementBlock(hidden_dim, use_attention=False)

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        """
        x: (b, t, input_patches, in_features)
        -> (b, t, output_patches, out_features)
        """
        # 投影和重塑
        x = self.input_proj(x)
        x = rearrange(x, 'b t (h w) d -> b t h w d', h=self.in_h, w=self.in_w)

        # 上采样
        x = self.upsample(x)

        # 细化
        x = self.refine(x)

        # 输出
        x = rearrange(x, 'b t h w d -> b t (h w) d')
        x = self.output_proj(x)

        return x


class PatchUpsampleBlock(nn.Module):
    """上采样 block"""
    def __init__(self, in_dim, out_dim, upsample_ratio=2):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_dim, out_dim,
            kernel_size=upsample_ratio,
            stride=upsample_ratio
        )

        self.refine = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        b, t, h, w, d = x.shape

        # 转置卷积上采样
        x = rearrange(x, 'b t h w d -> (b t) d h w')
        x = self.upsample(x)
        x = rearrange(x, '(b t) d h w -> b t h w d', b=b)

        # 细化
        x = self.refine(x)

        return x


if __name__ == "__main__":
    # 测试代码
    print("🧪 Testing HierarchicalProjector and HierarchicalDecoder...")
    print("📋 Based on current VWorldModel implementation:")
    print("   - Current: (b, t, num_patches, 394) -> (b, t, num_patches, 64)")
    print("   - Target:  (b, t, 196, 394) -> (b, t, 49, 64)")

    # 参数设置 - 匹配当前实现
    batch_size = 2
    time_steps = 3
    input_patches = 196  # 14x14 DINO patches
    output_patches = 49  # 7x7 compressed patches
    in_features = 394  # 384 (visual) + 10 (proprio)
    out_features = 64  # projected features

    # 创建模型
    encoder = HierarchicalProjector(
        in_features=in_features,
        out_features=out_features,
        input_patches=input_patches,
        output_patches=output_patches,
        num_layers=3,
        hidden_dim=256,
        use_cross_attention=True
    )

    decoder = HierarchicalDecoder(
        in_features=out_features,
        out_features=in_features,
        input_patches=output_patches,
        output_patches=input_patches,
        hidden_dim=256
    )

    print(f"📊 Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"📊 Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # 测试 forward pass
    print(f"\n🔬 Testing forward pass...")
    print(f"Input shape: ({batch_size}, {time_steps}, {input_patches}, {in_features})")

    # 创建随机输入
    x = torch.randn(batch_size, time_steps, input_patches, in_features)
    print(f"✅ Input tensor created: {x.shape}")

    # Encoder forward
    try:
        encoded = encoder(x)
        print(f"✅ Encoder output: {encoded.shape}")
        assert encoded.shape == (batch_size, time_steps, output_patches, out_features)
        print(f"✅ Encoder output shape correct!")
    except Exception as e:
        print(f"❌ Encoder failed: {e}")
        exit(1)

    # Decoder forward
    try:
        decoded = decoder(encoded)
        print(f"✅ Decoder output: {decoded.shape}")
        assert decoded.shape == (batch_size, time_steps, input_patches, in_features)
        print(f"✅ Decoder output shape correct!")
    except Exception as e:
        print(f"❌ Decoder failed: {e}")
        exit(1)

    # 测试梯度流
    try:
        loss = decoded.sum()
        loss.backward()
        print(f"✅ Gradients computed successfully!")
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
        exit(1)

    print(f"\n🎉 All tests passed!")
    print(f"📈 Compression ratio: {input_patches}/{output_patches} = {input_patches/output_patches:.1f}x patches")
    print(f"📈 Feature compression: {in_features}/{out_features} = {in_features/out_features:.1f}x features")
    print(f"📈 Total compression: {(input_patches*in_features)/(output_patches*out_features):.1f}x")