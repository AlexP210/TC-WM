import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_features=None,
        num_layers=2,
        dropout=0.0,
        activation=nn.ReLU,
    ):
        super().__init__()

        hidden_features = hidden_features or 2 * out_features

        layers = []
        prev_dim = in_features

        for i in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_features))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_features

        layers.append(nn.Linear(prev_dim, out_features))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class LinearTemporalProjector(nn.Module):
    """
    Linear Temporal Projector - Linear projection with optional history support

    This is a wrapper around nn.Linear that can optionally receive and process
    historical information, but for now just does standard linear projection.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        use_history: Whether to accept history information (for future extension)
    """

    def __init__(self, visual_emb_dim, proprio_emb_dim, projected_dim, use_history=True, act_embed_dim=None, num_hist=3):
        super().__init__()

        self.in_features = visual_emb_dim + proprio_emb_dim
        self.out_features = projected_dim
        
        self.use_history = use_history
        self.num_hist = num_hist

        if use_history:
            self.act_embed_dim = act_embed_dim
            # Linear layer for concatenated input: x_t + flattened history
            # x_t: in_features, history: num_hist * out_features
            concat_features = self.in_features + num_hist * (self.out_features + self.act_embed_dim)
            # self.concat_projector = nn.Linear(concat_features, out_features)
            self.concat_projector = nn.Linear(concat_features, self.out_features)
            print(f"LinearTemporalProjector initialized with history:")
            print(f"  - Input features: {self.in_features}")
            print(f"  - History features: {num_hist} * {self.out_features} = {num_hist * self.out_features}")
            print(f"  - Total input: {concat_features}")
            print(f"  - Output features: {self.out_features}")
        else:
            self.concat_projector = nn.Linear(self.in_features, self.out_features)
            print(f"LinearTemporalProjector initialized without history:")
            print(f"  - Input features: {self.in_features}")
            print(f"  - Output features: {self.out_features}")

    def forward(self, visual_emb, proprio_emb, act=None):
        """
        visual_emb:   (B, T, P, visual_emb_dim)
        proprio_emb: (B, T, P, proprio_emb_dim)
        act: (B, T, P, act_embed_dim)
        return:
        z:   (B, T, P, out_features)
        """
        B, T, P, _ = visual_emb.shape
        H = self.num_hist
        device, dtype = visual_emb.device, visual_emb.dtype
        
        if self.use_history:
            assert act is not None
            z_hist = torch.zeros(B, P, H, self.out_features, device=device, dtype=dtype)
            a_hist = torch.zeros(B, P, H, self.act_embed_dim, device=device, dtype=dtype)

            z_list = []
            for t in range(T):
                hist_flat = torch.cat([z_hist, a_hist], dim=-1).reshape(B, P, -1)
                x = torch.cat([hist_flat, visual_emb[:, t], proprio_emb[:, t]], dim=-1)
                z_t = self.concat_projector(x)  # (B, P, out_features)
                z_list.append(z_t)
                if H > 0:
                    if H > 1:
                        z_hist[:, :, :-1] = z_hist[:, :, 1:].clone()
                        a_hist[:, :, :-1] = a_hist[:, :, 1:].clone()
                    z_hist[:, :, -1] = z_t
                    a_hist[:, :, -1] = act[:, t]

            z = torch.stack(z_list, dim=1)  
        else: # now we use it for no history
            z = self.concat_projector(torch.cat([visual_emb, proprio_emb], dim=-1))
        return z

    def __repr__(self):
        return (f"LinearTemporalProjector(in_features={self.in_features}, "
                f"out_features={self.out_features}, use_history={self.use_history})")


if __name__ == "__main__":
    # Test the LinearTemporalProjector
    print("🧪 Testing LinearTemporalProjector...")

    # Test without history
    projector_no_hist = LinearTemporalProjector(416, 64, use_history=False)

    # Test with history
    projector_with_hist = LinearTemporalProjector(416, 64, use_history=True)

    # Test input
    x = torch.randn(2, 3, 196, 416)  # (batch, time, patches, features)
    z_history = torch.randn(2, 3, 196, 64)  # Mock history

    print(f"\n📥 Input shape: {x.shape}")

    # Test without history
    output1 = projector_no_hist(x)
    print(f"📤 Output (no history): {output1.shape}")

    # Test with history
    output2 = projector_with_hist(x, z_history)
    print(f"📤 Output (with history): {output2.shape}")

    print("✅ LinearTemporalProjector tests passed!")