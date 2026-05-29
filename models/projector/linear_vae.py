import torch
import torch.nn as nn
import torch.nn.functional as F

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

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_features))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_features

        layers.append(nn.Linear(prev_dim, out_features))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class VAETemporalProjector(nn.Module):
    """
    Variational Temporal Projector
    - Same I/O signature as your LinearTemporalProjector.
    - Ignores history if use_history=False (recommended based on your note).
    - Produces z by sampling with reparameterization from N(mu, sigma^2).
    - Stores KL divergence to standard Normal in `self.kl_loss` (mean over elements)
      and `self.kl_per_elem` (B, T, P) for loss bookkeeping.

    Args:
        in_features:    input feature dim
        out_features:   output latent dim
        act_embed_dim:  kept for interface parity; only used if use_history=True
        use_history:    keep signature; if True, concatenates history just like your class
        num_hist:       number of history steps when use_history=True
        hidden_features,num_layers,dropout,activation: MLP head config
        sample_in_eval: if True, still samples at eval time; otherwise uses mu
    """

    def __init__(
        self,
        in_features=None,
        out_features=None,
        act_embed_dim=0,
        use_history=False,
        num_hist=3,
        hidden_features=None,
        num_layers=2,
        dropout=0.0,
        activation=nn.ReLU,
        sample_in_eval=False,
        visual_emb_dim=None,
        proprio_emb_dim=None,
        projected_dim=None,
    ):
        super().__init__()
        if in_features is None:
            if visual_emb_dim is None or proprio_emb_dim is None:
                raise ValueError("Provide in_features or both visual_emb_dim and proprio_emb_dim.")
            in_features = visual_emb_dim + proprio_emb_dim
        if out_features is None:
            if projected_dim is None:
                raise ValueError("Provide out_features or projected_dim.")
            out_features = projected_dim
        self.in_features = in_features
        self.out_features = out_features
        self.act_embed_dim = act_embed_dim
        self.use_history = use_history
        self.num_hist = num_hist
        self.sample_in_eval = sample_in_eval

        # figure out the feature size fed into the encoder
        if use_history:
            enc_in = in_features + num_hist * (out_features + act_embed_dim)
        else:
            enc_in = in_features

        # small MLP encoder -> 2 heads (mu, logvar)
        hidden_features = hidden_features or max(out_features * 2, enc_in)
        self.encoder = MLP(
            in_features=enc_in,
            out_features=hidden_features,
            hidden_features=hidden_features,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )
        self.mu_head = nn.Linear(hidden_features, out_features)
        self.logvar_head = nn.Linear(hidden_features, out_features)

        # state for history path (only used if use_history=True)
        if use_history:
            self.register_buffer("z_hist_init", torch.zeros(1, 1, num_hist, out_features))
            self.register_buffer("a_hist_init", torch.zeros(1, 1, num_hist, act_embed_dim))

        # loss buffers
        self.kl_loss = torch.tensor(0.0)
        self.kl_per_elem = None  # (B, T, P)

    @staticmethod
    def _reparameterize(mu, logvar, sample: bool):
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _kl_standard_normal(mu, logvar, reduce_mean=True, dims_to_sum=(-1,)):
        # KL(q(z|x)||p(z)) with p(z)=N(0, I): 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
        kl = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
        if dims_to_sum is not None:
            kl = kl.sum(dim=dims_to_sum)  # sum over latent dims
        if reduce_mean:
            return kl.mean()
        return kl

    def forward(self, o, act=None):
        """
        o:   (B, T, P, in_features) or visual_emb
        act: (B, T, P, act_embed_dim) or proprio_emb when use_history=False
        return:
        z:   (B, T, P, out_features)
        """
        B, T, P, _ = o.shape
        H = self.num_hist

        # Build encoder input per time step
        if self.use_history:
            assert act is not None, "act must be provided when use_history=True"
            device, dtype = o.device, o.dtype
            z_hist = self.z_hist_init.expand(B, P, H, self.out_features).to(device=device, dtype=dtype).clone()
            a_hist = self.a_hist_init.expand(B, P, H, self.act_embed_dim).to(device=device, dtype=dtype).clone()

            z_out = []
            kl_elems = []
            for t in range(T):
                hist_flat = torch.cat([z_hist, a_hist], dim=-1).reshape(B, P, -1)
                x_t = torch.cat([hist_flat, o[:, t]], dim=-1)  # (B, P, enc_in)
                h = self.encoder(x_t)                          # (B, P, Hf)
                mu = self.mu_head(h)                           # (B, P, D)
                logvar = self.logvar_head(h)                   # (B, P, D)

                sample_flag = self.training or self.sample_in_eval
                z_t = self._reparameterize(mu, logvar, sample=sample_flag)
                z_out.append(z_t)

                # update history buffers
                if H > 0:
                    if H > 1:
                        z_hist[:, :, :-1] = z_hist[:, :, 1:].clone()
                        a_hist[:, :, :-1] = a_hist[:, :, 1:].clone()
                    z_hist[:, :, -1] = z_t
                    a_hist[:, :, -1] = act[:, t]

                # KL per (B,P) for timestep t (sum over latent dims)
                kl_t = self._kl_standard_normal(mu, logvar, reduce_mean=False, dims_to_sum=(-1,))
                kl_elems.append(kl_t.unsqueeze(1))  # (B,1,P)

            z = torch.stack(z_out, dim=1)  # (B, T, P, D)
            kl_per_elem = torch.cat(kl_elems, dim=1)  # (B, T, P)
        else:
            # Fast path: no history, just project each (B,T,P,Fin) independently
            if act is not None and o.shape[:-1] == act.shape[:-1]:
                if o.shape[-1] + act.shape[-1] == self.in_features:
                    x = torch.cat([o, act], dim=-1)
                else:
                    raise ValueError(
                        f"Input feature mismatch: got {o.shape[-1]}+{act.shape[-1]}, "
                        f"expected {self.in_features}."
                    )
            else:
                if o.shape[-1] != self.in_features:
                    raise ValueError(
                        f"Input feature mismatch: got {o.shape[-1]}, expected {self.in_features}."
                    )
                x = o
            h = self.encoder(x)              # (B, T, P, Hf)
            mu = self.mu_head(h)             # (B, T, P, D)
            logvar = self.logvar_head(h)     # (B, T, P, D)
            sample_flag = self.training or self.sample_in_eval
            z = self._reparameterize(mu, logvar, sample=sample_flag)

            kl_per_elem = self._kl_standard_normal(mu, logvar, reduce_mean=False, dims_to_sum=(-1,))  # (B,T,P)

        # store KL stats for the caller to use in their loss
        self.kl_per_elem = kl_per_elem  # (B, T, P)
        self.kl_loss = kl_per_elem.mean()

        return z  # keep I/O identical to your original projector

    def __repr__(self):
        return (
            f"VAETemporalProjector(in_features={self.in_features}, "
            f"out_features={self.out_features}, use_history={self.use_history}, "
            f"num_hist={self.num_hist}, sample_in_eval={self.sample_in_eval})"
        )

if __name__ == "__main__":
    # quick test
    torch.manual_seed(0)
    print("🧪 Testing VAETemporalProjector...")

    # no-history path (recommended per your note)
    proj = VAETemporalProjector(
        in_features=416,
        out_features=64,
        act_embed_dim=0,     # unused when use_history=False
        use_history=False,
        hidden_features=512,
        num_layers=2,
        dropout=0.0,
        activation=nn.ReLU,
        sample_in_eval=False,  # mean at eval
    )

    x = torch.randn(2, 3, 196, 416)  # (B,T,P,Fin)

    # train mode -> sampling
    proj.train()
    z = proj(x)
    print("Train z:", z.shape, " KL(mean):", float(proj.kl_loss))

    # eval mode -> deterministic mean by default
    proj.eval()
    z_eval = proj(x)
    print("Eval z:", z_eval.shape, " KL(mean):", float(proj.kl_loss))

    # if you want eval-time sampling:
    proj.sample_in_eval = True
    z_eval_sampled = proj(x)
    print("Eval(sampling) z:", z_eval_sampled.shape)
    print("✅ VAETemporalProjector tests passed!")
