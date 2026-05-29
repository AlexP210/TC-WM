import torch
import torch.nn as nn


def _resolve_activation(activation):
    if isinstance(activation, str):
        name = activation.strip()
        if "." in name:
            module_path, attr = name.rsplit(".", 1)
            try:
                module = __import__(module_path, fromlist=[attr])
                return getattr(module, attr)
            except (ImportError, AttributeError) as exc:
                raise ValueError(f"Unknown activation '{activation}'") from exc
        if not hasattr(nn, name):
            raise ValueError(f"Unknown activation '{activation}'")
        return getattr(nn, name)
    return activation


class EmbDecoderMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=None,
        num_layers=2,
        dropout=0.0,
        activation=nn.ReLU,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 2 * out_dim
        activation = _resolve_activation(activation)

        layers = []
        prev_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ActionConditionedEmbDecoderMLP(nn.Module):
    """
    EmbDecoderMLP conditioned on action history via a causal GRU.

    At each timestep t, the GRU encodes actions 0..t-1 (shifted by 1, zero-padded
    at t=0) into a fixed-size context vector, which is then concatenated with z
    before the MLP. This allows the decoder to model dynamics using the history of
    actions that led to the current state.

    z:      (b, t, p, in_dim)
    action: (b, t, action_dim)
    output: (b, t, p, out_dim)
    """

    def __init__(
        self,
        in_dim,
        action_dim,
        out_dim,
        gru_hidden_dim=64,
        hidden_dim=None,
        num_layers=2,
        dropout=0.0,
        activation=nn.ReLU,
        shift_action=True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 2 * out_dim
        activation = _resolve_activation(activation)

        self.action_dim = action_dim
        self.shift_action = shift_action
        self.action_gru = nn.GRU(action_dim, gru_hidden_dim, batch_first=True)

        layers = []
        prev_dim = in_dim + gru_hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, z, action):
        """
        z:      (b, t, p, in_dim)
        action: (b, t, action_dim)

        shift_action=True  (original): position t sees act[0..t-1]; t=0 sees zeros.
        shift_action=False (fixed):    position t sees act[0..t]; GRU includes the
                                       action that caused the transition to frame t.
        """
        b, t, p, _ = z.shape
        if self.shift_action:
            act_in = torch.zeros_like(action)
            act_in[:, 1:] = action[:, :-1]          # causal-shifted
        else:
            act_in = action                          # include current action
        h, _ = self.action_gru(act_in)              # (b, t, gru_hidden_dim)
        h_tiled = h.unsqueeze(2).expand(-1, -1, p, -1)    # (b, t, p, gru_hidden_dim)
        x = torch.cat([z, h_tiled], dim=-1)                # (b, t, p, in_dim+gru_hidden_dim)
        return self.mlp(x)
