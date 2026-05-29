import torch
import torch.nn as nn


from models.base import BaseProjector

class MLP(BaseProjector):
    def __init__(
        self,
        in_features=None,
        out_features=None,
        hidden_features=None,
        num_layers=2,
        dropout=0.0,
        activation=nn.ReLU,
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

        hidden_features = hidden_features or 2 * out_features

        if isinstance(activation, str):
            name = activation.strip()
            if "." in name:
                module_path, attr = name.rsplit(".", 1)
                try:
                    module = __import__(module_path, fromlist=[attr])
                    activation = getattr(module, attr)
                except (ImportError, AttributeError) as exc:
                    raise ValueError(f"Unknown activation '{activation}'") from exc
            else:
                if not hasattr(nn, name):
                    raise ValueError(f"Unknown activation '{activation}'")
                activation = getattr(nn, name)

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

    def forward(self, x, proprio=None):
        if proprio is not None and x.shape[:-1] == proprio.shape[:-1]:
            if x.shape[-1] + proprio.shape[-1] != self.mlp[0].in_features:
                raise ValueError(
                    f"Input feature mismatch: got {x.shape[-1]}+{proprio.shape[-1]}, "
                    f"expected {self.mlp[0].in_features}."
                )
            x = torch.cat([x, proprio], dim=-1)
        return self.mlp(x)
