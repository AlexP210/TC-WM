# Reusable residual block building blocks
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """2D residual block used in VQVAE encoder/decoder."""
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


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


class ResBlock1d(nn.Module):
    """1D residual block used in ResBlockProjector."""
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size=3,
        dropout=0.0,
        activation=nn.ReLU,
    ):
        super().__init__()
        activation = _resolve_activation(activation)
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv1d(
            channels, hidden_channels, kernel_size=kernel_size, padding=padding
        )
        self.act = activation()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.GroupNorm(1, hidden_channels)
        self.conv2 = nn.Conv1d(
            hidden_channels, channels, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.conv2(x)
        return x + residual
