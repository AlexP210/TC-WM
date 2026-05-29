import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def nonlinear_libs(nonlinear):
    if nonlinear == 'relu':
        return nn.ReLU()
    elif nonlinear == 'leaky_relu':
        return nn.LeakyReLU()
    elif nonlinear == 'elu':
        return nn.ELU()
    elif nonlinear == 'sigmoid':
        return nn.Sigmoid()
    elif nonlinear == 'softplus':
        return nn.Softplus()


class ActionAttention(nn.Module):

    HIDDEN_SIZE = 512

    def __init__(self, state_size, action_size, key_query_size, value_size, sqrt_scale,
                 abalation=False, use_sigmoid=False):

        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.key_query_size = key_query_size
        self.value_size = value_size
        self.sqrt_scale = sqrt_scale
        self.abalation = abalation
        self.use_sigmoid = use_sigmoid

        if self.use_sigmoid:
            self.normalizer = lambda x, dim: torch.sigmoid(x)
        else:
            self.normalizer = lambda x, dim: F.softmax(x, dim=dim)

        self.fc_key = MLP(self.state_size, self.key_query_size, self.HIDDEN_SIZE)
        self.fc_query = MLP(self.action_size, self.key_query_size, self.HIDDEN_SIZE)
        self.fc_value = MLP(self.action_size, self.value_size, self.HIDDEN_SIZE)

    def _split_inputs(self, x):
        if isinstance(x, (tuple, list)):
            state, action = x
        else:
            state, action = x[..., : self.state_size], x[..., self.state_size :]
        return self._reshape_state(state), self._reshape_action(action)

    def _reshape_state(self, state):
        if state.dim() == 4:
            b, t, p, d = state.shape
            state = state.reshape(b * t, p, d)
        elif state.dim() == 3:
            pass
        else:
            raise ValueError(f"Unsupported state shape {state.shape}")
        return state

    def _reshape_action(self, action):
        if action.dim() == 4:
            b, t, _, _ = action.shape
            action = action[:, :, 0, : self.action_size]
            action = action.reshape(b * t, -1)
        elif action.dim() == 3:
            b, t, _ = action.shape
            action = action[:, :, : self.action_size]
            action = action.reshape(b * t, -1)
        elif action.dim() == 2:
            action = action[:, : self.action_size]
        else:
            raise ValueError(f"Unsupported action shape {action.shape}")
        return action

    def forward(self, x):
        state, action = self._split_inputs(x)

        batch_size = state.size(0)
        patch_size = state.size(1)
        state_r = state.reshape(batch_size * patch_size, state.size(2))

        key_r = self.fc_key(state_r)
        query = self.fc_query(action)
        value = self.fc_value(action)

        key = key_r.reshape(batch_size, patch_size, self.key_query_size)

        scores = (key * query[:, None]).sum(dim=2)
        if self.sqrt_scale:
            scores = scores * (1.0 / (self.key_query_size ** 0.5))

        weights = self.normalizer(scores, dim=-1)

        if self.abalation:
            weights = torch.ones_like(weights) / weights.shape[1]

        return weights[:, :, None] * value[:, None, :]

    def forward_weights(self, x):
        state, action = self._split_inputs(x)

        batch_size = state.size(0)
        obj_size = state.size(1)
        state_r = state.reshape(batch_size * obj_size, state.size(2))

        key_r = self.fc_key(state_r)
        query = self.fc_query(action)

        key = key_r.reshape(batch_size, obj_size, self.key_query_size)

        scores = (key * query[:, None]).sum(dim=2)
        if self.sqrt_scale:
            scores = scores * (1.0 / (self.key_query_size ** 0.5))

        weights = self.normalizer(scores, dim=-1)

        if self.abalation:
            weights = torch.ones_like(weights) / weights.shape[1]

        return weights

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, nonlinear='relu'):
        super(MLP, self).__init__()

        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = nonlinear_libs(nonlinear)
        self.act2 = nonlinear_libs(nonlinear)

    def forward(self, x):
        h = self.act1(self.fc1(x))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)

if __name__ == "__main__":
    state_size = 128
    action_size = 64
    key_query_size = 64
    value_size = 64
    sqrt_scale = True
    abalation = False

    attn = ActionAttention(
        state_size=state_size,
        action_size=action_size,
        key_query_size=key_query_size,
        value_size=value_size,
        sqrt_scale=sqrt_scale,
        abalation=abalation,
        use_sigmoid=False
    )

    B = 32   # batch size
    P = 10  

    state = torch.randn(B, P, state_size)     # [32, 10, 128]
    action = torch.randn(B, action_size)      # [32, 64]
    out = attn((state, action))               # [32, 10, 64]
    weights = attn.forward_weights((state, action))   # [32, 10]

    # Time-batched example
    T = 3
    state_t = torch.randn(B, T, P, state_size)
    action_t = torch.randn(B, T, 1, action_size)
    weights_t = attn.forward_weights((state_t, action_t))
    print("Time weights shape:", weights_t.shape)

    print("Output shape:", out.shape)
    print("Weights shape:", weights.shape)

    print("Example weights (batch 0):", weights[0])
