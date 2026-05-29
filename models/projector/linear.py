# a linear projector

import torch.nn as nn

class LinearProjector(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projector = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.projector(x)