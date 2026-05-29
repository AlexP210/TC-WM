import torch
import torch.nn as nn
import torch.nn.functional as F

def symlog(x: torch.Tensor) -> torch.Tensor:
    # symlog(x) = sign(x) * log(1 + |x|)
    return torch.sign(x) * torch.log1p(torch.abs(x))

class SymlogMSELoss(nn.Module):
    """
    L = MSE( symlog(pred), symlog(target) )
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_t = symlog(input)
        target_t = symlog(target)
        return F.mse_loss(input_t, target_t, reduction=self.reduction)