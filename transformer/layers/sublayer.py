import torch
from torch import nn
from transformer.layers import LayerNorm


class Sublayer(nn.Module):
    def __init__(self, in_features: int,
                 dropout_prob: float) -> None:
        """
        """
        super(Sublayer, self).__init__()
        self.norm = LayerNorm(in_features)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor,
                sublayer: nn.Module) -> torch.Tensor:
        """
        """
        return x + self.dropout(sublayer(self.norm(x)))
