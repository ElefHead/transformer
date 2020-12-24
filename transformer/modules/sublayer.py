import torch
from torch import nn
from transformer.modules import LayerNorm

from typing import Union, List, Tuple


class Sublayer(nn.Module):
    def __init__(self, in_features: Union[int, List[int], Tuple[int, ...]],
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
