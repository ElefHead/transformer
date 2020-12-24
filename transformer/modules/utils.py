import torch
from torch import nn
from copy import deepcopy


def clone_layer(layer: nn.Module, N: int) -> nn.ModuleList:
    """
    Produce N identical layers
    """
    return nn.ModuleList(
        modules=[
            deepcopy(layer)
        ] * N
    )


def subsequent_mask(size: int) -> torch.Tensor:
    """
    """
    attn_shape = (1, size, size)
    sub_mask = torch.triu(
        torch.ones(size=attn_shape, dtype=torch.uint8),
        diagonal=1
    )
    return sub_mask == 0
