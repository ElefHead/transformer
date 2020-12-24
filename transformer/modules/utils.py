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
