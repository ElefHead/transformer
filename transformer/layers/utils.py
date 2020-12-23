from torch import nn
from copy import deepcopy


def clones(layer: nn.Module, N: int) -> nn.ModuleList:
    """
    Produce N identical layers
    """
    return nn.ModuleList(
        modules=[
            deepcopy(layer)
        ] * N
    )
