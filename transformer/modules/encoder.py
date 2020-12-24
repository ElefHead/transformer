import torch
from torch import nn
from transformer.modules import clone_layer, Sublayer, LayerNorm

from typing import Union, List, Tuple


class EncoderLayer(nn.Module):
    def __init__(self, size: Union[int, List[int], Tuple[int, ...]],
                 self_attn: nn.Module, feed_forward: nn.Module,
                 dropout_prob: int):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone_layer(Sublayer(size, dropout_prob), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        self.layers = clone_layer(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Pass the input x(and mask) through each layer in turn.
        """
        for layer in self.layers:
            """
            Encoder layer will also have encoder masks.
            Duh.
            """
            x = layer(x, mask)
        return self.norm(x)
