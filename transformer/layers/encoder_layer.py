import torch
from torch import nn
from transformer.layers import clone_layer, Sublayer

from typing import Union, List


class EncoderLayer(nn.Module):
    def __init__(self, size: Union[int, List[int], tuple],
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
