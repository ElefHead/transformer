import torch
from torch import nn

from transformer.modules import clone_module, Sublayer, LayerNorm


class DecoderLayer(nn.Module):
    def __init__(self, size,
                 masked_self_attn: nn.Module,
                 enc_attn: nn.Module,
                 feed_forward: nn.Module,
                 dropout_prob: float):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.masked_self_attn = masked_self_attn
        self.enc_attn = enc_attn
        self.feed_forward = feed_forward
        self.sublayer = clone_module(
            Sublayer(size, dropout_prob=dropout_prob), 3)

    def forward(self, x: torch.Tensor,
                memory: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        m = memory
        x = self.sublayer[0](
            x, lambda x: self.masked_self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.enc_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int):
        super(Decoder, self).__init__()
        self.layers = clone_module(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor,
                memory: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
