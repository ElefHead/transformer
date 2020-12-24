import torch
from torch import nn
from torch.nn.functional import softmax

from math import sqrt

from typing import Tuple

from transformer.modules import clone_module


def attention(query: torch.Tensor, key: torch.Tensor,
              value: torch.Tensor, mask: torch.Tensor = None,
              dropout: nn.Module = None) -> Tuple[torch.Tensor, torch.Tensor]:
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    attention_probs = softmax(scores, dim=-1)
    if dropout is not None:
        attention_probs = dropout(attention_probs)
    return torch.matmul(attention_probs, value), attention_probs


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int,
                 dropout_prob: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clone_module(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # linear projection using first 3 linears
        # last one is for combined output
        query, key, value = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        context_vector, self.attn = attention(
            query=query, key=key,
            value=value, mask=mask,
            dropout=self.dropout
        )

        context_vector = context_vector.transpose(1, 2).contiguous()\
                                       .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](context_vector)
