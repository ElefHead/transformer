import torch
from torch import nn
from torch.nn.functional import softmax

from math import sqrt

from typing import Tuple


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
