import torch
from torch import nn
from torch.nn.functional import log_softmax


class Generator(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super(Generator, self).__init__()
        self.proj = nn.Linear(
            in_features=d_model,
            out_features=vocab_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return log_softmax(self.proj(x), dim=-1)
