import torch
from torch import nn
from math import sqrt


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(Embeddings, self).__init__()
        self.look_up = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.look_up(x) / sqrt(self.d_model)
