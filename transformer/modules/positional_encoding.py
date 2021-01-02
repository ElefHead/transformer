import torch
from torch import nn
from torch.autograd import Variable

from math import log


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout_prob, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class PositionalEncodingOther(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float,
                 max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = torch.tensor(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + torch.tensor(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
