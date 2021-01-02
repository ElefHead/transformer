import torch
from torch import nn
from torch.nn import functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int,
                 d_ff: int,
                 dropout_prob: float = 0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.w_2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
