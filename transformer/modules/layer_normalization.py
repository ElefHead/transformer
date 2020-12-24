import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self,
                 in_features,
                 gamma=True,
                 beta=True,
                 epsilon=1e-6):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param in_features: The shape of the input tensor
            or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super(LayerNorm, self).__init__()
        if isinstance(in_features, int):
            in_features = (in_features,)
        else:
            in_features = (in_features[-1],)
        self.in_features = torch.Size(in_features)
        self.epsilon = epsilon
        if gamma:
            self.gamma = nn.Parameter(torch.Tensor(*in_features))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.Parameter(torch.Tensor(*in_features))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(std=-1, keepdim=True)
        y = (x - mean) / (std + self.epsilon)
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta
        return y

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, '
            f'gamma={self.gamma is not None}, '
            f'beta={self.beta is not None}, '
            f'epsilon={self.epsilon}'
        )
