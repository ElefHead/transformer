import torch
from torch import nn
from torch.optim import Optimizer, Adam
from transformer.model import EncoderDecoder
from transformer.modules import Generator

from typing import List, Optional


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, d_model: int,
                 factor: int, warmup_steps: int,
                 optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self._step = 0
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.d_model = d_model
        self._rate = 0

    def step(self):
        "Update params and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.d_model ** -0.5) * \
            min(step ** -0.5, step * (self.warmup_steps ** -1.5))


def get_std_opt(model: EncoderDecoder):
    return NoamOpt(
        model.d_model,
        factor=2,
        warmup_steps=4000,
        optimizer=Adam(
            model.parameters(),
            lr=0, betas=(0.9, 0.98), eps=1e-9
        )
    )


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size: int, padding_idx: int,
                 smoothing: float = 0.0) -> None:
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, torch.tensor(true_dist, requires_grad=False))


class SimpleLossCompute:
    "A simple loss compute and train function"

    def __init__(self, generator: Generator, criterion: nn.Module,
                 opt: NoamOpt = None) -> None:
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x: torch.Tensor, y: torch.Tensor, norm: int) -> float:
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


class MultiGPULossCompute:
    def __init__(self, generator: Generator,
                 criterion: Optimizer,
                 devices: List[int],
                 opt: Optional[NoamOpt] = None,
                 chunk_size: int = 5) -> None:
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, norm):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)

        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            out_column = [
                [torch.tensor(o[:, i:i+chunk_size].data,
                 requires_grad=self.opt is not None)]
                for o in out_scatter
            ]
            gen = nn.parallel.parallel_apply(generator, out_column)

            y = [
                (
                    g.contiguous().view(-1, g.size(-1)),
                    t[:, i:i+chunk_size].contiguous().view(-1)
                )
                for g, t in zip(gen, targets)
            ]

            loss = nn.parallel.parallel_apply(self.criterion, y)

            l_value = nn.parallel.gather(loss, target_device=self.devices[0])
            l_value = l_value.sum() / norm

            total += l_value.item()

            if self.opt is not None:
                l_value.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        if self.opt is not None:
            out_grad = [torch.cat(og, dim=1) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])

            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return total * norm
