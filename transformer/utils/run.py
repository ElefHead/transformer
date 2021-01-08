import torch
from torch import nn
from transformer.model import EncoderDecoder
from transformer.datasets import Batch, subsequent_mask

from typing import Iterator
from time import time


def run_epoch(data_iter: Iterator[Batch], model: EncoderDecoder,
              loss_compute: nn.Module) -> float:
    "Standard Training and Logging Function"
    start = time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            src=batch.src, tgt=batch.trg,
            src_mask=batch.src_mask,
            tgt_mask=batch.trg_mask
        )
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time() - start
            print(
                f"Epoch step: {i} Loss: {loss / batch.ntokens} "
                f"Tokens per sec: {tokens/elapsed}"
            )
            start = time()
            tokens = 0

    return total_loss / total_tokens


def greedy_decode(model: EncoderDecoder, src: torch.Tensor,
                  src_mask: torch.Tensor, max_len: int,
                  start_symbol: int) -> torch.Tensor:
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(
            memory, src_mask,
            ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([
            ys,
            torch.ones(1, 1).type_as(src.data).fill_(next_word)
            ], dim=1)
    return ys
