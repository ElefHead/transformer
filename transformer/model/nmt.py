import torch
from torch import nn

from transformer.modules import Encoder, Decoder, Generator
from transformer.modules import MultiHeadAttention, PositionalEncoding
from transformer.modules import PositionwiseFeedForward, Embeddings
from transformer.modules import EncoderLayer, DecoderLayer

from copy import deepcopy


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for this and many other models.
    """

    def __init__(self, encoder: Encoder,
                 decoder: Decoder, src_embed: nn.Sequential,
                 tgt_embed: nn.Sequential,
                 generator: Generator, d_model: int) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.d_model = d_model

    def forward(self, src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor,
               src_mask: torch.Tensor, tgt: torch.Tensor,
               tgt_mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(
            self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(src_vocab_size: int, tgt_vocab_size: int,
               N: int = 6, d_model: int = 512, d_ff: int = 2048,
               h: int = 8, dropout_prob: float = 0.1,
               max_len: int = 5000) -> EncoderDecoder:
    c = deepcopy
    attn = MultiHeadAttention(h=h, d_model=d_model,
                              dropout_prob=dropout_prob)
    ff = PositionwiseFeedForward(d_model=d_model,
                                 d_ff=d_ff, dropout_prob=dropout_prob)
    position = PositionalEncoding(
        d_model=d_model,
        dropout_prob=dropout_prob,
        max_len=max_len
    )
    model = EncoderDecoder(
        encoder=Encoder(
            EncoderLayer(
                size=d_model,
                self_attn=c(attn),
                feed_forward=c(ff),
                dropout_prob=dropout_prob
            ), N=N
        ),
        decoder=Decoder(
            DecoderLayer(
                size=d_model,
                masked_self_attn=c(attn),
                enc_attn=c(attn),
                feed_forward=c(ff),
                dropout_prob=dropout_prob
            ), N=N
        ),
        src_embed=nn.Sequential(
            Embeddings(
                d_model=d_model,
                vocab_size=src_vocab_size
            ),
            c(position)
        ),
        tgt_embed=nn.Sequential(
            Embeddings(
                d_model=d_model,
                vocab_size=tgt_vocab_size
            ),
            c(position)
        ),
        generator=Generator(
            d_model=d_model,
            vocab_size=tgt_vocab_size
        ),
        d_model=d_model
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model
