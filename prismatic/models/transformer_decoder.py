"""Vendored fallback for vint_train's MultiLayerDecoder_trans.

The navigation training pipeline relies on
``vint_train.models.vint.self_attention.MultiLayerDecoder_trans`` from the
``Learning-to-Drive-Anywhere-with-MBRA`` repo. The SO-101 training pipeline
re-uses the same Edge_adapter class but does not require any other MBRA
component, so to keep the manipulation entrypoint self-contained we provide a
drop-in replacement here that ``prismatic/models/small_head.py`` falls back to
when ``vint_train`` is not on the Python path.

Shape contract (matches how Edge_adapter calls it):
    forward(x: (B, seq_len, embed_dim)) -> (B, seq_len, embed_dim)

The original MBRA implementation is a Transformer encoder stack with learned
positional embeddings; we keep the same surface so checkpoints saved while
``vint_train`` was available remain *loadable* — though weights won't match
across the two implementations and the network will need to retrain from
scratch when the fallback is used.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


class _PositionalEmbedding(nn.Module):
    """Learned positional embedding of shape (1, seq_len, embed_dim)."""

    def __init__(self, seq_len: int, embed_dim: int) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        return x + self.pos[:, :T]


class MultiLayerDecoder_trans(nn.Module):
    """Transformer encoder used as the decoder block in Edge_adapter.

    Parameters mirror the original signature so a one-line swap works:

        MultiLayerDecoder_trans(
            embed_dim=1024,
            seq_len=10,
            output_layers=[256, 128, 64, 32],   # accepted, currently unused
            nhead=4,
            num_layers=4,
            ff_dim_factor=4,
        )
    """

    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        output_layers: Optional[Sequence[int]] = None,
        nhead: int = 4,
        num_layers: int = 4,
        ff_dim_factor: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        # output_layers is accepted for signature parity with the MBRA
        # implementation, but the Edge_adapter slices a single token out of the
        # decoder output and feeds its own MLP head, so no extra projection is
        # needed here.
        self.output_layers = list(output_layers) if output_layers is not None else None

        self.pos_embedding = _PositionalEmbedding(seq_len=seq_len, embed_dim=embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * ff_dim_factor,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, embed_dim) with T <= self.seq_len
        x = self.pos_embedding(x)
        return self.encoder(x)


# The full vint_train module also exports the two classes below; we provide
# permissive stubs so a "from prismatic.models.transformer_decoder import *"
# style import doesn't break if anyone else relies on the names. Both stubs
# fall through to ``MultiLayerDecoder_trans`` so the network at least runs.
MultiLayerDecoder = MultiLayerDecoder_trans
MultiLayerDecoder_idcat = MultiLayerDecoder_trans
