"""Small permutation-equivariant transformer for four-candidate selection."""

from __future__ import annotations

import torch
from torch import nn


class CandidateTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        layers: int = 2,
        heads: int = 4,
        feedforward_size: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dropout = nn.Dropout(dropout * 0.5)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.scorer = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(self.input_dropout(features))
        hidden = self.input_norm(hidden)
        hidden = self.encoder(hidden)
        return self.scorer(self.output_norm(hidden)).squeeze(-1)
