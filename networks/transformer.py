from __future__ import annotations

from copy import deepcopy
import math
import numpy as np
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        if self.batch_first:
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:,:x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048, 
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        batch_first: bool = True,
    ):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, dropout, batch_first=batch_first)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.encoder(x)
        return x
















class ViTEncoder(nn.Module):
    """modified from monai.ViT
    """    
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        dropout_rate: float = 0.0,
        spatial_dims: int = 1,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        x = self.patch_embedding(x)
        #hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            #hidden_states_out.append(x)
        x = self.norm(x)
        return x