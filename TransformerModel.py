import math
from typing import Tuple

import torch
torch.manual_seed(1)
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class TransformerModel(nn.Module):

    def __init__(self, num_mfcc_feature: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, num_class: int = 3):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(num_mfcc_feature, dropout)
        encoder_layers = TransformerEncoderLayer(num_mfcc_feature, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.num_mfcc_feature = num_mfcc_feature
        self.decoder = nn.Linear(num_mfcc_feature, num_class)

        self.init_weights()


    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, num_class]
        """
        src = self.pos_encoder(src)
        # output shape: [seq_len, batch_size, num_class]
        output = self.transformer_encoder(src, src_mask)
        # for classification, calculate mean along the seq_len, output shape: [batch_size, num_class]
        output = output.mean(dim=0)
        # print(f"Output shape: {output.shape}")
        output = self.decoder(output)
        return output




class PositionalEncoding(nn.Module):

    def __init__(self, num_mfcc_feature: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_mfcc_feature, 2) * (-math.log(10000.0) / num_mfcc_feature))
        pe = torch.zeros(max_len, 1, num_mfcc_feature)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, num_mfcc_feature]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
