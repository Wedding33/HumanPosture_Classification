import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, flag):
        super(TransformerModel, self).__init__()
        self.src_mask = None
        self.ninp = 34 if flag == 0 else 68
        self.max_len = 301
        self.nnDropout = 0.2
        self.pos_encoder = PositionalEncoding(
            d_model=self.ninp,
            dropout=0.5,
            max_len=self.max_len,
        )
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.ninp,
            nhead=2,
            batch_first=False
        )
        self.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer=self.TransformerEncoderLayer,
            num_layers=6
        )
        self.fc = nn.Sequential(
            nn.Linear(self.ninp * 301, 6400),
            nn.Dropout(p=self.nnDropout),
            nn.ReLU(inplace=True),
            nn.Linear(6400, 1600),
            nn.Dropout(p=self.nnDropout),
            nn.ReLU(inplace=True),
            nn.Linear(1600, 160),
            nn.Dropout(p=self.nnDropout),
            nn.ReLU(inplace=True),
            nn.Linear(160, 5),
            nn.Softmax(dim=0)
        )

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.pos_encoder(src)
        x = self.TransformerEncoder(src, self.src_mask)
        x = x.flatten()
        output = self.fc(x)
        return output.unsqueeze(0)
