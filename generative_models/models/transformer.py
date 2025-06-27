from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 512
    dropout: float = 0.1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", pos_enc)

    def forward(self, x):
        return x + self.encoding[: x.size(1), :].unsqueeze(0)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.scale = self.d_head**0.5

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, key_padding_mask=None):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(q.size(0), -1, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(k.size(0), -1, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(v.size(0), -1, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        causal_mask = (
            torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        )
        scores.masked_fill_(causal_mask, -1e9)
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :]
            scores.masked_fill_(kpm, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        x_attn = torch.matmul(attn, v)
        x_attn = (
            x_attn.transpose(1, 2).contiguous().view(x_attn.size(0), -1, self.d_model)
        )
        x = self.norm1(x + self.dropout(x_attn))
        x_ff = self.ff(x)
        x = self.norm2(x + self.dropout(x_ff))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.d_ff = config.d_ff
        self.dropout = config.dropout

        self.input_projection = nn.Linear(2, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.attention_blocks = nn.ModuleList(
            [
                MultiHeadAttentionBlock(
                    self.d_model, self.n_heads, self.d_ff, self.dropout
                )
                for _ in range(self.n_layers)
            ]
        )

        self.output_layer = nn.Linear(self.d_model, 2)

    def forward(self, x, key_padding_mask=None):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        for block in self.attention_blocks:
            x = block(x, key_padding_mask)
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    config = TransformerConfig(
        d_model=128, n_heads=8, n_layers=6, d_ff=512, dropout=0.1
    )
    model = Transformer(config)
    x = torch.randn(10, 5, 2)
    print(model(x, key_padding_mask=torch.zeros(10, 5).bool()).shape)
