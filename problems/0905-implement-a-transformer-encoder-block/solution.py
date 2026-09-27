import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout = dropout, batch_first = True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

        # TODO: norm1, attn, norm2, mlp, dropout
        pass

    def forward(self, x):
        #attention: layernorm, then attention, the dropout
        normed_x1 = self.norm1(x)
        atten, _ = self.attn(key = normed_x1, query = normed_x1, value = normed_x1)
        x = x + self.dropout(atten)

        #neural network
        normed_x2 = self.norm2(x)
        neu = self.mlp(normed_x2)
        x = x + self.dropout(neu)
        return x

        # TODO: pre-LN attention sublayer, then pre-LN MLP sublayer, both with residual
        pass
