import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.empty(4, 64))
        nn.init.normal_(self.query_tokens, mean=0., std=0.02)
        self.cross_attention = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
        self.out_proj = nn.Linear(256, 1, bias=False)

    def forward(self, embeddings):
        values = torch.stack(embeddings, dim=1)
        queries = self.query_tokens.unsqueeze(0).expand(values.shape[0], -1, -1)
        attended, weights = self.cross_attention(queries, values, values, need_weights=True, average_attn_weights=False)
        return self.out_proj(attended.reshape(values.shape[0], 256)), weights
