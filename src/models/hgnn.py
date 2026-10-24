import torch
from torch import nn
from torch.nn import functional as F


class HGNNconv(nn.Module):
    def __init__(self, in_dim=64, hid=64, dropout=0.3):
        super().__init__()
        self.W = nn.Linear(in_dim, hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, incidence):
        degree_v = 1. / (incidence.sum(1) + 1e-6).clamp(min=1e-6)
        degree_e = 1. / (incidence.sum(0) + 1e-6).clamp(min=1e-6)
        edge = degree_e.view(-1, 1) * (incidence.t() @ features)
        vertex = (incidence @ edge) * degree_v.view(-1, 1)
        return self.dropout(F.relu(self.W(vertex)))


class EmbeddingHGNN(nn.Module):
    def __init__(self, modality_idx, dropout=0.3):
        super().__init__()
        self.modality_idx = modality_idx
        self.conv = HGNNconv(dropout=dropout)

    def forward(self, features, incidence):
        return self.conv(features, incidence)
