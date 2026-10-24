import torch
from torch.nn import functional as F


def build_incidence(features, neighbors=10):
    normalized = F.normalize(features, p=2, dim=1)
    similarity = normalized @ normalized.t()
    _, indices = torch.topk(similarity, k=min(neighbors + 1, len(features)), dim=1)
    incidence = torch.zeros(len(features), len(features), device=features.device)
    rows = torch.arange(len(features), device=features.device).view(-1, 1).expand_as(indices)
    incidence[rows.reshape(-1), indices.reshape(-1)] = 1.
    incidence.fill_diagonal_(0)
    return incidence
