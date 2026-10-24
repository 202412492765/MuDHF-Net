import math
import torch
from torch.nn import functional as F
from src.models.mspae import MODALITIES


def cosine_orthogonality(shared, private):
    if torch.mean(torch.norm(private, p=2, dim=1, keepdim=True)) < 1e-8:
        return torch.tensor(1., device=private.device)
    return torch.mean(torch.abs(torch.sum(
        F.normalize(shared, p=2, dim=1, eps=1e-8)
        * F.normalize(private, p=2, dim=1, eps=1e-8), dim=1)))


def encoder_loss(output, inputs, masks, alignment=0.3, orthogonality=0.4):
    reconstruction = output['h_shared'].sum() * 0
    align = reconstruction
    orth = reconstruction
    for m in MODALITIES:
        difference = torch.where(masks[m], (output['reconstruction'][m] - inputs[m]).square(),
                                 torch.zeros_like(inputs[m]))
        reconstruction = reconstruction + difference.sum() / masks[m].sum().clamp_min(1).to(difference.dtype)
        align = align + F.mse_loss(output['shared'][m], output['h_shared'])
        orth = orth + cosine_orthogonality(output['shared'][m], output['private'][m])
    return 1. * reconstruction + alignment * align + orthogonality * orth


def upper_tail_cvar(values, alpha=0.2):
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError('alpha must be finite and in (0, 1]')
    if values.ndim != 1 or not values.numel() or not values.is_floating_point():
        raise ValueError('Expected nonempty floating pair losses')
    if not bool(torch.isfinite(values).all()):
        raise ValueError('Non-finite pair losses')
    if alpha == 1:
        return values.mean()
    ordered = torch.sort(values, descending=True, stable=True).values
    mass = float(alpha * values.numel())
    boundary = ordered[math.ceil(mass) - 1].detach()
    above, tied = ordered > boundary, ordered == boundary
    remaining = mass - int(above.sum().item())
    numerator = ordered[above].sum() + (remaining / int(tied.sum().item())) * ordered[tied].sum()
    return numerator / mass


def prognosis_loss(risk, time, event, task, ranking_weight, tail_weight, tail_mass=0.2):
    risk, time, event = risk.view(-1), time.view(-1), event.view(-1)
    if task == 'CI':
        order = torch.argsort(time, descending=True)
        ordered_risk, ordered_event = risk[order], event[order]
        gamma = torch.exp(ordered_risk)
        cumulative = torch.cumsum(gamma.cpu(), dim=0).to(gamma.device) if gamma.is_cuda else torch.cumsum(gamma, dim=0)
        base = -torch.sum((ordered_risk - torch.log(cumulative + 1e-7)) * ordered_event) / (ordered_event.sum() + 1e-7)
        mask = (time[:, None] < time[None, :]) & (event[:, None] == 1)
    else:
        base = F.binary_cross_entropy_with_logits(risk, event.float())
        mask = (event[:, None] == 1) & (event[None, :] == 0)
    if not mask.any():
        return base + risk.sum() * 0
    pairs = F.softplus(-(risk[:, None] - risk[None, :])[mask] / 1.)
    return base + (ranking_weight * pairs.mean() + tail_weight * upper_tail_cvar(pairs, tail_mass))
