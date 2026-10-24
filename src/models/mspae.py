import torch
from torch import nn

MODALITIES = ('mirna', 'meth', 'mrna')
DIMENSIONS = {'mirna': 500, 'meth': 500, 'mrna': 2000}
REPRESENTATIONS = ('shared', 'private_mirna', 'private_meth', 'private_mrna')


class PairedEncoder(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(features, 64), nn.LeakyReLU(0.1))
        self.priv = nn.Sequential(nn.Linear(features, 64), nn.LeakyReLU(0.1))

    def forward(self, values):
        return self.shared(values), self.priv(values)


class SharedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, 64))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(64, 4, dropout=0, bias=False, batch_first=True)

    def forward(self, shared):
        values = torch.stack(shared, dim=1)
        query = self.query.expand(values.shape[0], -1, -1)
        result, weights = self.attention(query, values, values, need_weights=True, average_attn_weights=False)
        return result[:, 0, :], weights.mean(dim=1)[:, 0, :]


class MSPAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({m: PairedEncoder(DIMENSIONS[m]) for m in MODALITIES})
        self.fusion = SharedFusion()
        self.decoders = nn.ModuleDict({
            key: nn.Sequential(nn.Linear(64, DIMENSIONS[m]), nn.LeakyReLU(0.1))
            for prefix in ('shared', 'priv')
            for m, key in [(m, prefix + '_' + ('mir' if m == 'mirna' else m)) for m in MODALITIES]
        })

    def forward(self, values):
        pairs = {m: self.encoders[m](values[m]) for m in MODALITIES}
        shared = {m: pairs[m][0] for m in MODALITIES}
        private = {m: pairs[m][1] for m in MODALITIES}
        integrated, weights = self.fusion([shared[m] for m in MODALITIES])
        reconstruction = {
            m: self.decoders['shared_' + ('mir' if m == 'mirna' else m)](integrated)
            + self.decoders['priv_' + ('mir' if m == 'mirna' else m)](private[m])
            for m in MODALITIES
        }
        return {'shared': shared, 'private': private, 'h_shared': integrated,
                'alpha': weights, 'reconstruction': reconstruction}

    @staticmethod
    def representations(output):
        return [output['h_shared']] + [output['private'][m] for m in MODALITIES]
