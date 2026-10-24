import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.models.hgnn import EmbeddingHGNN
from src.models.fusion import CrossAttentionFusion
from src.models.losses import prognosis_loss
from src.utils.reproducibility import configure, clone_state
from src.utils.hypergraph import build_incidence
from src.utils.metrics import compute_metrics


def create_models(device, dropout=0.3):
    nets = nn.ModuleList([EmbeddingHGNN(i, dropout).to(device) for i in range(4)])
    fusion = CrossAttentionFusion().to(device)
    return nets, fusion


def predict(nets, fusion, features, graphs):
    nets.eval()
    fusion.eval()
    with torch.no_grad():
        risk, _ = fusion([nets[i](features[i], graphs[i]) for i in range(4)])
    return risk.cpu().numpy().reshape(-1)


def prepare_folds(arrays, time, event, folds, device, neighbors):
    prepared = []
    for fold in range(5):
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        x_train = [torch.as_tensor(np.ascontiguousarray(x[train]), device=device) for x in arrays]
        x_valid = [torch.as_tensor(np.ascontiguousarray(x[valid]), device=device) for x in arrays]
        prepared.append({'fold': fold, 'indices': valid, 'train': x_train, 'valid': x_valid,
                         'graphs_train': [build_incidence(x, neighbors) for x in x_train],
                         'graphs_valid': [build_incidence(x, neighbors) for x in x_valid],
                         'time_train': torch.as_tensor(time[train], device=device),
                         'event_train': torch.as_tensor(event[train], device=device, dtype=torch.long),
                         'time': time[valid], 'event': event[valid]})
    return prepared


def fit_fold(data, task, ranking, tail, config, device):
    configure()
    nets, fusion = create_models(device, config['dropout'])
    parameters = list(nets.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    best, counter, payload = 0., 0, None
    for epoch in range(1, config['epochs'] + 1):
        nets.train()
        fusion.train()
        optimizer.zero_grad()
        risk, _ = fusion([nets[i](data['train'][i], data['graphs_train'][i]) for i in range(4)])
        loss = prognosis_loss(risk, data['time_train'], data['event_train'], task, ranking, tail, config['tail_mass'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite prediction objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Non-finite prediction gradient')
        optimizer.step()
        if epoch % config['validation_interval'] == 0:
            predictions = predict(nets, fusion, data['valid'], data['graphs_valid'])
            score = compute_metrics(data['time'], data['event'], predictions)[task]
            if score > best:
                best, counter = score, 0
                payload = {'nets': clone_state(nets), 'fuse': clone_state(fusion),
                           'epoch': epoch, 'fold': data['fold'], 'task': task,
                           'ranking_weight': ranking, 'tail_weight': tail}
            else:
                counter += 1
                if counter >= config['patience']:
                    break
    if payload is None:
        raise RuntimeError('No valid checkpoint was selected')
    nets.load_state_dict(payload['nets'], strict=True)
    fusion.load_state_dict(payload['fuse'], strict=True)
    risk = predict(nets, fusion, data['valid'], data['graphs_valid'])
    return payload, risk, compute_metrics(data['time'], data['event'], risk)


def run_downstream(arrays, clinical, folds, task, config, device, output, pairs, checkpoints=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    time = np.ascontiguousarray(clinical.time.to_numpy(dtype=np.float32))
    event = np.ascontiguousarray(clinical.event.to_numpy(dtype=np.float32))
    prepared = prepare_folds(arrays, time, event, folds, device, config['neighbors'])
    best_mean, summaries = -float('inf'), []
    for ranking, tail in pairs:
        scores, rows, selected = [], [], []
        for data in prepared:
            fold = data['fold']
            if checkpoints is not None:
                payload = torch.load(Path(checkpoints) / f'fold{fold}.pth', map_location='cpu', weights_only=True)
                if payload['task'] != task or payload['fold'] != fold:
                    raise ValueError('Checkpoint task or fold differs')
                configure()
                nets, fusion = create_models(device, config['dropout'])
                nets.load_state_dict(payload['nets'], strict=True)
                fusion.load_state_dict(payload['fuse'], strict=True)
                risk = predict(nets, fusion, data['valid'], data['graphs_valid'])
                metrics = compute_metrics(data['time'], data['event'], risk)
            else:
                payload, risk, metrics = fit_fold(data, task, ranking, tail, config, device)
            scores.append(metrics[task])
            rows.append({'fold': fold, 'best_epoch': payload['epoch'], **metrics})
            selected.append((payload, pd.DataFrame({'patient_id': clinical.index[data['indices']],
                             'time': data['time'], 'event': data['event'], 'risk': risk})))
            print(f'{task} fold={fold}, ranking={ranking}, tail={tail}, score={metrics[task]:.8f}', flush=True)
        mean = float(np.mean(scores))
        summary = {'task': task, 'ranking_weight': ranking, 'tail_weight': tail,
                   'mean': mean, 'sample_SD': float(np.std(scores, ddof=1)), 'folds': rows}
        summaries.append(summary)
        if mean > best_mean:
            best_mean = mean
            for fold, (payload, frame) in enumerate(selected):
                torch.save(payload, output / f'fold{fold}.pth')
                frame.to_csv(output / f'fold{fold}_predictions.csv', index=False, float_format='%.9g')
            (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'search_results.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    return json.loads((output / 'summary.json').read_text(encoding='utf-8'))
