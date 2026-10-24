import json
from pathlib import Path
import numpy as np
import torch
from src.models.mspae import MSPAE
from src.models.losses import encoder_loss
from src.utils.reproducibility import configure, clone_state
from src.utils.data import load_omics, save_representations


def encode(root, cohort, output, device, config, checkpoint=None):
    configure()
    ids, values, masks = load_omics(root, cohort)
    inputs = {m: torch.as_tensor(x, device=device) for m, x in values.items()}
    observed = {m: torch.as_tensor(x, device=device) for m, x in masks.items()}
    model = MSPAE().to(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(payload['model_state_dict'], strict=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        best, meaningful, last, epoch = float('inf'), float('inf'), 0, 0
        history = []
        while True:
            epoch += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = encoder_loss(model(inputs), inputs, observed, config['alignment_weight'], config['orthogonality_weight'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite MSPAE objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Non-finite MSPAE gradient')
            optimizer.step()
            model.eval()
            with torch.no_grad():
                value = float(encoder_loss(model(inputs), inputs, observed, config['alignment_weight'], config['orthogonality_weight']))
            if not np.isfinite(value):
                raise FloatingPointError('Non-finite MSPAE evaluation')
            if value < best:
                best = value
                payload = {'model_state_dict': clone_state(model), 'epoch': epoch, 'total_loss': best}
            delta = max(config['absolute_min_delta'], abs(meaningful) * config['relative_min_delta']) if np.isfinite(meaningful) else 0.
            if not np.isfinite(meaningful) or value < meaningful - delta:
                meaningful, last = value, epoch
            history.append({'epoch': epoch, 'loss': value})
            if epoch % 100 == 0:
                print(f'MSPAE {cohort}: epoch={epoch}, loss={value:.8f}', flush=True)
            if epoch >= config['minimum_epochs'] and epoch - last >= config['patience']:
                break
            if config['max_epochs'] is not None and epoch >= config['max_epochs']:
                break
        torch.save(payload, output / 'mspae.pth')
        (output / 'encoder_training.json').write_text(json.dumps({'epochs_trained': epoch, 'selected_epoch': payload['epoch'], 'history': history}, indent=2), encoding='utf-8')
        model.load_state_dict(payload['model_state_dict'], strict=True)
    model.eval()
    with torch.no_grad():
        arrays = [x.cpu().numpy().copy() for x in MSPAE.representations(model(inputs))]
    save_representations(output / 'representations', ids, arrays)
    return output / 'representations'
