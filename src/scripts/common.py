import argparse
from pathlib import Path
import yaml
from src.utils import reproducibility


def parser():
    root = Path(__file__).resolve().parents[2]
    result = argparse.ArgumentParser()
    result.add_argument('--cohort', required=True, choices=['BRCA', 'LUAD', 'STAD', 'LIHC', 'BLCA'])
    result.add_argument('--task', required=True, choices=['CI', 'AUC'])
    result.add_argument('--root', type=Path, default=root)
    result.add_argument('--config', type=Path)
    result.add_argument('--device', default='auto')
    result.add_argument('--mode', choices=['selected', 'grid', 'evaluate'], default='selected')
    result.add_argument('--output', type=Path)
    result.add_argument('--epochs', type=int)
    result.add_argument('--encoder-max-epochs', type=int)
    result.add_argument('--representations', type=Path)
    result.add_argument('--checkpoint', type=Path)
    return result


def run(entry):
    args = parser().parse_args()
    import torch
    from src.encoder import encode
    from src.training import run_downstream
    from src.utils.data import load_clinical, load_representations
    root = args.root.resolve()
    config = yaml.safe_load((args.config or root / 'configs' / 'default.yaml').read_text(encoding='utf-8'))
    if config['seed'] != 42:
        raise ValueError('The released protocol uses seed 42')
    if entry == 'pipeline' and args.mode == 'evaluate':
        raise ValueError('Use a checkpoint or saved representations with evaluate mode')
    if args.epochs is not None:
        if args.epochs < config['hgnn']['validation_interval']:
            raise ValueError('epochs must include at least one validation interval')
        config['hgnn']['epochs'] = args.epochs
    if args.encoder_max_epochs is not None:
        if args.encoder_max_epochs < 1:
            raise ValueError('encoder-max-epochs must be positive')
        config['encoder']['max_epochs'] = args.encoder_max_epochs
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    reproducibility.configure(config['seed'])
    output = args.output or root / 'results' / 'runs' / args.cohort / args.task / entry
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output}')
    if entry == 'pipeline':
        directory = encode(root, args.cohort, output / 'encoder', device, config['encoder'])
    elif entry == 'checkpoint':
        checkpoint = args.checkpoint or root / 'pretrained' / args.cohort / 'mspae.pth'
        directory = encode(root, args.cohort, output / 'encoder', device, config['encoder'], checkpoint)
    else:
        directory = args.representations or root / 'pretrained' / args.cohort / 'representations'
    clinical, folds = load_clinical(root, args.cohort)
    arrays = load_representations(directory, clinical.index)
    selected = yaml.safe_load((root / 'configs' / 'cohorts' / f'{args.cohort}.yaml').read_text(encoding='utf-8'))[args.task]
    if args.mode == 'grid':
        search = yaml.safe_load((root / 'configs' / f'search_{args.task.lower()}.yaml').read_text(encoding='utf-8'))
        pairs = [(a, b) for a in search['ranking_weight'] for b in search['tail_weight']]
    else:
        pairs = [(selected['ranking_weight'], selected['tail_weight'])]
    checkpoints = root / 'pretrained' / args.cohort / args.task if args.mode == 'evaluate' else None
    summary = run_downstream(arrays, clinical, folds, args.task, config['hgnn'], device,
                             output / 'prediction', pairs, checkpoints)
    print(f"{args.cohort} {args.task}: {summary['mean']:.10f} +/- {summary['sample_SD']:.10f}", flush=True)
