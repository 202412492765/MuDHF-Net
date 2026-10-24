from pathlib import Path
import numpy as np
import pandas as pd
from src.models.mspae import DIMENSIONS, MODALITIES, REPRESENTATIONS


def read_frame(path, ids=None):
    frame = pd.read_csv(path, index_col=0, float_precision='round_trip')
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f'Duplicate identifiers: {path}')
    if ids is not None and not frame.index.equals(pd.Index(ids, name=frame.index.name)):
        raise ValueError(f'Patient order differs: {path}')
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise ValueError(f'Non-finite data: {path}')
    return frame


def load_clinical(root, cohort):
    clinical = read_frame(root / 'data' / cohort / f'{cohort}_clinical.csv')
    if list(clinical.columns) != ['time', 'event']:
        raise ValueError('Clinical columns must be time and event')
    if not clinical.event.isin([0, 1]).all() or not (clinical.time > 0).all():
        raise ValueError('Invalid survival endpoints')
    split = read_frame(root / 'splits' / f'{cohort}.csv', clinical.index)
    if list(split.columns) != ['fold'] or set(split.fold) != set(range(5)):
        raise ValueError('Expected five folds numbered 0 to 4')
    return clinical, split.fold.to_numpy(dtype=np.int64)


def load_omics(root, cohort):
    values, masks, ids = {}, {}, None
    for m in MODALITIES:
        frame = read_frame(root / 'data' / cohort / f'{cohort}_{m}.csv', ids)
        ids = frame.index
        if frame.shape[1] != DIMENSIONS[m]:
            raise ValueError(f'Unexpected {m} feature count')
        values[m] = np.ascontiguousarray(frame.to_numpy(dtype=np.float32))
        if m == 'meth':
            mask = read_frame(root / 'data' / 'masks' / f'{cohort}_meth_observed.csv', ids)
            if not mask.columns.equals(frame.columns) or not np.isin(mask.to_numpy(), [0, 1]).all():
                raise ValueError('Invalid methylation mask')
            masks[m] = np.ascontiguousarray(mask.to_numpy(dtype=bool))
        else:
            masks[m] = np.ones(values[m].shape, dtype=bool)
    return ids, values, masks


def load_representations(directory, ids):
    arrays = []
    for name in REPRESENTATIONS:
        frame = read_frame(Path(directory) / f'{name}.csv', ids)
        if frame.shape[1] != 64:
            raise ValueError('Expected 64 representation dimensions')
        arrays.append(np.ascontiguousarray(frame.to_numpy(dtype=np.float32)))
    return arrays


def save_representations(directory, ids, arrays):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, array in zip(REPRESENTATIONS, arrays):
        pd.DataFrame(array, index=pd.Index(ids, name='patient_id'),
                     columns=[f'd{i + 1:02d}' for i in range(64)]).to_csv(directory / f'{name}.csv', float_format='%.9g')
