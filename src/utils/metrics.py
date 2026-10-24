import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score


def compute_metrics(time, event, risk):
    risk = np.asarray(risk).reshape(-1)
    return {'CI': float(concordance_index(time, -risk, event)),
            'AUC': float(roc_auc_score(event, risk))}
