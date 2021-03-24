from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss, average_precision_score, log_loss
import functools
import numpy as np


def histogram_width(y_true, y_proba):
    return 4 * (np.sum((y_proba - 0.5) ** 2) / len(y_proba))


METRICS = {
    'roc_auc': {
        'score_type': 'gain',
        'func': roc_auc_score
    },
    'neg_log_loss': {
        'score_type': 'loss',
        'func': log_loss
    },
    'average_precision': {
        'score_type': 'gain',
        'func': average_precision_score
    },
    'neg_brier_score': {
        'score_type': 'loss',
        'func': brier_score_loss
    },
    'histogram_width': {
        'score_type': 'gain',
        'func': histogram_width
    }
}


def neg_score(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        score = func(*args, **kwargs)
        return 1.0 - score
    return wrapper


def get_metric(name, option):
    if name not in METRICS.keys():
        raise ValueError("Metric not supported: " + name)
    if option not in {'loss', 'real'}:
        raise TypeError("Option not supported: " + option)
    metric = METRICS[name]
    if option == 'real' or metric['score_type'] == 'loss':
        return metric['func']
    else:
        return neg_score(metric['func'])


def evaluate_metrics(y_true, y_proba, loss_metric, peeking_metrics):
    evaluations = {}
    evaluations['loss_metric'] = get_metric(loss_metric, 'loss')(y_true, y_proba)
    evaluations['peeking_metrics'] = {}
    for peeking_metric in peeking_metrics:
        evaluations['peeking_metrics'][peeking_metric] = get_metric(peeking_metric, 'real')(y_true, y_proba)
    return evaluations


def average_metrics(fold_metrics):
    metrics = {}
    metrics['loss_metric'] = np.mean([metric['loss_metric'] for metric in fold_metrics])
    metrics['peeking_metrics'] = {}
    for peeking_metric in fold_metrics[0]['peeking_metrics'].keys():
        metrics['peeking_metrics'][peeking_metric] = np.mean([metric['peeking_metrics'][peeking_metric]
                                                              for metric in fold_metrics])
    return metrics


def is_supported(name):
    return name in METRICS.keys()
