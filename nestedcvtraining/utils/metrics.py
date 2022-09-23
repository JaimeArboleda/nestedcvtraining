import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class DummyClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y):
        self.classes_ = np.array(range(self.num_classes))
        return self

    def predict(self, X):
        return X

    def predict_proba(self, X):
        return X


def make_scorer_for_predicted_values(scorer, num_classes):
    dc = DummyClassifier(num_classes).fit(None, None)

    def new_scorer(y_true, y_proba):
        return scorer(dc, y_proba, y_true)

    return new_scorer


def evaluate_metrics(y_true, y_proba, optimizing_metric, other_metrics, num_classes):
    evaluations = {}
    evaluations['optimizing_metric'] = [
        make_scorer_for_predicted_values(optimizing_metric, num_classes)(y_true, y_proba)]
    for metric in other_metrics:
        evaluations[metric._score_func.__name__] = [
            make_scorer_for_predicted_values(metric, num_classes)(y_true, y_proba)]
    return evaluations


def average_metrics(evaluated_metrics):
    return {
        k: [np.mean(evaluated_metrics[k])]
        for k in evaluated_metrics
    }

