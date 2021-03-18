from sklearn.base import TransformerMixin
from sklearn.utils import _print_elapsed_time
from imblearn.pipeline import Pipeline
from skopt.space.transformers import Identity
import numpy as np
from collections import Counter
from .reporting import MetadataFit, prop_minority_class


class MidasPipeline(Pipeline):

    def fit(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)
        self._metadata_fit = MetadataFit(
            num_init_samples_bf=len(y),
            num_init_samples_af=len(yt),
            prop_minority_class_bf=prop_minority_class(Counter(y)),
            prop_minority_class_af=prop_minority_class(Counter(yt)))
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, yt, **fit_params_last_step)
        return self

    def get_metadata_fit(self):
        return self._metadata_fit


class MidasIdentity(Identity):
    def fit(self, X, y):
        return self


class MidasEnsembleClassifiersWithPipeline:
    '''
    Base class for an ensemble of fitted classifiers with a fitted pipeline having a common transformation.
    '''

    def __init__(self, post_process_pipeline, list_estimators):
        self.post_process_pipeline = post_process_pipeline
        self.list_estimators = list_estimators
        self.classes_ = self.list_estimators[0].classes_

    def fit(self, X, y):
        # It's already fitted
        return

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        if self.post_process_pipeline:
            X_t = self.post_process_pipeline.transform(X)
        else:
            X_t = X

        mean_proba = np.zeros((X_t.shape[0], len(self.classes_)))
        for classifier in self.list_estimators:
            proba = classifier.predict_proba(X_t)
            mean_proba += proba
        mean_proba /= len(self.list_estimators)
        return mean_proba


class OptionedPostProcessTransformer(TransformerMixin):

    def __init__(self, dict_pipelines):
        # TODO: Check if there is a resampler inside
        self.dict_pipelines = dict_pipelines
        self.option = list(dict_pipelines.keys())[0]
        super().__init__()

    def fit(self, X, y=None):
        self.dict_pipelines[self.option].fit(X, y)
        return self

    def set_params(self, **params):
        self.option = params['option']
        return self

    def transform(self, X):
        return self.dict_pipelines[self.option].transform(X)

    def fit_transform(self, X, y=None):
        return self.dict_pipelines[self.option].fit_transform(X, y)