from imblearn.pipeline import Pipeline
import numpy as np
from collections import Counter
from .reporting import MetadataFit, prop_minority_to_rest_class
from skopt.space.transformers import Identity


class _MidasIdentity(Identity):
    """
    Convenience class for creating a Pipeline that does not perform any transformation.
    It can be handy when in combination with OptionedPostProcessTransformer.
    """
    def fit(self, X, y):
        return self

class MidasInspectTransformer(Identity):
    def fit(self, X, y):
        self.samples = len(y)
        self.prop_minority_to_rest_class = prop_minority_to_rest_class(Counter(y))
        return self


def wrap_pipeline(pipeline):
    wrapped_steps = [('first_inspect_MIDAS', MidasInspectTransformer())] \
                    + pipeline.steps \
                    + [('last_inspect_MIDAS', MidasInspectTransformer())]
    return Pipeline(wrapped_steps)


def unwrap_pipeline(pipeline):
    unwrapped_steps = [(name, transformer)
                       for (name, transformer) in pipeline.steps
                       if name not in ['first_inspect_MIDAS', 'last_inspect_MIDAS']]
    return Pipeline(unwrapped_steps)


def get_metadata_fit(pipeline):
    steps = pipeline.steps
    first_inspect_MIDAS = [transformer for name, transformer in steps if name == 'first_inspect_MIDAS'][0]
    last_inspect_MIDAS = [transformer for name, transformer in steps if name == 'last_inspect_MIDAS'][0]
    return MetadataFit(
        num_init_samples_bf=first_inspect_MIDAS.samples,
        num_init_samples_af=last_inspect_MIDAS.samples,
        prop_minority_class_bf=first_inspect_MIDAS.prop_minority_to_rest_class,
        prop_minority_class_af=last_inspect_MIDAS.prop_minority_to_rest_class
    )


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
        raise NotImplementedError("This ensemble model is already fitted and cannot be refitted")

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

    def get_complete_pipeline_to_fit(self):
        if isinstance(self.list_estimators[0], MidasEnsembleClassifiersWithPipeline):
            return self.list_estimators[0].get_complete_pipeline_to_fit()
        else:
            if self.post_process_pipeline:
                complete_steps = self.post_process_pipeline.steps + [("model", self.list_estimators[0])]
            else:
                complete_steps = [("model", self.list_estimators[0])]
            return Pipeline(complete_steps)


