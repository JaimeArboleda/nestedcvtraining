from imblearn.pipeline import Pipeline
import numpy as np
from collections import Counter
from skopt.space.transformers import Identity


class IdentityTransformer(Identity):
    """
    Convenience class for creating a Pipeline that does not perform any transformation.
    It can be handy when in combination with OptionedPostProcessTransformer.
    """

    def fit(self, X, y):
        return self


class Ensemble:
    '''
    Base class for an ensemble of fitted classifiers with a fitted pipeline having a common transformation.
    '''

    def __init__(self, pipeline, estimators):
        self.pipeline = pipeline
        self.estimators = estimators
        self.classes_ = self.estimators[0].classes_

    def fit(self, X, y):
        # It's already fitted
        raise NotImplementedError("This ensemble model is already fitted and cannot be refitted")

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        if self.pipeline:
            X_t = self.pipeline.transform(X)
        else:
            X_t = X

        mean_proba = np.zeros((X_t.shape[0], len(self.classes_)))
        for classifier in self.estimators:
            proba = classifier.predict_proba(X_t)
            mean_proba += proba
        mean_proba /= len(self.estimators)
        return mean_proba

    def get_complete_pipeline_to_fit(self):
        if isinstance(self.estimators[0], Ensemble):
            return self.estimators[0].get_complete_pipeline_to_fit()
        else:
            if self.pipeline:
                complete_steps = self.pipeline.steps + [("model", self.estimators[0])]
            else:
                complete_steps = [("model", self.estimators[0])]
            return Pipeline(complete_steps)

