from collections import Counter

import numpy as np
from joblib import Parallel
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.fixes import delayed
from sklearn.base import ClassifierMixin, MetaEstimatorMixin, BaseEstimator, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


def _fit_estimator(estimator, X, y, **fit_params):
    estimator = clone(estimator)
    estimator.fit(X, y, **fit_params)
    return estimator


class UnderSamplingClassifier(
    ClassifierMixin, MetaEstimatorMixin, BaseEstimator
):
    """ Under Sampling Classifier

    It trains an ensemble of estimators on several folds obtained
    by using all samples of the minority class and undersampling
    the rest of classes.

    It implements the strategy described [here](http://proceedings.mlr.press/v94/ksieniewicz18a/ksieniewicz18a.pdf).

    It is compatible with having a resampler before it, as long as the resampler only
    performs a partial reduction of the imbalance problem.

    Args:
        estimator (estimator object): An estimator
            object implementing `fit` and `predict_proba`

        n_jobs (int, optional): The number
            of jobs to use for the computation

    Attributes:
        estimators_ (list): list of fitted estimators
            used for predictions.
        classes_ (array): Class labels
        n_classes_ (int): Number of classes.
        label_encoder_ (LabelEncoder object): LabelEncoder object
            used to encode multiclass labels
        n_features_in_ (int): Number of features seen during `fit`. Only defined if the
            underlying estimator exposes such an attribute when fit.
        feature_names_in_ (ndarray): Names of features seen during `fit`.
            Only defined if the underlying estimator exposes such an attribute when fit.
    """

    def __init__(self, estimator, *, max_k_under_sampling=5, n_jobs=None):
        self.estimator = estimator
        self.max_k_under_sampling = max_k_under_sampling
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit underlying estimators by undersampling all classes but the minority class.
        Args:
            X (array-like of shape (n_samples, n_features) ): Features
            y (array-like of shape (n_samples,) ): Targets
        Returns:
            self : object
                Instance of fitted estimator.
        """
        self.label_encoder_ = LabelEncoder()
        Y = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        counter_classes = Counter(Y)
        minority_class_key = counter_classes.most_common()[-1][0]

        minority_class_idxs = np.where(y == minority_class_key)[0]
        rest_idxs = np.where(y != minority_class_key)[0]

        # K is the imbalanced ratio round to int (with a minimum of 2 and a max of max_k_undersamling)
        imbalance_ratio = (
                len(rest_idxs) / len(minority_class_idxs)
        )
        k_majority_class = int(np.around(imbalance_ratio))
        k_majority_class = k_majority_class if k_majority_class < self.max_k_under_sampling else self.max_k_under_sampling
        k_majority_class = k_majority_class if k_majority_class > 2 else 2

        def under_sampling_iterator():
            splitter = StratifiedKFold(n_splits=k_majority_class)
            for _, index in splitter.split(rest_idxs, y[rest_idxs]):
                idxs = np.concatenate([minority_class_idxs, rest_idxs[index]])
                yield idxs

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator,
                X[idxs],
                y[idxs]
            )
            for idxs in under_sampling_iterator()
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def predict(self, X):
        check_is_fitted(self)
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X):
        check_is_fitted(self)
        mean_proba = np.zeros((X.shape[0], len(self.classes_)))
        for classifier in self.estimators_:
            proba = classifier.predict_proba(X)
            mean_proba += proba
        mean_proba /= len(self.estimators_)
        return mean_proba

    @property
    def n_classes_(self):
        """Number of classes."""
        return len(self.classes_)
