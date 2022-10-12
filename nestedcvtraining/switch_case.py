"""Implements a SwitchCase or *fork* transformer.

This class makes it easy to design, in a modular way, complex
pipelines, with both discrete choices and continuous hyperparameters
for hyperparameter tuning.

The `SwitchCase` can handle `estimators` implementing `predict`,
`transformers` implementing `transform` and `imblearn` `resamplers`
implementing `fit_resample`.

It's also possible to include `sklearn` `Pipelines` inside the
`cases`. However, `imblearn` `Pipelines` should be included with
extra care, as their internal design does not allow them to be nested.
See [examples](examples.md) for worked out examples and explanation of
this restriction.

"""

from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.metaestimators import available_if
from sklearn.utils._estimator_html_repr import _VisualBlock
import numpy as np
from sklearn.utils.validation import check_is_fitted


def _current_transformer_has(attr):
    """Check that current transformer has `attr`.
    Used together with `avaliable_if` """

    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self.current_transformer, attr)
        return True

    return check


class SwitchCase(_BaseComposition):
    """Selectively applies result of one transformer object depending on switch.
    Args:
        cases (list of tuples (str, transformer) ): Transformer objects to be applied
            to the data, conditioned on the value of the next parameter (switch). The first half of
            each tuple is the name of the transformer, and it has two functions:
            to determine (along with the switch) what is the transformer that will be applied
            and to specify the parameters of the transformer using double underscore.
            If transformer == "passthrough", it will apply the identity function.
            Transformers can be:

            - Sklearn transformers implementing at least `fit` and `transform`.
            - Sklearn estimators implementing at least `fit` and `predict`.
            - Imblearn resamplers implementing at least `fit_resample`.
        switch (str): It determines the transformer to be applied to the data.
    """

    _required_parameters = ["cases", "switch"]

    def __init__(
        self, cases, switch
    ):
        self.cases = cases
        self.switch = switch
        self._validate_switch()
        self._validate_transformers()

    @property
    def cases_names(self):
        return [name for name, _ in self.cases]

    @property
    def current_transformer(self):
        for name, trans in self.cases:
            if trans == "passthrough":
                trans = FunctionTransformer()
            if name == self.switch:
                return trans

    @property
    def current_transformer_idx(self):
        idx = 0
        for name, trans in self.cases:
            if name == self.switch:
                return idx
            idx += 1

    def _validate_switch(self):
        if self.switch not in self.cases_names:
            raise AttributeError(f"Value of switch {self.switch} is not among cases names {self.cases_names}")

    def _validate_transformers(self):
        names, transformers = zip(*self.cases)

        self._validate_names(names)

        for t in transformers:
            if t == "passthrough":
                continue
            if not (
                hasattr(t, "fit") or hasattr(t, "fit_transform") or hasattr(t, "fit_resample")
            ):
                raise TypeError(
                    "All transformers should implement fit or fit_transform or "
                    "fit_resample (but not both) or be a string 'passthrough' "
                    "'%s' (type %s) doesn't)" % (t, type(t))
                )

            if hasattr(t, "fit_resample") and (
                hasattr(t, "fit_transform") or hasattr(t, "transform")
            ):
                raise TypeError(
                    "All transformers should implement fit and transform or fit_resample."
                    " '%s' implements both)" % t
                )

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Returns the parameters given in the constructor as well as the
        estimators contained within the `cases` of the
        `SwitchCase`.
        Args:
            deep (bool): If True, will return the parameters for this estimator and
                contained subobjects that are estimators.
        Returns:
            params (dict): mapping of string to any
                Parameter names mapped to their values.
        """
        return self._get_params("cases", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``. Note that
        you can directly set the parameters of the estimators contained in
        `cases`.
        Args:
            **kwargs (dict): Parameters of this estimator or parameters of estimators contained
                in `cases`. Parameters of the transformers may be set
                using its name and the parameter name separated by a '__'.
        Returns:
            self (object): SwitchCase class instance.
        """
        self._set_params("cases", **kwargs)
        self._validate_switch()
        self._validate_transformers()
        return self

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.
        Args:
            input_features (array-like of str or None): Input features.
        Returns:
            feature_names_out (ndarray of str): Transformed feature names.
        """
        transformer = self.current_transformer
        name = self.cases[self.current_transformer_idx][0]
        feature_names = []
        feature_names.extend(
            [f"{name}__{f}" for f in transformer.get_feature_names_out(input_features)]
        )
        return np.asarray(feature_names, dtype=object)

    @available_if(_current_transformer_has("fit"))
    def fit(self, X, y=None, **fit_params):
        idx = self.current_transformer_idx
        transformer = self.current_transformer
        transformer.fit(X, y, **fit_params)
        self.cases[idx] = (self.cases[idx][0], transformer)
        return self

    @available_if(_current_transformer_has("fit_transform"))
    def fit_transform(self, X, y=None, **fit_params):
        transformer = self.current_transformer
        idx = self.current_transformer_idx
        Xt = transformer.fit_transform(X, y, **fit_params)
        self.cases[idx] = (self.cases[idx][0], transformer)
        return Xt

    @available_if(_current_transformer_has("fit_resample"))
    def fit_resample(self, X, y=None, **fit_params):
        transformer = self.current_transformer
        idx = self.current_transformer_idx
        X_res, y_res = transformer.fit_resample(X, y, **fit_params)
        self.cases[idx] = (self.cases[idx][0], transformer)
        return X_res, y_res

    @available_if(_current_transformer_has("transform"))
    def transform(self, X):
        transformer = self.current_transformer
        return transformer.transform(X)

    @available_if(_current_transformer_has("predict"))
    def predict(self, X):
        transformer = self.current_transformer
        return transformer.predict(X)

    @available_if(_current_transformer_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        transformer = self.current_transformer
        idx = self.current_transformer_idx
        y_pred = transformer.fit_predict(X, y, **fit_params)
        self.cases[idx] = (self.cases[idx][0], transformer)
        return y_pred

    @available_if(_current_transformer_has("predict_proba"))
    def predict_proba(self, X):
        transformer = self.current_transformer
        return transformer.predict_proba(X)

    @available_if(_current_transformer_has("decision_function"))
    def decision_function(self, X):
        transformer = self.current_transformer
        return transformer.decision_function(X)

    @available_if(_current_transformer_has("score_samples"))
    def score_samples(self, X):
        transformer = self.current_transformer
        return transformer.score_samples(X)

    @available_if(_current_transformer_has("predict_log_proba"))
    def predict_log_proba(self, X):
        transformer = self.current_transformer
        return transformer.predict_log_proba(X)

    @available_if(_current_transformer_has("inverse_transform"))
    def inverse_transform(self, Xt):
        transformer = self.current_transformer
        return transformer.inverse_transform(Xt)

    @available_if(_current_transformer_has("score"))
    def score(self, X, y=None, sample_weight=None):
        transformer = self.current_transformer
        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight
        return transformer.score(X, y, **score_params)

    @property
    def n_features_in_(self):
        transformer = self.current_transformer
        return transformer.n_features_in_

    @property
    def feature_names_in(self):
        transformer = self.current_transformer
        return transformer.feature_names_in_

    @property
    def classes_(self):
        transformer = self.current_transformer
        return transformer.classes_

    def __sklearn_is_fitted__(self):
        transformer = self.current_transformer
        check_is_fitted(transformer)
        return True

    def _sk_visual_block_(self):
        names, transformers = zip(*self.cases)
        return _VisualBlock("parallel", transformers, names=names)
