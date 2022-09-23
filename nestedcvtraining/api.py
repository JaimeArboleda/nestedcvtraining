import sklearn.pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from skopt import gp_minimize
from skopt.space import Dimension
from .utils.reporting import LoopInfo
from .utils.training import train_model
from collections import Counter
from sklearn.base import TransformerMixin
from sklearn.metrics._scorer import _BaseScorer
from sklearn.utils.validation import check_X_y
from inspect import isfunction, signature


def find_best_model(
        X,
        y,
        model_search_spaces,
        optimizing_metric,
        k_outer=5,
        skip_outer_folds=None,
        k_inner=5,
        skip_inner_folds=None,
        n_initial_points=5,
        n_calls=10,
        calibrated=False,
        other_metrics=None,
        skopt_func=gp_minimize,
        verbose=False,
):
    """Finds the best calibrated classification model and provides
    a dataframe with information of the performed nested cross validation. In the inner
    cross validation, doing a Bayesian Search, the best parameters are found.
    In the outer cross validation, the model is validated.
    Finally, the whole procedure is used for the full dataset to return
    the best possible model.


    Parameters
    ----------
    X : np.array
        Feature set.

    y : np.array
        Classification target to predict. It works for multiclass and binary classifications.

    optimizing_metric : callable, default=None
        Metric to use in order to find the best parameters in Bayesian Search.

    model_search_spaces : Dict[str : List[List[skopt.Space]]
        Dict of models to try inside the inner loops. For each model, there is
        the corresponding list of space objects to delimit where the parameters live,
        including the pipeline postprocess to make. It admits also an option to set
        undersampling_majority_class method. It admits two values, True or False. If True
        it builds an ensemble model in the inner loop by creating many balanced folds
        by using the minority class with undersampling of the majority class. If using
        this option, it also admits an Int max_k_undersampling, in order to limit the number of
        splits made for this (because if the imbalance ratio is for example 1000:1, it will
        create 1000 splits, which can be too much).

    k_outer : int, default=5
        Number of folds for the outer cross-validation.

    skip_outer_folds : list, default=None
        If set, list of folds to skip during the loop.

    k_inner : int, default=5
        Number of folds for the inner cross-validation.

    skip_inner_folds : list, default=None
        If set, list of folds to skip during the loop.

    n_initial_points : int, default=5
        Number of initial points to use in Bayesian Optimization.

    n_calls : int, default=5
        Number of additional calls to use in Bayesian Optimization.

    calibrated : bool, default=False
        If True, all models are calibrated using CalibratedClassifierCV

    other_metrics : List[str], default=[]
        If not empty, in the report there will be a comparison between the metric of
        evaluation on the inner fold and the list of metrics in peeking_metrics.

    skopt_func : callable, default=gp_minimize
        Minimization function of the skopt library to be used.

    verbose : bool, default=False
        If True, you can trace the progress in the terminal.

    Returns
    -------
    model : Model trained with the full dataset using the same procedure
            as in the inner cross validation.
    report_dfs : Dict of dataframes, one key for each model in model_search_spaces.
                 each key, a dataframe with all inner models built with their
                 params and loss_metric.
    """
    # Validation of inputs
    X, y = check_X_y(X, y,
                     accept_sparse=['csc', 'csr', 'coo'],
                     force_all_finite=False, allow_nd=True)
    counter = Counter(y)

    y_values = set(counter.keys())
    if y_values != set(range(len(counter))):
        raise NotImplementedError("Values of target are expected to be in {0, 1, ..., n-1} where n is the number of "
                                  "classes")

    if not _validate_model_search_space(model_search_spaces):
        raise ValueError("model_search_spaces is not well formed")

    if skip_inner_folds is None:
        skip_inner_folds = []
    if skip_outer_folds is None:
        skip_outer_folds = []
    if other_metrics is None:
        other_metrics = []

    if not _validate_folds(k_outer, skip_outer_folds, k_inner, skip_inner_folds):
        raise ValueError("Folds parameters are not well formed")

    if not _validate_bayesian_search(n_initial_points, n_calls, skopt_func):
        raise ValueError("Bayesian search parameters are not well formed")

    if not isinstance(optimizing_metric, _BaseScorer):
        raise NotImplementedError(f"The metric must be provided as a scoring function, as output by sklearn make_scorer")

    if not isinstance(other_metrics, list):
        raise ValueError("Peeking metrics must be a list of str")

    for metric in other_metrics:
        if not isinstance(metric, _BaseScorer):
            raise NotImplementedError(f"The metric must be provided as a scoring function, as output by sklearn "
                                      f"make_scorer")

    if not isinstance(calibrated, bool):
        raise ValueError("calibrated must be a boolean")

    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean")

    outer_cv = StratifiedKFold(n_splits=k_outer)
    loop_info = LoopInfo()
    for k, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
        if k not in skip_outer_folds:
            _, inner_loop_info = train_model(
                X_outer_train=X[train_index], y_outer_train=y[train_index], model_search_spaces=model_search_spaces,
                X_outer_test=X[test_index], y_outer_test=y[test_index],
                k_inner=k_inner, skip_inner_folds=skip_inner_folds,outer_kfold=k,
                n_initial_points=n_initial_points, n_calls=n_calls,
                calibrated=calibrated, optimizing_metric=optimizing_metric, other_metrics=other_metrics,
                verbose=verbose, skopt_func=skopt_func, refit_best=False, num_classes=len(y_values))
            loop_info.extend(inner_loop_info)


    # After assessing the procedure, we repeat it on the full dataset:
    final_model = None
    final_model, _ = train_model(
        X_outer_train=X, y_outer_train=y, model_search_spaces=model_search_spaces,
        X_outer_test=[], y_outer_test=[],
        k_inner=k_inner, skip_inner_folds=skip_inner_folds,outer_kfold=None,
        n_initial_points=n_initial_points, n_calls=n_calls,
        calibrated=calibrated, optimizing_metric=optimizing_metric, other_metrics=[],
        verbose=verbose, skopt_func=skopt_func, refit_best=True, num_classes=len(y_values))
    return final_model, loop_info.to_dataframe()


class OptionedTransformer(TransformerMixin):
    """This class converts a dictionary of different options of post-process
    (each option will be a fully defined pipeline) in a transformer that
    has only one parameter, the option.
    In this way, you can perform a bayesian optimization search including a postprocess
    pipeline that uses this set of categorical fixed options.

    Example of input dict:
        dict_pipelines_post_process = {
        "option_1": Pipeline(
            [("scale", StandardScaler()), ("reduce_dims", PCA(n_components=5))]
        ),
        "option_2": Pipeline(
            [
                ("scale", StandardScaler()),
                ("reduce_dims", SelectKBest(mutual_info_classif, k=5)),
            ]
        ),
        "option_3": Pipeline([("identity", MidasIdentity())])
        }
    """

    def __init__(self, dict_pipelines):
        if not self._validate_dict(dict_pipelines):
            raise ValueError("The dictionary is not well formed. Please check the docs and examples. ")
        if not self._validate_resamplers(dict_pipelines):
            raise ValueError("OptionedPostProcessTransformer does not "
                             "support resamplers inside ")
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

    def _validate_dict(self, dict_pipelines):
        check_pipelines = any(
            [
                not isinstance(pipeline, sklearn.pipeline.Pipeline)
                for pipeline in dict_pipelines.values()
            ]
        )
        return not check_pipelines

    def _validate_resamplers(self, dict_pipelines):
        all_steps = []
        for pipeline in dict_pipelines.values():
            all_steps.extend(pipeline.steps)
        exists_resampler = any(
            [
                hasattr(step[1], "fit_resample")
                for step in all_steps
            ]
        )
        return not exists_resampler


def _validate_model_search_space(model_search_spaces):
    if not isinstance(model_search_spaces, dict):
        raise ValueError("model_search_spaces must be a dict")
    for model_search in model_search_spaces.values():
        if not isinstance(model_search, dict):
            raise ValueError("model_search_spaces must be a dict of dicts")
        for (key, value) in model_search.items():
            if key not in ['model', 'pipeline', 'search_space']:
                raise ValueError("Some inner key is not in ['model', 'pipeline', 'search_space']")
            if key == 'model':
                if not hasattr(value, 'predict_proba'):
                    raise ValueError("Estimator has not predict_proba method")
                if not hasattr(value, 'fit'):
                    raise ValueError("Estimator has not fit method")
            if key == 'pipeline':
                if value:
                    if not isinstance(value, sklearn.pipeline.Pipeline):
                        raise ValueError("pipeline is not a pipeline")
            if key == 'search_space':
                if not isinstance(value, list):
                    raise ValueError("search_space is not a list")
                for element in value:
                    if not isinstance(element, Dimension):
                        raise ValueError("search_space is not valid")
    return True


def _validate_folds(k_outer_fold, skip_outer_folds, k_inner_fold, skip_inner_folds):
    if not isinstance(k_outer_fold, int):
        raise ValueError("k_outer_fold must be int")
    if not isinstance(k_inner_fold, int):
        raise ValueError("k_inner_fold must be int")
    if not isinstance(skip_outer_folds, list):
        raise ValueError("skip_outer_folds must be a list")
    if not isinstance(skip_inner_folds, list):
        raise ValueError("k_outer_fold must be a list")
    for element in skip_outer_folds:
        if element not in list(range(k_outer_fold)):
            raise ValueError("skip_outer_folds must be contained in [0, k_outer_folds-1]")
    for element in skip_inner_folds:
        if element not in list(range(k_inner_fold)):
            raise ValueError("skip_inner_folds must be contained in [0, k_inner_fold-1]")
    return True


def _validate_bayesian_search(n_initial_points, n_calls, skopt_func):
    if not isinstance(n_initial_points, int):
        raise ValueError("n_initial_points must be int")
    if not isinstance(n_calls, int):
        raise ValueError("n_calls must be int")
    if n_initial_points < 0 or n_calls < 0:
        raise ValueError("n_calls and n_initial_points must positive")
    if n_calls < n_initial_points:
        raise ValueError("n_calls cannot be lesser than n_initial_points")
    if not isfunction(skopt_func):
        raise ValueError("skopt_func must be callable")
    signature_skopt_func = signature(skopt_func)
    kwargs = [parameter for parameter in signature_skopt_func.parameters]
    if 'func' not in kwargs:
        raise ValueError("skopt_func must have func in kwargs")
    if 'dimensions' not in kwargs:
        raise ValueError("skopt_func must have dimensions in kwargs")
    if 'n_initial_points' not in kwargs:
        raise ValueError("skopt_func must have n_initial_points in kwargs")
    if 'n_calls' not in kwargs:
        raise ValueError("skopt_func must have n_calls in kwargs")
    return True
