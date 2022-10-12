from sklearn.model_selection import StratifiedKFold
from skopt import gp_minimize
from .utils.reporting import LoopInfo
from .utils.training import train_model
from sklearn.utils.validation import check_X_y


def find_best_model(
        X,
        y,
        model,
        search_space,
        optimizing_metric,
        k_outer=5,
        skip_outer_folds=None,
        k_inner=5,
        skip_inner_folds=None,
        n_initial_points=5,
        n_calls=10,
        calibrate="no",
        calibrate_params=None,
        other_metrics=None,
        skopt_func=gp_minimize,
        verbose=False,
):
    """Performs nested cross validation to find the best classification model.
    The inner loop does hyperparameters tuning (using a `skopt` primitive)
    and the outer loop computes metrics for assessing the quality of the model
    without risk of overfitting bias.

    After the nested loop, the whole procedure is used with the full dataset to return
    a single model trained on all available data.

    Args:
        X (array-like of shape (n_samples, n_features) ): Features
        y (array-like of shape (n_samples,) ): Targets to predict.
            It has to be discrete (for classification), and both binary and multiclass
            targets are supported.
        model (estimator object): estimator object implementing `fit` and `predict`
            The object to use to fit the data. It can be a complex object like a pipeline
            or another composite object with hyperparameters to tune.
        search_space (list of tuple): Search space dimensions provided as a list.
            Each search dimension should be defined as an instance of
            a `skopt` `Dimension` object, that is, `Real`, `Integer` or
            `Categorical`.
            In addition to the constraints and the prior, when applicable,
            the `Dimension` object should have as name the name of the param
            in the model, using the double underscore convention for nested objects.
            See [examples](examples.md) for several examples for how to provide the search space.
        optimizing_metric (str or callable): Strategy to evaluate
            the performance of the cross-validated model on
            each inner test set, to find the best hyperparameters. It should
            follow the `sklearn` convention of *greater is better*.
            One can use:

            - a single string
            - a callable
        k_outer (int): Number of folds for the outer cross-validation.
        skip_outer_folds (list): If set, list of folds to skip during the loop,
            to reduce computational cost.
        k_inner (int): Number of folds for the inner cross-validation.
        skip_inner_folds (list): If set, list of folds to skip during the loop,
            to reduce computational cost.
        n_initial_points (int): Number of initial points to use in Skopt Optimization.
        n_calls (int): Number of additional calls to use in Skopt Optimization.
        calibrate (str): Whether to calibrate the output probabilities.
            Options:

            - "no" if no calibration at all.
            - "only_best" if only the best model on the inner loop should be calibrated
            - "all" if all inner models should be calibrated (maybe more accurate results,
               but much more time-consuming)
        calibrate_params (dict): Dictionary of params for the CalibratedClassifierCV
        other_metrics (dict): If not empty, in the report output every metric specified in this parameter
            will be computed, showing the results over the inner folds (during tuning)
            and over the outer folds (during performance evaluation).
            The parameter should be provided as a dictionary with metric names as keys
            and callables or str a values. See [examples](examples.md) for examples.
        skopt_func (callable): Minimization function of the skopt library to be used.
            Available options are:

            - gp_minimize: performs bayesian optimization using Gaussian Processes.
            - dummy_minimize: performs a random search.
            - forest_minimize: performs sequential optimization using decision trees.
            - gbrt_minimize: performs sequential optimization using gradient boosting trees.
        verbose (bool): Whether to trace progress or not.

    Returns:
        model (estimator): Model trained with the full dataset using the same procedure
            as in the inner cross validation.
        loop_info (dataclass) : Dataclass with information about the optimization process.
            The opt_info object has a method `to_dataframe()` that converts the
            information into a dataframe, for easier exploration of results.
    """
    X, y = check_X_y(X, y,
                     accept_sparse=['csc', 'csr', 'coo'],
                     force_all_finite=False, allow_nd=True)

    if skip_inner_folds is None:
        skip_inner_folds = []
    if skip_outer_folds is None:
        skip_outer_folds = []
    if other_metrics is None:
        other_metrics = []

    if calibrate_params is None:
        calibrate_params = dict()

    outer_cv = StratifiedKFold(n_splits=k_outer)
    loop_info = LoopInfo()
    for k, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
        print(f"Looping over {k} outer fold")
        if k not in skip_outer_folds:
            _, inner_loop_info = train_model(
                X_outer_train=X[train_index], y_outer_train=y[train_index],
                model=model, search_space=search_space,
                X_outer_test=X[test_index], y_outer_test=y[test_index],
                k_inner=k_inner, skip_inner_folds=skip_inner_folds, outer_kfold=k, outer_test_indexes=test_index,
                n_initial_points=n_initial_points, n_calls=n_calls,
                calibrate=calibrate, calibrate_params=calibrate_params, optimizing_metric=optimizing_metric,
                other_metrics=other_metrics, verbose=verbose, skopt_func=skopt_func)
            loop_info.extend(inner_loop_info)

    # After assessing the procedure, we repeat it on the full dataset:
    final_model, _ = train_model(
        X_outer_train=X, y_outer_train=y,
        model=model, search_space=search_space,
        X_outer_test=[], y_outer_test=[],
        k_inner=k_inner, skip_inner_folds=skip_inner_folds, outer_kfold=None, outer_test_indexes=None,
        n_initial_points=n_initial_points, n_calls=n_calls,
        calibrate=calibrate, calibrate_params=calibrate_params, optimizing_metric=optimizing_metric, other_metrics={},
        verbose=verbose, skopt_func=skopt_func)
    return final_model, loop_info
