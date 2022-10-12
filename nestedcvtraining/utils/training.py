from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from skopt.utils import use_named_args
from sklearn.metrics._scorer import _MultimetricScorer, _check_multimetric_scoring
from sklearn.calibration import CalibratedClassifierCV
from .reporting import LoopInfo
import numpy as np


def evaluate_metrics(cv_results):
    evaluations = {}
    for key in cv_results:
        if key.startswith("test_"):
            evaluations[key[5:]] = [np.mean(cv_results[key])]
    return evaluations


def yield_splits(X, y, k, skip):
    cv = StratifiedKFold(n_splits=k)
    for cv_idx, (train_index, test_index) in enumerate(cv.split(X, y)):
        if k not in skip:
            yield train_index, test_index


def find_best_model(loop_info: LoopInfo):
    optimizing_metrics = loop_info.inner_validation_metrics["optimizing_metric"]
    index_best_model = optimizing_metrics.index(max(optimizing_metrics))
    return index_best_model


def train_model(X_outer_train, y_outer_train, model, search_space,
                X_outer_test, y_outer_test, k_inner, outer_kfold, outer_test_indexes,
                skip_inner_folds, n_initial_points, n_calls,
                calibrate, calibrate_params, optimizing_metric, other_metrics,
                skopt_func, verbose):
    loop_info = LoopInfo()
    all_metrics = {
        "optimizing_metric": optimizing_metric,
    }
    all_metrics.update(other_metrics)

    @use_named_args(search_space)
    def func_to_minimize(**params):
        if verbose:
            print(f"Optimizing model with parameters {params}")

        model_to_fit = model.set_params(**params)
        if calibrate == "all":
            model_to_fit = CalibratedClassifierCV(model_to_fit, **calibrate_params)

        cv_results = cross_validate(
            estimator=model_to_fit, X=X_outer_train, y=y_outer_train,
            cv=yield_splits(X_outer_train, y_outer_train, k_inner, skip_inner_folds),
            scoring=all_metrics, return_train_score=False, return_estimator=False
        )
        inner_metrics = evaluate_metrics(cv_results)
        outer_metrics = {k: [None] for k in inner_metrics}
        loop_info.append(
            False, outer_kfold, params, inner_metrics, outer_metrics, None, outer_test_indexes
        )

        # We make it negative because in skopt the objective is minimizing a function
        return - inner_metrics["optimizing_metric"][0]

    # perform optimization
    skopt_func(
        func=func_to_minimize,
        dimensions=search_space,
        n_initial_points=n_initial_points,
        n_calls=n_calls,
    )

    index_best_model = find_best_model(loop_info)
    loop_info.best[index_best_model] = True
    best_params = loop_info.params[index_best_model]
    best_model = model.set_params(**best_params)

    if verbose:
        print("Training final model")

    if calibrate != "no":
        best_model = CalibratedClassifierCV(best_model, **calibrate_params)
    best_model.fit(X_outer_train, y_outer_train)
    loop_info.model[index_best_model] = best_model

    # For the best model, we need to fill the outer metrics
    if len(y_outer_test) > 0:
        metrics = _check_multimetric_scoring(best_model, all_metrics)
        scorer = _MultimetricScorer(**metrics)
        scores = scorer(best_model, X_outer_test, y_outer_test)
        for k in loop_info.outer_test_metrics:
            loop_info.outer_test_metrics[k][index_best_model] = scores[k]

    return best_model, loop_info
