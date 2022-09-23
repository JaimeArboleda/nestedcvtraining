from sklearn.model_selection import StratifiedKFold, KFold
from skopt.utils import use_named_args
from .pipes_and_transformers import Ensemble, IdentityTransformer
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from copy import deepcopy
from .metrics import evaluate_metrics, average_metrics
from .reporting import LoopInfo
import numpy as np
from collections import Counter, defaultdict


def _simple_train_without_undersampling(model, X, y):
    new_model = deepcopy(model)
    new_model.fit(X, y)
    return new_model


def _simple_train_with_undersampling(model, X, y, max_k_undersampling):
    # See https://github.com/w4k2/umce/blob/master/method.py
    # Firstly we analyze the training set to find majority class and to
    # establish the imbalance ratio
    counter_classes = Counter(y)
    minority_class_key = counter_classes.most_common()[-1][0]

    minority_class_idxs = np.where(y == minority_class_key)[0]
    rest_class_idxs = np.where(y != minority_class_key)[0]

    # K is the imbalanced ratio round to int (with a minimum of 2 and a max of max_k_undersamling)
    imbalance_ratio = (
            len(rest_class_idxs) / len(minority_class_idxs)
    )
    k_majority_class = int(np.around(imbalance_ratio))
    k_majority_class = k_majority_class if k_majority_class < max_k_undersampling else max_k_undersampling
    k_majority_class = k_majority_class if k_majority_class > 2 else 2

    inner_models = []
    kf = StratifiedKFold(n_splits=k_majority_class)
    for _, index in kf.split(rest_class_idxs, y[rest_class_idxs]):
        inner_model = deepcopy(model)
        idxs = np.concatenate([minority_class_idxs, rest_class_idxs[index]])
        X_train_inner, y_train_inner = X[idxs], y[idxs]
        inner_model.fit(X_train_inner, y_train_inner)
        inner_models.append(inner_model)
    ensemble_model = Ensemble(None, inner_models)

    return ensemble_model


def _train_cv(X, y, pipeline,
              model, model_name, params, optimizing_metric, other_metrics,
              k_inner, skip_inner_folds, undersampling_majority_class, num_classes,
              max_k_undersampling, calibrated, outer_kfold
              ):
    """
    This function is called in the process of optimization of hyperparameters. Given a concrete model definition, it performs:
    * Cross validation (for better measure of error of each model).
    * Using the available train set (for each loop in the cross validation), it trains the model.
    * It evaluates the performance using the available test set.
    * In case of calibrated, it calibrates each inner model against the available validation set
    * For convenience, it packs all inner models into an Ensemble, that could be further trained if not calibrated
    """
    fit_info = LoopInfo()
    fit_info.name.append(model_name)
    fit_info.best.append(False)
    fit_info.outer_kfold.append(outer_kfold)
    fit_info.params.append(params)
    inner_models = []  # List of all models built in this k-fold
    inner_metrics = defaultdict(list)  # List of all metrics

    inner_cv = StratifiedKFold(n_splits=k_inner)
    has_resampler = any([hasattr(step[1], "fit_resample") for step in pipeline.steps])
    if has_resampler:
        complete_steps = pipeline.steps + [("model", model)]
        model_to_train = Pipeline(complete_steps)
        pipeline_to_include = None
    else:
        model_to_train = model
        pipeline_to_include = pipeline
        X = pipeline.fit_transform(X, y)

    for k, (train_index, validation_index) in enumerate(inner_cv.split(X, y)):
        if k not in skip_inner_folds:
            X_inner_train, y_inner_train, X_inner_validation, y_inner_validation = (
                X[train_index],
                y[train_index],
                X[validation_index],
                y[validation_index],
            )
            if undersampling_majority_class:
                inner_model = _simple_train_with_undersampling(
                    model_to_train, X_inner_train, y_inner_train, max_k_undersampling
                )
            else:
                inner_model = _simple_train_without_undersampling(
                    model_to_train, X_inner_train, y_inner_train
                )
            if calibrated:
                inner_model = CalibratedClassifierCV(
                    inner_model, method="isotonic", cv="prefit"
                )
                inner_model.fit(X_inner_validation, y_inner_validation)
            inner_models.append(inner_model)
            y_proba = inner_model.predict_proba(X_inner_validation)
            evaluated_metrics = evaluate_metrics(y_inner_validation, y_proba, optimizing_metric, other_metrics, num_classes)
            for m in evaluated_metrics:
                inner_metrics[m].extend(evaluated_metrics[m])

    averaged_metrics = average_metrics(inner_metrics)
    fit_info.inner_test_metrics = averaged_metrics
    complete_model = Ensemble(
        pipeline_to_include, inner_models
    )
    return complete_model, fit_info


def find_best_model(loop_info: LoopInfo):
    optimizing_metrics = loop_info.inner_test_metrics["optimizing_metric"]
    index_best_model = optimizing_metrics.index(min(optimizing_metrics))
    return index_best_model


def train_model(X_outer_train, y_outer_train, model_search_spaces,
                X_outer_test, y_outer_test, k_inner, outer_kfold,
                skip_inner_folds, n_initial_points, n_calls,
                calibrated, optimizing_metric, other_metrics, num_classes,
                skopt_func, verbose, refit_best):
    loop_info = LoopInfo()
    models = []
    all_params = []

    # We can have several keys in the search space. In this case, we loop over all of them.
    for key in model_search_spaces.keys():
        pipeline = model_search_spaces[key]["pipeline"]
        if not pipeline:
            pipeline = Pipeline([("identity", IdentityTransformer())])
        model_name = key
        model = model_search_spaces[key]["model"]
        complete_steps = pipeline.steps + [("model", model)]
        complete_pipeline = Pipeline(complete_steps)
        search_space = model_search_spaces[key]["search_space"]

        @use_named_args(search_space)
        def func_to_minimize(**params):
            params_copy = params.copy()
            undersampling_majority_class = params_copy.pop(
                "undersampling_majority_class", False
            )
            max_k_undersampling = params_copy.pop(
                "max_k_undersampling", 0
            )
            all_params.append(params)

            complete_pipeline.set_params(**params_copy)
            if verbose:
                print(f"Optimizing model {model_name}\n")
                print(f"With parameters {params}\n")

            fitted_model, fitted_info = _train_cv(
                model_name=model_name, params=params,
                X=X_outer_train, y=y_outer_train, pipeline=pipeline, num_classes=num_classes,
                model=model, optimizing_metric=optimizing_metric, other_metrics=other_metrics,
                k_inner=k_inner, skip_inner_folds=skip_inner_folds, outer_kfold=outer_kfold,
                undersampling_majority_class=undersampling_majority_class,
                max_k_undersampling=max_k_undersampling, calibrated=calibrated
            )
            models.append(fitted_model)
            if len(y_outer_test) > 0:
                y_outer_test_proba = fitted_model.predict_proba(X_outer_test)
                fitted_info.outer_test_metrics = evaluate_metrics(
                    y_outer_test, y_outer_test_proba, optimizing_metric, other_metrics, num_classes
                )
            loop_info.extend(fitted_info)
            # We make it negative because in skopt the objective is minimizing a function
            return - fitted_info.inner_test_metrics["optimizing_metric"][0]

        # perform optimization
        skopt_func(
            func=func_to_minimize, dimensions=search_space,
            n_initial_points=n_initial_points,
            n_calls=n_calls,
        )


    index_best_model = find_best_model(loop_info)
    best_model = models[index_best_model]
    undersampling = all_params[index_best_model].get('undersampling_majority_class', False)

    if refit_best and not calibrated:
        # In case of calibrated, the best model has already used all available data and no further training needs to be done.
        if verbose:
            print("Training final model")
        if undersampling:
            max_k_undersampling = all_params[index_best_model].get('max_k_undersampling', 0)
            best_model = _simple_train_with_undersampling(best_model.get_complete_pipeline_to_fit(),
                                                                X_outer_train, y_outer_train, max_k_undersampling)
        else:
            best_model = best_model.get_complete_pipeline_to_fit()
            best_model.fit(X_outer_train, y_outer_train)

    return best_model, loop_info
