from sklearn.model_selection import StratifiedKFold, KFold
from skopt.utils import use_named_args
from .pipes_and_transformers import MidasEnsembleClassifiersWithPipeline, wrap_pipeline, get_metadata_fit, _MidasIdentity
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from copy import deepcopy
from .metrics import evaluate_metrics, average_metrics
from .reporting import (
    write_train_report,
    prop_minority_to_rest_class,
    MetadataFit,
    averaged_metadata_list, create_report_dfs,
)
import numpy as np
from collections import Counter

def exists_resampler(pipeline):
    return any(
            [
                hasattr(step[1], "fit_resample")
                for step in pipeline.steps
            ]
    )


def train_model_without_undersampling(model, X, y, exists_resampler):
    fold_model = deepcopy(model)
    fold_model.fit(X, y)
    if exists_resampler:  # In this case, model is a pipeline with a resampler.
        metadata = get_metadata_fit(fold_model)
    else:
        metadata = MetadataFit(len(y), prop_minority_to_rest_class(Counter(y)))
    return fold_model, metadata


def train_ensemble_model_with_undersampling(model, X, y, exists_resampler, max_k_undersampling):
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

    fold_models = []
    list_metadata = []
    kf = KFold(n_splits=k_majority_class)
    for _, index in kf.split(rest_class_idxs):
        fold_model = deepcopy(model)
        fold_idx = np.concatenate([minority_class_idxs, rest_class_idxs[index]])
        X_train_f, y_train_f = X[fold_idx], y[fold_idx]
        fold_model.fit(X_train_f, y_train_f)
        fold_models.append(fold_model)
        if exists_resampler:  # In this case, model is a pipeline with a resampler.
            list_metadata.append(get_metadata_fit(fold_model))
        else:
            list_metadata.append(
                MetadataFit(len(y_train_f), prop_minority_to_rest_class(Counter(y_train_f)))
            )
    ensemble_model = MidasEnsembleClassifiersWithPipeline(None, fold_models)

    return ensemble_model, averaged_metadata_list(list_metadata)


def ensemble_model_with_resampling(X, y, pipeline_post_process,
                                   model, loss_metric, peeking_metrics,
                                   k_inner_fold, skip_inner_folds, undersampling_majority_class,
                                   max_k_undersampling, calibrated
                                   ):

    pipeline_post_process = wrap_pipeline(pipeline_post_process)
    complete_steps = pipeline_post_process.steps + [("model", model)]
    complete_pipeline = Pipeline(complete_steps)
    fold_models = []
    fold_metrics = []
    list_metadata = []
    comments = {}
    comments["option"] = "build model with resampling"
    inner_cv = StratifiedKFold(n_splits=k_inner_fold)
    for k, (train_index, test_index) in enumerate(inner_cv.split(X, y)):
        if k not in skip_inner_folds:
            X_train, y_train, X_test, y_test = (
                X[train_index],
                y[train_index],
                X[test_index],
                y[test_index],
            )
            if undersampling_majority_class:
                (
                    fold_base_model,
                    fold_metadata,
                ) = train_ensemble_model_with_undersampling(
                    complete_pipeline, X_train, y_train, True, max_k_undersampling
                )
            else:
                fold_base_model, fold_metadata = train_model_without_undersampling(
                    complete_pipeline, X_train, y_train, True
                )
            list_metadata.append(fold_metadata)
            fold_final_model = fold_base_model
            if calibrated:
                fold_final_model = CalibratedClassifierCV(
                    fold_base_model, method="isotonic", cv="prefit"
                )
                fold_final_model.fit(X_test, y_test)
            fold_models.append(fold_final_model)
            y_proba = fold_final_model.predict_proba(X_test)[:, 1]
            fold_metrics.append(
                evaluate_metrics(y_test, y_proba, loss_metric, peeking_metrics)
            )
    averaged_metrics = average_metrics(fold_metrics)
    complete_model = MidasEnsembleClassifiersWithPipeline(None, fold_models)
    metadata = averaged_metadata_list(list_metadata)
    comments["number of folds"] = len(fold_models)
    comments[
        "average size of training set before resampling"
    ] = metadata.get_num_init_samples_bf()
    comments[
        "average prop of minority class before resampling"
    ] = metadata.get_prop_minority_class_bf()
    comments[
        "average size of training set after resampling"
    ] = metadata.get_num_init_samples_af()
    comments[
        "average prop of minority class after resampling"
    ] = metadata.get_prop_minority_class_af()
    return complete_model, averaged_metrics, comments


def ensemble_model_without_resampling(X, y, pipeline_post_process,
                                      model, loss_metric, peeking_metrics,
                                      k_inner_fold, skip_inner_folds, undersampling_majority_class,
                                      max_k_undersampling, calibrated
                                      ):

    list_metadata = []
    fold_models = []  # List of all models builded in this k-fold
    fold_metrics = []  # List of all metrics
    comments = {}  # Dict of comments, used for reporting
    comments["option"] = "build model without resampling"

    inner_cv = StratifiedKFold(n_splits=k_inner_fold)
    pipeline_post_process = deepcopy(pipeline_post_process)
    # For efficiency we transform the data once, for all folds
    X_t = pipeline_post_process.fit_transform(X, y)
    y_t = y
    for k, (train_index, test_index) in enumerate(inner_cv.split(X_t, y_t)):
        if k not in skip_inner_folds:
            X_train, y_train, X_test, y_test = (
                X_t[train_index],
                y_t[train_index],
                X_t[test_index],
                y_t[test_index],
            )
            if undersampling_majority_class:
                fold_base_model, fold_metadata = train_ensemble_model_with_undersampling(
                    model, X_train, y_train, False, max_k_undersampling
                )
            else:
                fold_base_model, fold_metadata = train_model_without_undersampling(
                    model, X_train, y_train, False
                )
            list_metadata.append(fold_metadata)
            fold_final_model = fold_base_model
            if calibrated:
                fold_final_model = CalibratedClassifierCV(
                    fold_base_model, method="isotonic", cv="prefit"
                )
                fold_final_model.fit(X_test, y_test)
            fold_models.append(fold_final_model)
            y_proba = fold_final_model.predict_proba(X_test)[:, 1]
            fold_metrics.append(
                evaluate_metrics(y_test, y_proba, loss_metric, peeking_metrics)
            )
    averaged_metrics = average_metrics(fold_metrics)
    metadata = averaged_metadata_list(list_metadata)
    complete_model = MidasEnsembleClassifiersWithPipeline(
        pipeline_post_process, fold_models
    )
    comments["number of folds"] = len(fold_models)
    comments["average size of training set"] = metadata.get_num_init_samples_bf()
    comments["average prop of minority class"] = metadata.get_prop_minority_class_bf()
    return complete_model, averaged_metrics, comments


def find_best_model(list_models, list_metrics):
    list_loss_metrics = [metric["loss_metric"] for metric in list_metrics]
    index_best_model = list_loss_metrics.index(min(list_loss_metrics))
    best_model = list_models[index_best_model]
    return best_model, index_best_model


def train_inner_model(X, y, model_search_spaces,
                      X_hold_out, y_hold_out, k_inner_fold,
                      skip_inner_folds, n_initial_points, n_calls, ensemble,
                      calibrated, loss_metric, peeking_metrics,
                      skopt_func, verbose, report_doc):

    list_params = []
    list_models = []
    list_metrics = []
    list_holdout_metrics = []
    list_comments = []

    for key in model_search_spaces.keys():
        pipeline_post_process = model_search_spaces[key]["pipeline_post_process"]
        if not pipeline_post_process:
            pipeline_post_process = Pipeline([("identity", _MidasIdentity())])
        model_name = key
        model = model_search_spaces[key]["model"]
        complete_steps = pipeline_post_process.steps + [("model", model)]
        complete_pipeline = Pipeline(complete_steps)
        search_space = model_search_spaces[key]["search_space"]
        exists_resampler = exists_resampler(pipeline_post_process)

        @use_named_args(search_space)
        def func_to_minimize(**params):
            copy_params = params.copy()
            undersampling_majority_class = copy_params.pop(
                "undersampling_majority_class", False
            )
            max_k_undersampling = copy_params.pop(
                "max_k_undersampling", 0
            )

            complete_pipeline.set_params(**copy_params)
            list_params.append({**params, **{"model": model_name}})
            if verbose:
                print(f"Optimizing model {model_name}\n")
                print(f"With parameters {params}\n")

            if exists_resampler:
                ensemble_model, metrics, comments = ensemble_model_with_resampling(
                    X=X, y=y, pipeline_post_process=pipeline_post_process,
                    model=model, loss_metric=loss_metric, peeking_metrics=peeking_metrics,
                    k_inner_fold=k_inner_fold, skip_inner_folds=skip_inner_folds,
                    undersampling_majority_class=undersampling_majority_class,
                    max_k_undersampling=max_k_undersampling, calibrated=calibrated
                )
            else:
                ensemble_model, metrics, comments = ensemble_model_without_resampling(
                    X=X, y=y, pipeline_post_process=pipeline_post_process,
                    model=model, loss_metric=loss_metric, peeking_metrics=peeking_metrics,
                    k_inner_fold=k_inner_fold, skip_inner_folds=skip_inner_folds,
                    undersampling_majority_class=undersampling_majority_class,
                    max_k_undersampling=max_k_undersampling, calibrated=calibrated
                )
            list_models.append(ensemble_model)
            list_metrics.append(metrics)
            list_comments.append(comments)
            if verbose:
                print(f"Metric is {metrics['loss_metric']}\n")
            if len(y_hold_out) > 0:
                y_hold_out_proba = ensemble_model.predict_proba(X_hold_out)[:, 1]
                list_holdout_metrics.append(
                    evaluate_metrics(
                        y_hold_out, y_hold_out_proba, loss_metric, peeking_metrics
                    )
                )
            return metrics["loss_metric"]

        # perform optimization
        skopt_func(
            func=func_to_minimize, dimensions=search_space,
            n_initial_points=n_initial_points,
            n_calls=n_calls,
        )

    best_model, index_best_model = find_best_model(list_models, list_metrics)
    undersampling = list_params[index_best_model].get('undersampling_majority_class', default=False)

    if not ensemble and undersampling:
        exists_resampler = exists_resampler(best_model.get_complete_pipeline_to_fit())
        max_k_undersampling = list_params[index_best_model].get('max_k_undersampling', default=0)
        best_model = train_ensemble_model_with_undersampling(best_model.get_complete_pipeline_to_fit(),
                                                X, y, exists_resampler, max_k_undersampling)
    if not ensemble and not undersampling:
        best_model = best_model.get_complete_pipeline_to_fit().fit(X, y)

    if verbose:
        print("Best model found")
    report_dfs = create_report_dfs(list_params, list_metrics, loss_metric)
    if report_doc:
        write_train_report(
            report_doc=report_doc, list_params=list_params, list_metrics=list_metrics,
            list_holdout_metrics=list_holdout_metrics, peeking_metrics=peeking_metrics,
            list_comments=list_comments
        )

    return best_model, list_params[index_best_model], list_comments[index_best_model], report_dfs
