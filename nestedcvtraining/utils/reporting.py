import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import chain
import copy


def _extend_nested_dict(dict1, dict2):
    for k in set(chain.from_iterable([dict1.keys(), dict2.keys()])):
        dict1[k].extend(dict2[k])


@dataclass
class Report:
    """ Nested Cross Validation Report

    It contains in a single dataclass all relevant information concerning the training loop.

    In addition, it has several handy methods for comparing the performance on all folds,
    comparing the best params found on all folds, and iterating over all models (with their corresponding
    outer fold) for further custom checks.

    It contains one element (row when converted to a dataframe) for each trained model during the inner loop
    including the refitted model over the whole inner dataset. That is, if the number of outer folds is k_o and the
    number of inner folds is k_i, and s_o and s_i folds are skipped respectively, then the number of rows will be:
    (k_o - s_o) * (k_i - s_i + 1), where the + 1 is added for the refitting of the best model on the whole
    (X_outer_train, y_outer_train).

    The attributes section contains all available information for each model (row).

    Attributes:
        best (list(bool) ): list of boolean values where True corresponds to the models that
            have been refitted and trained over the whole (X_outer_train, y_outer_train).
        outer_kfold (list(int) ): list of integers that hold the correspondence between each
            model and the corresponding outer fold.
        params (list(dict) ): list of dictionaries holding the parameters of each model
        inner_validation_metrics (dict(list) ): dictionary containing a list for each metric
            as averaged over all inner validation folds.
        outer_test_metrics (dict(list) ): dictionary containing a list for each metric
            as computed over the outer folds (only available for the best model of each outer
            fold). This is the most important piece for checking performance evaluation.
        model (list(estimator) ): list of models (only available for the best model of each outer
            fold).
        outer_test_indexes (list(ndarray) ): list of indexes for corresponding outer fold for each
            model. This attribute, with the previous one, are output to make it easier to perform
            further checks.
    """
    best: list = field(default_factory=list)
    outer_kfold: list = field(default_factory=list)
    params: list = field(default_factory=list)
    inner_validation_metrics: dict = field(default_factory=lambda: defaultdict(list))
    outer_test_metrics: dict = field(default_factory=lambda: defaultdict(list))
    model: list = field(default_factory=list)
    outer_test_indexes: list = field(default_factory=list)

    def to_dataframe(self):
        """ It converts the information to a dataframe.
        Returns:
            df (dataframe): A dataframe with one row per trained model, with all attached information.
        """
        self_dict = copy.deepcopy(self.__dict__)
        inner_validation_metrics = self_dict.pop("inner_validation_metrics")
        outer_test_metrics = self_dict.pop("outer_test_metrics")
        all_params = self_dict.pop("params")
        all_params_keys = sorted(set(chain.from_iterable([[ik for ik in ks] for ks in all_params])))
        for param_key in all_params_keys:
            self_dict["param__" + param_key] = []
            for param in all_params:
                self_dict["param__" + param_key].append(param.get(param_key, None))
        for k in inner_validation_metrics:
            self_dict["inner_validation_metrics__" + k] = inner_validation_metrics[k]
        for k in outer_test_metrics:
            self_dict["outer_test_metrics__" + k] = outer_test_metrics[k]
        return pd.DataFrame.from_dict(self_dict)

    def _append(
            self, best, outer_kfold, params, inner_validation_metrics, outer_test_metrics, model, outer_test_indexes
    ):
        self.best.append(best)
        self.outer_kfold.append(outer_kfold)
        self.params.append(params)
        self.model.append(model)
        self.outer_test_indexes.append(outer_test_indexes)
        _extend_nested_dict(self.inner_validation_metrics, inner_validation_metrics)
        _extend_nested_dict(self.outer_test_metrics, outer_test_metrics)

    def _extend(self, other):
        self.best.extend(other.best)
        self.outer_kfold.extend(other.outer_kfold)
        self.params.extend(other.params)
        self.model.extend(other.model)
        self.outer_test_indexes.extend(other.outer_test_indexes)
        _extend_nested_dict(self.inner_validation_metrics, other.inner_validation_metrics)
        _extend_nested_dict(self.outer_test_metrics, other.outer_test_metrics)

    def _best_idx(self):
        for idx, best in enumerate(self.best):
            if best:
                yield idx

    def get_best_params(self, only_diff=False):
        """ It outputs the parameters of all best models, to check to what extent they differ.
        Args:
            only_diff (bool): If True, only the parameters where thhere is a difference among all models
                are output. Otherwise, all params are output.
        Returns:
            params (dict): A dictionary where the keys are the param names and the values are
                lists of corresponding values (one per best model).
        """
        best_idxs = [idx for idx in self._best_idx()]
        params = defaultdict(list)
        for idx in best_idxs:
            for k in self.params[idx]:
                params[k].append(self.params[idx][k])
        if only_diff:
            return {
                k: v
                for k, v in params.items()
                if len(set(v)) > 1
            }
        else:
            return dict(params)

    def _filter_only_best(self, alist):
        return [
            alist[idx]
            for idx in self._best_idx()
        ]

    def get_outer_metrics_report(self):
        """ It outputs a basic report of outer metrics. For each outer metric, it outputs
        the mean, sd and range.

        Returns:
            metrics_report (dict): A nested dictionary where outer keys are metric names and
                inner keys are `mean`, `sd`, `min`, `max`.
        """
        metric_report = {}
        for metric in self.outer_test_metrics:
            metric_report[metric] = {
                "mean": np.mean(self._filter_only_best(self.outer_test_metrics[metric])),
                "sd": np.std(self._filter_only_best(self.outer_test_metrics[metric])),
                "min": min(self._filter_only_best(self.outer_test_metrics[metric])),
                "max": max(self._filter_only_best(self.outer_test_metrics[metric])),
            }
        return metric_report

    def iter_models_test_idxs(self):
        """ Iterator that yields all refitted models with their corresponding
        outer test indexes, so that it's easy to compute other metrics or make
        some custom plots to check the performance.

        Yields:
            t (tuple): a tuple containing model and test_idxs
        """
        for t in zip(self._filter_only_best(self.model), self._filter_only_best(self.outer_test_indexes)):
            yield t
