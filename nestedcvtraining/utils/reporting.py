import pandas as pd
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from itertools import chain
import copy


@dataclass
class LoopInfo:
    name: list = field(default_factory=list)
    best: list = field(default_factory=list)
    outer_kfold: list = field(default_factory=list)
    params: list = field(default_factory=list)
    inner_test_metrics: dict = field(default_factory=lambda: defaultdict(list))
    outer_test_metrics: dict = field(default_factory=lambda: defaultdict(list))

    def to_dataframe(self):
        self_dict = copy.deepcopy(self.__dict__)
        inner_test_metrics = self_dict.pop("inner_test_metrics")
        outer_test_metrics = self_dict.pop("outer_test_metrics")
        all_params = self_dict.pop("params")
        all_params_keys = sorted(set(chain.from_iterable([[ik for ik in ks] for ks in all_params])))
        for param_key in all_params_keys:
            self_dict["param__" + param_key] = []
            for param in all_params:
                self_dict["param__" + param_key].append(param.get(param_key, None))
        for k in inner_test_metrics:
            self_dict["inner_test_metrics__" + k] = inner_test_metrics[k]
        for k in outer_test_metrics:
            self_dict["outer_test_metrics__" + k] = outer_test_metrics[k]
        return pd.DataFrame.from_dict(self_dict)

    def extend(self, other):
        def extend_nested_dict(dict1, dict2):
            for k in set(chain.from_iterable([dict1.keys(), dict2.keys()])):
                dict1[k].extend(dict2[k])

        self.name.extend(other.name)
        self.best.extend(other.best)
        self.outer_kfold.extend(other.outer_kfold)
        self.params.extend(other.params)
        extend_nested_dict(self.inner_test_metrics, other.inner_test_metrics)
        extend_nested_dict(self.outer_test_metrics, other.outer_test_metrics)


def dataframe_from_loop_infos(*loop_infos):
    dfs = []
    for loop_info in loop_infos:
        dfs.append(loop_info.to_dataframe())
    df = pd.concat(dfs).reset_index(drop=True)
    df.columns = ["name", "best", "outer_kfold",
                  *sorted([c for c in df.columns if "param__" in c]),
                  *sorted([c for c in df.columns if "inner_test_" in c]),
                  *sorted([c for c in df.columns if "outer_test_" in c]),
                  ]
    return df

