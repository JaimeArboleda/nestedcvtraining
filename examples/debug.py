import numpy as np, pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
sys.path.append('..')
from nestedcvtraining.under_sampling_classifier import UnderSamplingClassifier
from nestedcvtraining.switch_case import SwitchCaseTransformer, SwitchCaseResampler
from nestedcvtraining.api import find_best_model
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer, fbeta_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from skopt.space import Real, Integer, Categorical
from skopt import gbrt_minimize

if __name__ == "__main__":

    dataset_prueba = pd.read_csv(
        "https://raw.githubusercontent.com/JaimeArboleda/nestedcvtraining/master/datasets/new-thyroid.csv", header=None
    )
    values = dataset_prueba.values
    X, y = values[:, :-1], (values[:, -1] - 1).astype(int)

    resampler = SwitchCaseResampler(
        cases=[
            (
                "resampler_1",
                SMOTE(k_neighbors=3)
            ),
            (
                "resampler_2",
                "passthrough"
            )
        ],
        switch="resampler_1"
    )

    preprocessor = SwitchCaseTransformer(
        cases=[
            (
                "prep_1",
                Pipeline([
                    ("scale", StandardScaler()),
                    ("reduce_dims", PCA(n_components=5))
                ])
            ),
            (
                "prep_2",
                Pipeline([
                    ("scale", StandardScaler()),
                    ("reduce_dims", SelectKBest(mutual_info_classif, k=5)),
                ])
            ),
            (
                "prep_3",
                "passthrough"
            )
        ],
        switch="prep_1"
    )

    model = SwitchCaseTransformer(
        cases=[
            (
                "model_1",
                LogisticRegression()
            ),
            (
                "model_2",
                RandomForestClassifier()
            )
        ],
        switch="model_1"
    )

    clf = Pipeline(
        [("resampler", resampler), ("preprocessor", preprocessor), ("model", model)]
    )

    search_space= [
        Categorical(["resampler_1", "resampler_2"], name="resampler__switch"),
        Categorical(["prep_1", "prep_2", "prep_3"], name="preprocessor__switch"),
        Categorical(["model_1", "model_2"], name="model__switch"),
        Categorical(["minority", "all"], name="resampler__resampler_1__sampling_strategy"),
        Integer(5, 15, name="model__model_2__max_depth")
    ]

    best_model, report = find_best_model(
        X=X,
        y=y,
        model=clf,
        search_space=search_space,
        verbose=True,
        k_inner=6,
        k_outer=6,
        skip_inner_folds=[3],
        skip_outer_folds=[3],
        n_initial_points=10,
        n_calls=10,
        calibrate="only_best",
        calibrate_params={"method": "isotonic"},
        optimizing_metric=make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True),
        other_metrics={"acc": "accuracy"},
        skopt_func=gbrt_minimize
    )