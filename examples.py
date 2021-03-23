import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from skopt.space import Real, Integer, Categorical
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
import lightgbm as lgb
from nestedcvtraining.utils.pipes_and_transformers import MidasIdentity, OptionedPostProcessTransformer
from nestedcvtraining.api import find_best_binary_model

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
    "option_3": Pipeline([("identity", MidasIdentity())]),
}


dict_models_example = {
    "gradient_boosting": {
        "model": GradientBoostingClassifier(),
        "pipeline_post_process": Pipeline([("identity", MidasIdentity())]),
        "search_space": [
            Integer(4, 12, name="model__max_depth"),
            Integer(10, 500, name="model__n_estimators"),
            Real(0.001, 0.15, prior="log-uniform", name="model__learning_rate"),
            Real(0.005, 0.10, prior="log-uniform", name="model__min_samples_split"),
            Real(0.005, 0.10, prior="log-uniform", name="model__min_samples_leaf"),
            Real(0.8, 1, prior="log-uniform", name="model__subsample"),
        ],
    },
    "random_forest": {
        "model": RandomForestClassifier(),
        "pipeline_post_process": Pipeline(
            [("scale", StandardScaler()), ("reduce_dims", PCA(n_components=50))]
        ),
        "search_space": [
            Integer(30, 100, name="reduce_dims__n_components"),
            Integer(0, 1, name="model__bootstrap"),
            Integer(10, 1000, name="model__n_estimators"),
            Integer(5, 15, name="model__max_depth"),
            Integer(5, 50, name="model__min_samples_split"),
            Integer(1, 4, name="model__min_samples_leaf"),
            Categorical(["auto", "sqrt"], name="model__max_features"),
            Categorical(["balanced", "balanced_subsample"], name="model__class_weight"),
        ],
    },
    "xgboost": {
        "model": XGBClassifier(),
        "pipeline_post_process": Pipeline(
            [
                (
                    "post_process",
                    OptionedPostProcessTransformer(dict_pipelines_post_process),
                )
            ]
        ),
        "search_space": [
            Categorical(
                ["option_1", "option_2", "option_3"], name="post_process__option"
            ),
            Integer(5, 15, name="model__max_depth"),
            Real(0.05, 0.31, prior="log-uniform", name="model__learning_rate"),
            Integer(1, 10, name="model__min_child_weight"),
            Real(0.8, 1, prior="log-uniform", name="model__subsample"),
            Real(0.13, 0.8, prior="log-uniform", name="model__colsample_bytree"),
            Real(0.1, 10, prior="log-uniform", name="model__scale_pos_weight"),
            Categorical(["binary:logistic"], name="model__objective"),
        ],
    },
    "lightgbm": {
        "model": lgb.LGBMClassifier(),
        "pipeline_post_process": Pipeline([("identity", MidasIdentity())]),
        "search_space": [
            Categorical([True, False], name="undersampling_majority_class"),
            Real(0.01, 0.5, prior="log-uniform", name="model__learning_rate"),
            Integer(1, 30, name="model__max_depth"),
            Integer(10, 400, name="model__num_leaves"),
            Real(0.1, 1.0, prior="uniform", name="model__feature_fraction"),
            Real(0.1, 1.0, prior="uniform", name="model__subsample"),
            Categorical(["balanced"], name="model__class_weight"),
            Categorical(["binary"], name="model__objective"),
        ],
    },
}
if __name__ == "__main__":
    dataset_prueba = pd.read_csv(
        "https://raw.githubusercontent.com/shraddha-an/cleansed-datasets/master/credit_approval.csv"
    )
    y = dataset_prueba["Target"].to_numpy()
    X = dataset_prueba[[c for c in dataset_prueba.columns if c != "Target"]].to_numpy()
    dict_models = {
        "xgboost": {
            "model": XGBClassifier(),
            "pipeline_post_process": Pipeline(
                [
                    (
                        "post_process",
                        OptionedPostProcessTransformer(dict_pipelines_post_process),
                    ),
                    ("resample", SMOTE()),
                ]
            ),
            "search_space": [
                Categorical([True, False], name="undersampling_majority_class"),
                Categorical(["minority", "all"], name="resample__sampling_strategy"),
                Categorical(
                    ["option_1", "option_2", "option_3"], name="post_process__option"
                ),
                Integer(5, 15, name="model__max_depth"),
                Real(0.05, 0.31, prior="log-uniform", name="model__learning_rate"),
                Integer(1, 10, name="model__min_child_weight"),
                Real(0.8, 1, prior="log-uniform", name="model__subsample"),
                Real(0.13, 0.8, prior="log-uniform", name="model__colsample_bytree"),
                Real(0.1, 10, prior="log-uniform", name="model__scale_pos_weight"),
                Categorical(["binary:logistic"], name="model__objective"),
            ],
        },
        "random_forest": {
            "model": RandomForestClassifier(),
            "pipeline_post_process": None,
            "search_space": [
                Categorical([True, False], name="undersampling_majority_class"),
                Integer(0, 1, name="model__bootstrap"),
                Integer(10, 100, name="model__n_estimators"),
                Integer(2, 10, name="model__max_depth"),
                Integer(5, 20, name="model__min_samples_split"),
                Integer(1, 4, name="model__min_samples_leaf"),
                Categorical(["auto", "sqrt"], name="model__max_features"),
                Categorical(
                    ["balanced", "balanced_subsample"], name="model__class_weight"
                ),
            ],
        },
    }

    best_model, document = find_best_binary_model(
        X=X,
        y=y,
        model_search_spaces=dict_models,
        verbose=True,
        k_inner_fold=10,
        k_outer_fold=10,
        skip_inner_folds=[0, 2, 4, 6, 8],
        skip_outer_folds=[0, 2, 4, 6, 8],
        n_initial_points=10,
        n_calls=10,
        loss_metric="average_precision",
        peeking_metrics=[
            "roc_auc",
            "neg_log_loss",
            "average_precision",
            "neg_brier_score",
        ],
    )
    document.save("report_dataset_prueba.docx")
