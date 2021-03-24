import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from skopt.space import Real, Integer, Categorical
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from nestedcvtraining.api import find_best_binary_model, MidasIdentity, OptionedPostProcessTransformer
from skopt import gbrt_minimize

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


if __name__ == "__main__":
    dataset_prueba = pd.read_csv(
        "https://raw.githubusercontent.com/JaimeArboleda/nestedcvtraining/master/datasets/credit_approval.csv"
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
                Integer(5, 6, name="max_k_undersampling"),
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

    best_model, report_doc, report_dfs = find_best_binary_model(
        X=X,
        y=y,
        model_search_spaces=dict_models,
        verbose=True,
        k_inner_fold=10,
        k_outer_fold=9,
        skip_inner_folds=[0, 2, 4, 6, 8, 9],
        skip_outer_folds=[0, 2, 3, 4, 6, 8],
        n_initial_points=10,
        n_calls=10,
        build_final_model=False,
        loss_metric="average_precision",
        peeking_metrics=[
            "roc_auc",
            "neg_log_loss",
            "average_precision",
            "neg_brier_score",
        ],
        skopt_func=gbrt_minimize
    )
    report_doc.save("report_dataset.docx")
    for model in report_dfs.keys():
        report_dfs[model].to_csv('report_dfs ' + model + '.csv', sep=';', index=False)


