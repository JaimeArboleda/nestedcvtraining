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

dict_pipelines_post_process = {
    "option_1": Pipeline(
        [("scale", StandardScaler()), ("reduce_dims", PCA(n_components=2))]
    ),
    "option_2": Pipeline(
        [
            ("scale", StandardScaler()),
            ("reduce_dims", SelectKBest(mutual_info_classif, k=2)),
        ]
    ),
    "option_3": Pipeline([("identity", MidasIdentity())]),
}


if __name__ == "__main__":

    df = pd.read_csv(
        'https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv', header=None
    )
    df[df.columns[-1]] = df[df.columns[-1]].replace(1, 0).replace(2, 1)
    values = df.values
    X, y = values[:, :-1], values[:, -1]

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
                Integer(2, 3, name="max_k_undersampling"), # The imbalanced ratio is 2,7.
                Categorical(["minority", "all"], name="resample__sampling_strategy"),
                Categorical(
                    ["option_1", "option_2", "option_3"], name="post_process__option"
                ),
                Integer(2, 8, name="model__max_depth"),
                Real(0.05, 0.31, prior="log-uniform", name="model__learning_rate"),
                Integer(1, 10, name="model__min_child_weight"),
                Real(0.8, 1, prior="log-uniform", name="model__subsample"),
                Real(0.13, 0.8, prior="log-uniform", name="model__colsample_bytree"),
                Real(0.1, 10, prior="log-uniform", name="model__scale_pos_weight"),
                Categorical(["binary:logistic"], name="model__objective"),
            ]
        }
    }

    best_model, document = find_best_binary_model(
        X=X,
        y=y,
        model_search_spaces=dict_models,
        verbose=True,
        k_inner_fold=10,
        k_outer_fold=10,
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
    document.save("report_dataset_second_example.docx")



