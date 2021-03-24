# nestedcvtraining

nestedcvtraining: Python package to search for best parameters, train, (optionally) calibrate and validate the error of classification models using a Nested Cross-Validation approach.

## Overview

nestedcvtraining is built on top of several libraries, mainly:
- scikit-learn.
- imbalanced-learn.
- scikit-optimize.
- python-docx.

Inside of a standard machine learning flow, this package helps the automation of the last stages of hyperparameter optimization, model validation and model training. One of the benefits of using it is that hyperparameter optimization is not restricted to hyperparameters of the model, but also of the post-process pipeline, as it will be shown. Another benefit is the easy integration with a probability calibration step. When calibrated, the predict_proba method can be directly interpreted as a probability.

In other words, the package facilitates: 
- Search for best parameters, including parameters of the transformation process (for example, whether to use PCA or not, the number of components in PCA, whether to scale or not the features...). 
- Perform a nested Cross Validation in which: 
  -  The outer loop evaluates the quality of the models with respect to the given metrics. 
  -  The inner loop builds a (optionally calibrated) ensemble model selecting best parameters using Bayesian Search (with Skopt implementation). It's an ensemble model built using the trained models for each inner fold. 
  -  After the train, you get a report of the process (with lots of information and plots) and a final ensemble model is built repeating the inner procedure in the whole dataset. 
- This way, each layer of the nested cross validation serves only for one purpose: the outer layer for validating the model (including the procedure for model selection) and the inner layer for selecting the model searching the best parameters with a Bayesian Search.

## How it works

<p align="center">
<img src="https://github.com/JaimeArboleda/nestedcvtraining/blob/master/images/ncv.png">
Nested Cross Validation Scheme
</p>

Image taken from [here](https://mlfromscratch.com/nested-cross-validation-python-code), a good source of Machine Learning explanations. 

Nested Cross Validation aims at two things: 
- The Cross Validation part reduces the variance of all metrics and estimators by averaging them among several folds. 
- The nested part ensures that the procedure for model selection is completely independent from the model quality assessment (they use separated datasets). 

This package performs nested cross validation in this way: 
- The outer loop is used only for estimating the error of the model built in the inner loop. 
- The inner loop is used only for training an ensemble model by selecting the best parameters and hyperparameters for it. 

How does the training in the inner loop takes place? 
- If k is the number of inner folds on the inner loop, then for each combination of parameters and hyperparameters performed by the bayesian search engine, k models will be trained, and an ensemble model using all of them will be served (preditions of this ensemble modelis the averaged prediction of all base models). 
- If, in addition, the calibrated option is selected, all base models of the ensemble will be calibrated (each one of them, by using the corresponding inner validation dataset). 

This package follows all these rules and recommended practices: 
- You should only use a Cross Validation step for one thing: either for model selection, either for estimating the error of the model. If you use it for both things, you are at risk of underestimating the error of the model. 
- If you have a post-processing step on your data pipeline that uses info of all rows (for example, PCA, normalization, feature selection based on variance or information gain with respect to the target), this step should be done inside of the cross validation, fitting with the train set and transforming the test/validation set accordingly in the same fashion. If this care is not taking, you are at risk of understimating the error of the model. This is specially important when you use the information of the target to select the features. Otherwise, if the target is not used and you have big datasets (much more rows tan columns) this effect can be very small. 
- If you use cross validation for model selection, once you have checked that the model selection procedure is good (i.e. it has low variance, the metric scores are well enough), then you should apply it the same way to the whole dataset. 

## Install nestedcvtraining

```bash
pip install nestedcvtraining
```

As usual, you need all libraries listed in requierements.txt

## Getting Started with nestedcvtraining

This package has only one main entry point with many options. It has also one utility that may come in handy. 

The main API is the following:

```
def find_best_binary_model(
        X,
        y,
        model_search_spaces,
        k_outer_fold=5,
        skip_outer_folds=[],
        k_inner_fold=5,
        skip_inner_folds=[],
        n_initial_points=5,
        n_calls=10,
        calibrated=True,
        loss_metric='average_precision',
        peeking_metrics=[],
        report_level=11,
        size_variance_validation=20,
        skopt_func=gp_minimize,
        verbose=False,
        build_final_model=True
    ):
    """Finds best binary calibrated classification model and optionally
    generate a report doing a nested cross validation. In the inner
    cross validation, doing a Bayesian Search, the best parameters are found.
    In the outer cross validation, the model is validated.
    Finally, the whole procedure is used for the full dataset to return
    the best possible model.


    Parameters
    ----------
    X : np.array
        Feature set.

    y : np.array
        Classification target to predict. For the moment only binary labels are allowed, and
        values are supposed to be {0, 1} or {-1, 1}

    model_search_spaces : Dict[str : List[List[skopt.Space]]
        Dict of models to try inside of the inner loops. For each model, there is
        the corresponding list of space objects to delimit where the parameters live,
        including the pipeline postprocess to make. It admits also an option to set
        undersampling_majority_class method. It admits two values, True or False. If True
        it builds an ensemble model in the inner loop by creating many balanced folds
        by using the minority class with a undersampling of the majority class. If using
        this option, it also admits an Int max_k_undersampling, in order to limit the number of
        splits made for this (because if the imbalance ratio is for example 1000:1, it will
        create 1000 splits, which can be too much).

    k_outer_fold : int, default=5
        Number of folds for the outer cross-validation.

    skip_outer_folds : list, default=None
        If set, list of folds to skip during the loop.

    k_inner_fold : int, default=5
        Number of folds for the inner cross-validation.

    skip_inner_folds : list, default=None
        If set, list of folds to skip during the loop.

    n_initial_points : int, default=5
        Number of initial points to use in Bayesian Optimization.

    n_calls : int, default=5
        Number of additional calls to use in Bayesian Optimization.

    calibrated : bool, default=False
        If True, all models are calibrated using CalibratedClassifierCV

    loss_metric : str, default='auc'
        Metric to use in order to find best parameters in Bayesian Search. Options:
        - roc_auc
        - average_precision
        - neg_brier_score
        - neg_log_loss
        - histogram_width

    peeking_metrics : List[str], default=[]
        If not empty, in the report there will be a comparison between the metric of
        evaluation on the inner fold and the list of metrics in peeking_metrics.

    report_levels : int, default=11
        If 00, no report is returned.
        If 01, plots are not included. All peeking-metrics are evaluated on the outer fold for each inner-fold model.
        If 10, plots are included. No evaluation of peeking-metrics on the outer fold for each inner-fold model.
        If 11, a full report (it can be more time consuming).

    size_variance_validation : int, default=20
        Number of samples to use to check variance of different models.

    skopt_func : callable, default=gp_minimize
        Minimization function of the skopt library to be used.

    verbose : bool, default=False
        If True, you can trace the progress in the terminal.

    build_final_model : bool, default=True
        If False, no final model is built (only the report doc is returned). It can be convenient
        during the experimental phase.

    Returns
    -------
    model : Model trained with the full dataset using the same procedure
    as in the inner cross validation.
    doc : Document python-docx if report_level > 0. Otherwise, None
    """
```

A concise explanation of the parameters is the following: 
- X and y are, as usual, the feature set and target. 
- model_search_space is a dict of dicts made to specify how many models will be trained (it can be one or several), and, for each model: 
  -  The pipeline of post-process transformers (it can be an sklearn pipeline or an imblearn pipeline if there are resamplers inside).
  -  The search space for the bayesian optimization algorithm. This search space can have parameters of the model and of each step of the pipeline, using a prefix as documented in sklearn docs. 
-  k_outer_fold is the number of folds in the outer cross-validation.
-  skip_outer_folds is a list (can be empty) of outer folds to skip. As it can be computationally expensive to use a lot of folds, but at the same time using more folds increase the size of the training datasets, this parameter can be handy when one wants to have bigger training datasets without wanting to train a model for each fold. 
-  k_inner_fold and skip_inner_folds have the same meaning, but for the inner loop.
-  n_initial_points and n_calls are parameters directly passed to the optimizer function (gp_minimize by default).
-  calibrated allows you to specify if you want to have calibrated models or not. 
-  loss_metric is the metric used for the optimization procedure. 
-  peeking_metrics is a list of metrics that will be analyzed (but not used for minization) in the process. Its only purpose is enhance the reporting. 
-  report_levels is used to set the size of the reporting, as specified. It can be reduced if computationally the process is too expensive. 
-  size_variance_validation is the number of instances that will be completely left out in order to make a prediction on them for all ensemble models. This will be added to the report doc and can help assess if the models have big variance on individual predictions or not. 
-  skopt_func is the optimization function. You can check skopt docs for alternatives to the default.
-  verbose = False just limit some prints.
-  Build_final_model, if False, does not train a final model using all data. 

The other things in the api are those classses:

```
class OptionedPostProcessTransformer(TransformerMixin):
    """This class converts a dictionary of different options of post-process
    (each option will be a fully defined pipeline) in a transformer that
    has only one parameter, the option.
    In this way, you can perform a bayesian optimization search including a postprocess
    pipeline that uses this set of categorical fixed options.

    Example of input dict:
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
    """

class MidasIdentity(Identity):
    """
    Convenience class for creating a Pipeline that does not perform any transformation.
    It can be handy when in combination with OptionedPostProcessTransformer.
    """
    def fit(self, X, y):
        return self

```
Both the classes and the function will be more clarified in the examples section.

## Limitations

- For the moment it only works in binary classification settings, but I plan to adapt it to multilabel classification.
- The model returned cannot be fitted, because it is an ensemble model built using the inner cross-validation procedure and fitting it should be done only by using this procedure.
- The OptionedPostProcessTransformer cannot have resamplers inside any of the options. 

## Examples

Example 1: 

Suppose you have several post-processing options and you want to find which one works best for your model. Then, you can use this class to build a transformer that, depending on the option, will make one or the other transformation, and the bayesian search procedure can find the optimum value of the option (and all the other hyperparameters of the model). 

You need to define a dict like that: 

´´´
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
´´´

(option_3 is just doing nothing, and you can set this option by using the MidasIdentity transformer in the api). 

And then, when defining the search space for the model, you can set this optioned pipeline as follows: 

```
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
```

And you can invoke the main function as follows: 

```
    best_model, document = find_best_binary_model(
        X=X,
        y=y,
        model_search_spaces=dict_models,
        verbose=True,
        k_inner_fold=10,
        k_outer_fold=10,
        skip_inner_folds=[0, 2, 4, 6, 8, 9],
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
        skopt_func=gbrt_minimize
    )
```

This means that: 
- Two models will be tried (for each model, all optimization procedure, using n_inicial_points+n_initial_calls will be made; each model is completely independent from the other).
- XGBClassifier() has a two step pipeline, first a transformation (that in itself can be scaling + PCA -option_1-, scaling + SelectKBest -option_2- or nothing -option_3-) and then a SMOTE() for increasing samples of the minority class.  
- XGBClassifier() has a search space that plays with some parameters of the model, some parameters of the post-process (the option and the resampler strategy) and also it uses some "obscure" parameters that are not part neither of the model nor of the pipeline. This two parameters specify whether and how to carry out an undersampling of majority class ensemble strategy as described [here](http://proceedings.mlr.press/v94/ksieniewicz18a/ksieniewicz18a.pdf). This is compatible with having a resampler, because after the undersampling of majority class, a resampler can help make the classes even more balanced. 
- RandomForestClassifier(), on the other hand, has not a pipeline of post-process transformations, so the search parameters are mainly of the model. 

## Story behind nestedcvtraining

When working on a classification project where we wanted to get a calibrated probability prediction, I found it wasn't very easy to do the following things at once: 

- Searching for best parameters of model (and of post-process pipeline if possible).
- Training a model.
- Calibrating the model.
- Assessing the quality of the model.

The reason is that the class CalibratedClassifierCV of Sklearn needs an independent dataset in order to calibrate the base model. So if you want to perform a nested cross validation loop to separate the optimization procedure from the quality assessment, it's not easy to implement. CalibratedClassifierCV has essentially two fit methods: 
- One with the "prefit" option, that trains a regressor to calibrate the probabilities (one has to take care that the data is independent of the data on where the base model was trained).
- Another one with the "cv" option, that performs a Cross-Validation to train several base models and regressors using the splits, and builds an ensemble of models. 

In a nested cross validation, the second approach will be appropriated for the inner loop. But then a problem arises: if you are trying to optimize some metric using a Bayesian Search, how can you measure this metric when you have a model that has been trained on the whole dataset? I tried to solve this by accessing the inner base models of the CalibratedClassifierCV and evaluating the metric on the outer fold of the inner cross-validation. But this approach was not very elegant and finally I tried to "assemble" my own CalibratedClassifierCV by generating the inner split myself, training several base models (one for each fold), calibrating them using the "prefit" option and evaluating the loss metric of the Bayesian Search by averaging all metrics of all calibrated models on their corresponding validation dataset. 

After I did that, and found that it worked, I added more functionality to make it a more complete solution: 
- Generating a docx report of training, with many metrics and plots.
- Adding a simple way of optimizing post-processing steps in the same nested cross validation.
- Adding an option of using undersampling using this [idea](http://proceedings.mlr.press/v94/ksieniewicz18a/ksieniewicz18a.pdf), that is implemented [here](https://github.com/w4k2/umce/blob/master/method.py).
- And some other features... 

## Author

[Jaime Arboleda @JaimeArboleda](https://github.com/JaimeArboleda)

- <[Linkedin](https://www.linkedin.com/in/jaime-arboleda-castilla-1a676b207/)>

## Contributors are welcome!

Pull requests and/or suggestions are more than welcome!

## License

[MIT](https://choosealicense.com/licenses/mit/)


