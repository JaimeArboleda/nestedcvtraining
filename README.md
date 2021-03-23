# nestedcvtraining

nestedcvtraining: Python package to search for best parameters, train and calibrate binary classification models using a Nested Cross-Validation approach.

## Overview

nestedcvtraining is built on top of several libraries, mainly:
- Sciki-Learn.
- Imblearn.
- Skopt.
- Python-Docx

Inside of a standard macihne learning flow, this package helps the automation of the last stages of hyperparameter optimization, model validation and model training. One of the benefits of using it is that hyperparameter optimization is not restricted to hyperparameters of the model, but also of the post-process pipeline, as it will be shown. 

The model is optionally calibrated, so that the predict_proba method can be directly interpreted as a probability.

In other words, the package facilitates: 
- Search for best parameters, including parameters of the transformation process (for example, whether to use PCA or not, the number of components in PCA, whether to scale or not the features...). 
- Perform a nested Cross Validation in which: 
  -  The outer loop evaluates the quality of the models with respect to the given metrics. 
  -  The inner loop builds a (optionally calibrated) model selecting best parameters using Bayesian Search (with Skopt implementation). 
  -  After the train, you get a report of the process (with lots of information and plots) and a final model is built repeating the inner procedure in the whole dataset. 
- This way, each layer of the nested cross validation serves only for one purpose: the outer layer for validating the model (including the procedure for model selection) and the inner layer for selecting the model searching the best parameters with a Bayesian Search.

## Install nestedcvtraining

```bash
pip install nestedcvtraining
```

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
        Classification target to predict.

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



## Limitations

For the moment it only works in binary classification settings, but I plan to adapt it to multilabel classification. 



## Examples

This pack

## Story behind nestedcvtraining

When I was working on a Deep Learning project, it was very time-consuming to develop the pipeline for experimentation.
I wanted 2 features.

First one was an option to resume the pipeline using the intermediate data files instead of running the whole pipeline.
This was important for rapid Machine/Deep Learning experimentation.

Second one was modularity, which means keeping the 3 components, task processing, file/database access, and DAG definition, independent.
This was important for efficient software engineering.

After this project, I explored for a long-term solution.
I researched about 3 Python packages for pipeline development, Airflow, Luigi, and Kedro, but none of these could be a solution.

Luigi provided resuming feature, but did not offer modularity.
Kedro offered modularity, but did not provide resuming feature.

After this research, I decided to develop my own package that works on top of Kedro.
Besides, I added syntactic sugars including Sequential API similar to Keras and PyTorch to define DAG.
Furthermore, I added integration with MLflow, PyTorch, Ignite, pandas, OpenCV, etc. while working on more Machine/Deep Learning projects.

After I confirmed my package worked well with the Kaggle competition, I released it as PipelineX.

## Author

[Jaime Arboleda @JaimeArboleda](https://github.com/JaimeArboleda)

- <[Linkedin](https://www.linkedin.com/in/jaime-arboleda-castilla-1a676b207/)>

## Contributors are welcome!

Pull requests are more than welcome!

## License

[MIT](https://choosealicense.com/licenses/mit/)


