# nestedcvtraining

nestedcvtraining: Python package to search for best parameters, train and calibrate binary classification models using a Nested Cross-Validation approach.

## Overview

nestedcvtraining is built on top of several libraries, mainly:
- Sciki-Learn.
- Imblearn.
- Skopt.
- Python-Docx

Inside a standard machine learning flow, this package helps the automation of the last stages of hyperparameter optimization, model validation and model training. One of the benefits of using it is that hyperparameter optimization is not restricted to hyperparameters of the model, but also of the post-process pipeline, as it will be shown. 

The model is optionally calibrated, so that the predict_proba method can be directly interpreted as a probability.

In other words, the package facilitates: 
- Search for best parameters, including parameters of the transformation process (for example, whether to use PCA or not, the number of components in PCA, whether to scale or not the features...). 
- Perform a nested Cross Validation in which: 
  -  The outer loop evaluates the quality of the models with respect to the given metrics. 
  -  The inner loop builds a (optionally calibrated) model selecting best parameters using Bayesian Search (with Skopt implementation). 
  -  After the train, you get a report of the process (with lots of information and plots) and a final model is built repeating the inner procedure in the whole dataset. 
- This way, each layer of the nested cross validation serves only for one purpose: the outer layer for validating the model (including the procedure for model selection) and the inner layer for selecting the model searching the best parameters with a Bayesian Search.

## How it works

<p align="center">
<img src="https://mlfromscratch.com/content/images/size/w2000/2019/12/ncv.png">
Nested Cross Validation Scheme
</p>
Image taken from [here](https://mlfromscratch.com/nested-cross-validation-python-code), s good source of Machine Learning explanations. 

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


