from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from io import BytesIO
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter

def plot_calibration_curve(y, y_proba):
    fig, ax = plt.subplots()
    prob_true_1, prob_pred_1 = calibration_curve(y, y_proba, n_bins=12)
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    ax.plot(prob_true_1, prob_pred_1, marker='.')
    ax.set(xlabel='Average predicted probability in each bin', ylabel='Ratio of positives')
    memfile = BytesIO()
    plt.savefig(memfile)
    return memfile

def plot_precision_recall_curve(y, y_proba):
    fig, ax = plt.subplots()
    precision, recall, _ = precision_recall_curve(y, y_proba)
    average_precision = average_precision_score(y, y_proba)
    ax.plot(precision, recall, label=f'AP = {average_precision:0.2f}')
    ax.set(xlabel='Precision', ylabel='Recall')
    ax.legend(loc="lower left")
    memfile = BytesIO()
    plt.savefig(memfile)
    return memfile

def plot_roc_curve(y, y_proba):
    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = roc_auc_score(y, y_proba)
    ax.plot(fpr, tpr, label=f'ROC = {roc_auc:0.2f}')
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.legend(loc="lower right")
    memfile = BytesIO()
    plt.savefig(memfile)
    return memfile

def plot_confusion_matrix(y, y_proba):
    y_pred = y_proba > 0.5
    fig, ax = plt.subplots()
    confm = confusion_matrix(y, y_pred)
    # Normalize
    confm = confm.astype('float') / confm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(confm, cmap='Oranges', annot=True)
    ax.set(xlabel='Predicted label', ylabel='True label')
    memfile = BytesIO()
    plt.savefig(memfile)
    return memfile

def plot_histogram(y, y_proba):
    fig, ax = plt.subplots()
    predictions = {
        'predicted label': y_proba,
        'True label': y
    }
    df = pd.DataFrame.from_dict(predictions)
    sns.kdeplot(data=df, x='predicted label', hue='True label', hue_order=sorted(list(Counter(y).keys())))
    #colors = ['cornflowerblue', 'gold']

    # for i, label in enumerate(sorted(list(Counter(y).keys()))):
    #     # Subset
    #     subset = df[df['true label'] == label]
    #
    #     # Draw the density plot
    #     sns.histplot(data=subset['predicted label'], kde=True, label=label, color=colors[i])
        # sns.distplot(subset['predicted label'], hist=False, kde=True,
        #              kde_kws={'linewidth': 2},
        #              label=label)
    ax.set_yticks([], [])
    ax.set_xlim(0, 1)
    #plt.legend(prop={'size': 5}, title='Predicted label')
    plt.title('Density of predicted labels')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    memfile = BytesIO()
    plt.savefig(memfile)
    return memfile


def plot_param_metric_relation(data, x, y, height):
    fig, ax = plt.subplots()

    if len(Counter(data[x])) > 8 and pd.api.types.is_numeric_dtype(data[x]):
        data[x + ' '] = pd.cut(data[x], bins=8)
        data[x + ' '] = data[x + ' '].apply(lambda w: np.round(w.mid, 3))

        sns.violinplot(data=data[[x + ' ', y]], x=x + ' ', y=y, height=height, ax=ax)

        data.drop(columns=x + ' ', inplace=True)
    else:
        ax = sns.violinplot(data=data, x=x, y=y, height=height)

    memfile = BytesIO()
    plt.savefig(memfile)
    return memfile
