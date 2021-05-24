import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

sns.set(font_scale=1.5)
sns.set_color_codes("muted")
font = {'size': 15}
plt.rc('font', **font)
plt.figure(figsize=(10, 8))


class RMSE:
    decreasing = True
    name = 'rmse'
    color = 'red'

    def __init__(self, y_pred, y_true, verbose=False):
        self.y_pred = y_pred[~np.isnan(y_pred)].flatten()
        self.y_true = y_true[~np.isnan(y_true)].flatten()
        self.verbose = verbose

    def eval(self):
        return metrics.mean_squared_error(self.y_true, self.y_pred, squared=False)


class MAE:
    decreasing = True
    name = 'mae'
    color = 'blue'

    def __init__(self, y_pred, y_true, verbose=False):
        self.y_pred = y_pred[~np.isnan(y_pred)].flatten()
        self.y_true = y_true[~np.isnan(y_true)].flatten()
        self.verbose = verbose

    def eval(self):
        return metrics.mean_absolute_error(self.y_true, self.y_pred)


class F1Score:
    decreasing = False
    name = 'f1'
    color = 'red'

    def __init__(self, y_pred, y_true, verbose=False):
        self.y_pred = y_pred[~np.isnan(y_pred)].flatten()
        self.y_true = y_true[~np.isnan(y_true)].flatten()
        self.y_pred[self.y_pred < 0.5] = 0
        self.y_pred[self.y_pred >= 0.5] = 1
        self.verbose = verbose

    def eval(self):
        return metrics.f1_score(self.y_true, self.y_pred)


class ROCScore:
    decreasing = False
    name = 'auc-roc'
    color = 'blue'

    def __init__(self, y_pred, y_true, verbose=False):
        self.y_pred = y_pred[~np.isnan(y_pred)].flatten()
        self.y_true = y_true[~np.isnan(y_true)].flatten()
        self.verbose = verbose

    def eval(self):
        if self.verbose:
            self.plot_roc_curve()

        return metrics.roc_auc_score(self.y_true, self.y_pred)

    def plot_roc_curve(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.y_true, self.y_pred)
        plt.plot(fpr, tpr, label="ROC curve")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC curve")

        thresholds = list(reversed(thresholds[:-1]))
        step = (len(thresholds)) // 5
        for i in range(1, len(thresholds), step - 1):
            plt.scatter(fpr[i], tpr[i], s=20, c=['m'])
            threshold = str(thresholds[i])[:4]
            plt.annotate(threshold, (fpr[i], tpr[i]), (fpr[i] + .03, tpr[i] + .03))

        plt.savefig(os.path.join('data', 'images', 'roc_curve.png'))
        plt.clf()


class PRScore:
    decreasing = False
    name = 'auc-pr'
    color = 'green'

    def __init__(self, y_pred, y_true, verbose=False):
        self.y_pred = y_pred[~np.isnan(y_pred)].flatten()
        self.y_true = y_true[~np.isnan(y_true)].flatten()
        self.verbose = verbose
        self.precision, self.recall, self.thresholds = metrics.precision_recall_curve(self.y_true, self.y_pred)

    def eval(self):
        if self.verbose:
            self.plot_pr_curve()

        return metrics.auc(self.recall, self.precision)

    def plot_pr_curve(self):
        plt.plot(self.recall, self.precision, 'r', label="PR curve")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("PR curve")

        self.thresholds = list(reversed(self.thresholds[:-1]))
        step = len(self.thresholds) // 5
        for i in range(1, len(self.thresholds), step - 1):
            plt.scatter(self.recall[i], self.precision[i], s=20, c=['m'])
            threshold = str(self.thresholds[i])[:4]
            plt.annotate(threshold, (self.recall[i], self.precision[i]), (self.recall[i] + .03, self.precision[i] + .03))

        plt.savefig(os.path.join('data', 'images', 'pr_curve.png'))
        plt.clf()


class NDCGScore:
    decreasing = False
    name = 'ndcg'
    color = 'green'

    def __init__(self, y_pred, y_true, verbose=False):
        self.y_pred = np.nan_to_num(y_pred)
        self.y_true = np.nan_to_num(y_true)
        self.verbose = verbose

    def eval(self):
        return metrics.ndcg_score(self.y_true, self.y_pred)
