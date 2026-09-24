"""
Animator and visualization utilities for KEPPNet training.
Provides confusion matrix, ROC, PRC, and metrics bar chart plotting.
"""

from os.path import join, exists
from os import makedirs
from copy import deepcopy

import numpy as np
import itertools
import matplotlib
matplotlib.use("Agg")  # headless-safe; must be before pyplot import
from matplotlib import pyplot as plt
try:
    from matplotlib_inline import backend_inline
except ImportError:
    backend_inline = None
from sklearn import metrics

# Headless / debug flags — safe defaults for open-source release
debug = False
local = False


def use_svg_display():
    if backend_inline is not None:
        backend_inline.set_matplotlib_formats('svg')


def set_axes(axes, x_label, y_label, x_lim, y_lim, x_scale, y_scale, legend):
    if x_label is not None:
        axes.set_xlabel(x_label)
    if y_label is not None:
        axes.set_ylabel(y_label)
    if x_scale is not None:
        axes.set_xscale(x_scale)
    if y_scale is not None:
        axes.set_yscale(y_scale)
    axes.set_xlim(x_lim)
    axes.set_ylim(y_lim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Animator:
    """For plotting training curves and evaluation figures."""

    def __init__(self, x_label=None, y_label=None, legend=None, x_lim=None,
                 y_lim=None, x_scale='linear', y_scale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), num_rows=1, num_cols=1,
                 fig_size=(5., 3.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
        if num_rows * num_cols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: set_axes(
            self.axes[0], x_label, y_label, x_lim, y_lim, x_scale, y_scale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        # In headless mode (debug=False), skip live display
        if debug and local:
            self.axes[0].cla()
            for x, y, fmt in zip(self.X, self.Y, self.fmts):
                self.axes[0].plot(x, y, fmt)
            self.config_axes()
            try:
                from IPython import display
                display.display(self.fig)
                display.clear_output(wait=True)
            except Exception:
                pass
            plt.pause(0.01)

    def show(self):
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=False,
                              title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.gcf().subplots_adjust(bottom=0.25)

    @staticmethod
    def get_confusion_matrix(cnf_matrix, saving_dir=None, model_name=''):
        plt.rcParams.update({'font.size': 19})
        plt.figure()
        Animator.plot_confusion_matrix(cnf_matrix, classes=[0, 1],
                                       title='Confusion matrix, without normalization')
        if saving_dir is not None:
            if not exists(saving_dir):
                makedirs(saving_dir)
            plt.savefig(join(saving_dir, f'{model_name}_confusion'))
        try:
            plt.show()
        except Exception:
            pass
        plt.close('all')

        plt.figure()
        Animator.plot_confusion_matrix(cnf_matrix, normalize=True, classes=[0, 1],
                                       title='Normalized confusion matrix')
        if saving_dir is not None:
            plt.savefig(join(saving_dir, f'{model_name}_confusion_normalized'))
        try:
            plt.show()
        except Exception:
            pass
        plt.close('all')

    @staticmethod
    def get_metrics(effect, saving_dir=None, model_name=''):
        effect = deepcopy(effect)
        effect.pop('model_name', None)
        plt.figure()
        if isinstance(effect, dict):
            ax = list(effect.keys())
            ay = list(effect.values())
            plt.ylim([0.0, 1.05])
            plt.tick_params(axis='x', labelsize=12)
            plt.bar(ax, ay)
            for a, b, i in zip(ax, ay, range(len(ax))):
                plt.text(a, b + 0.01, "%.2f" % ay[i], ha='center', fontsize=12)
            if saving_dir is not None:
                if not exists(saving_dir):
                    makedirs(saving_dir)
                plt.savefig(join(saving_dir, f'{model_name}_metrics'))
        try:
            plt.show()
        except Exception:
            pass
        plt.close('all')

    @staticmethod
    def get_auc(y_prob, y_true, saving_dir=None, model_name=''):
        fig = plt.figure()
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=12)
        plt.legend(loc="lower right")
        if saving_dir is not None:
            if not exists(saving_dir):
                makedirs(saving_dir)
            fig.savefig(join(saving_dir, f'{model_name}_auc_curves'))
        try:
            plt.show()
        except Exception:
            pass
        plt.close('all')

    @staticmethod
    def get_auprc(y_prob, y_true, saving_dir=None, model_name=''):
        fig = plt.figure()
        fig.set_size_inches((10, 6))
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob, pos_label=1)
        ap = metrics.average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=12)
        plt.legend(loc="lower right")
        if saving_dir is not None:
            if not exists(saving_dir):
                makedirs(saving_dir)
            fig.savefig(join(saving_dir, f'{model_name}_auprc_curves'))
        try:
            plt.show()
        except Exception:
            pass
        plt.close('all')
