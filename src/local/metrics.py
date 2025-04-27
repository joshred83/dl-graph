# -*- coding: utf-8 -*-
"""
Metrics used to evaluate the outlier detection performance
"""
# Author: Yingtong Dou <ytongdou@gmail.com>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
)
import torch

def eval_roc_auc(label, score):
    """
    ROC-AUC score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    roc_auc : float
        Average ROC-AUC score across different labels.
    """

    roc_auc = roc_auc_score(y_true=label, y_score=score)
    return roc_auc


def eval_recall_at_k(label, score, k=None):
    """
    Recall score for top k instances with the highest outlier scores.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.
    k : int, optional
        The number of instances to evaluate. ``None`` for
        recall. Default: ``None``.

    Returns
    -------
    recall_at_k : float
        Recall for top k instances with the highest outlier scores.
    """

    if k is None:
        k = sum(label)
    recall_at_k = sum(label[score.topk(k).indices]) / sum(label)
    return recall_at_k


def eval_precision_at_k(label, score, k=None):
    """
    Precision score for top k instances with the highest outlier scores.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.
    k : int, optional
        The number of instances to evaluate. ``None`` for
        precision. Default: ``None``.

    Returns
    -------
    precision_at_k : float
        Precision for top k instances with the highest outlier scores.
    """

    if k is None:
        k = sum(label)
    precision_at_k = sum(label[score.topk(k).indices]) / k
    return precision_at_k


def eval_average_precision(label, score):
    """
    Average precision score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    ap : float
        Average precision score.
    """

    ap = average_precision_score(y_true=label, y_score=score)
    return ap


def eval_f1(label, pred):
    """
    F1 score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : torch.Tensor
        Outlier prediction in shape of ``(N, )``.

    Returns
    -------
    f1 : float
        F1 score.
    """

    f1 = f1_score(y_true=label, y_pred=pred)
    return f1
def eval_classification_report(
    label,
    pred,
    target_names=("normal", "outlier"),
    output_dict=True,
):
    """
    sklearn ``classification_report`` with minimal wrapping.

    Parameters
    ----------
    label : torch.Tensor or np.ndarray
        Ground-truth binary labels (0 = normal, 1 = outlier).
    pred : torch.Tensor or np.ndarray
        Binary predictions (same shape as ``label``).
    target_names : tuple of str, optional
        Class names in report order; default ``("normal", "outlier")``.
    output_dict : bool, optional
        If ``True`` (default) return the structured dict produced by
        scikit-learn; if ``False`` return the plain text report.

    Returns
    -------
    report : dict | str
        Full classification report.
    """
    # Accept torch tensors transparently
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()

    return classification_report(
        y_true=label,
        y_pred=pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,          # avoid warnings when a class is absent
    )
def eval_precision(label, pred):
    """
    Precision for binary classification.
    """
    return precision_score(y_true=label, y_pred=pred, zero_division=0)


def eval_recall(label, pred):
    """
    Recall for binary classification.
    """
    return recall_score(y_true=label, y_pred=pred, zero_division=0)