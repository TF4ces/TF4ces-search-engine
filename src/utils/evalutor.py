#!/usr/bin/env python3
# encoding: utf-8

"""
    Evaluation metrics for Information Retrieval for unordered labelled dataset.
        - Recall@K
        - MeanRecall@K
        - Precision@K
        - MeanPrecision@K

    Author : TF4ces
"""

import numpy as np
import matplotlib.pyplot as plt


def recall_K(gold, pred, k):
    if len(gold) == 0: return 0.0   # Check divide by 0, CWE-369
    gold, pred = set(gold), set(pred[:k])
    return len(pred & gold) / len(gold)


def mean_recall_K(golds, preds, k):
    return np.average([
        recall_K(gold=gold, pred=pred, k=k)
        for gold, pred in zip(golds, preds)
    ])


def precision_K(gold, pred, k):
    if len(pred) == 0: return 0.0   # Check divide by 0, CWE-369
    gold, pred = set(gold), set(pred[:k])
    return len(pred & gold) / len(pred)


def mean_precision_K(golds, preds, k):
    return np.average([
        precision_K(gold=gold, pred=pred, k=k)
        for gold, pred in zip(golds, preds)
    ])


def calc_mean_recall_n_precision(gold_doc_ids, pred_doc_ids, k=None, to_display='', bl_print=True):
    if bl_print: print(f"-------------- Evaluation --------------")
    recall_k = mean_recall_K(golds=gold_doc_ids, preds=pred_doc_ids, k=k)
    precision_k = mean_precision_K(golds=gold_doc_ids, preds=pred_doc_ids, k=k)
    if bl_print: print(f"{to_display} Recall@{k} : {recall_k:.3f}")
    if bl_print: print(f"{to_display} Precision@{k} : {precision_k:.3f}")
    if bl_print: print(f"----------------------------------------")
    return recall_k, precision_k


def plot_precision_and_recall_curve(gold_doc_ids, preds_doc_ids, max_k=50):
    rs, ps, ks = list(), list(), list()
    for k in range(1, max_k):
        r, p = calc_mean_recall_n_precision(
            gold_doc_ids=gold_doc_ids,
            pred_doc_ids=preds_doc_ids,
            k=k,
            bl_print=False
        )
        rs.append(r)
        ps.append(p)
        ks.append(k)

    fig, ax = plt.subplots(1)
    ax.plot(ks, rs, label="Mean Recall", color='C2', lw=3, alpha=1)
    ax.plot(ks, ps, label="Mean Precision", color='C3', lw=3, alpha=1)
    ax.set(xlabel="K", ylabel="Metric")
    ax.legend()
    plt.tight_layout()
