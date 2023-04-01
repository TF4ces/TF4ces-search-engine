#!/usr/bin/env python3
# encoding: utf-8

"""
    Evaluation metrics for information retrival.
        - Recall@K
        - MeanRecall@K
        - Precision@K
        - MeanPrecision@K

    Author : TF4ces
"""

import numpy as np


def recall_K(gold, pred, k):
    if len(gold) == 0: return 0.0
    gold, pred = set(gold), set(pred[:k])
    return len(pred & gold) / len(gold)


def mean_recall_K(golds, preds, k):
    return np.average([
        recall_K(gold=gold, pred=pred, k=k)
        for gold, pred in zip(golds, preds)
    ])


def precision_K(gold, pred, k):
    if len(pred) == 0: return 0.0
    gold, pred = set(gold), set(pred[:k])
    return len(pred & gold) / len(pred)


def mean_precision_K(golds, preds, k):
    return np.average([
        precision_K(gold=gold, pred=pred, k=k)
        for gold, pred in zip(golds, preds)
    ])
