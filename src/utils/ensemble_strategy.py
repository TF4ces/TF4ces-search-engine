#!/usr/bin/env python3
# encoding: utf-8

"""
    Ensemble Strategies.
        1. AOI : All Ordered Intersection
        2. TRO : Top Rank Order
        3. TXIOI : Top X interleaved with Ordered Intersections

    Author : TF4ces
"""

# Native imports

# Third-party imports

# User imports


def all_ordered_intersection(preds_1, preds_2):
    """
    AOI
    Finds the intersection of two predictions, aka. common predictions.
    Args:
        preds_1: predictions from model 1
        preds_2: predictions from model 2

    Returns:
        ensemble of preds, from preds_1 and preds_2
    """
    return [[doc_id for doc_id in doc_ids_1 if doc_id in doc_ids_2] for doc_ids_1, doc_ids_2 in zip(preds_1, preds_2)]


def top_rank_order(preds_1, preds_2):
    """
    TRO
    ordered interleaving of ranked documents from preds_1 and preds_2.

    Args:
        preds_1: predictions from model 1
        preds_2: predictions from model 2

    Returns:
        ensemble of preds, from preds_1 and preds_2
    """
    n = len(preds_1[0])

    ensemble_preds = []
    for doc_ids_1, doc_ids_2 in zip(preds_1, preds_2):
        ensemble_pred = []
        for doc_id_1, doc_id_2 in zip(doc_ids_1, doc_ids_2):
            if doc_id_1 not in ensemble_pred:
                ensemble_pred.append(doc_id_1)

            if doc_id_2 not in ensemble_pred:
                ensemble_pred.append(doc_id_2)

            if len(ensemble_pred) >= n:
                break
        ensemble_preds.append(ensemble_pred)
    return ensemble_preds


def top_x_interleaved_with_ordered_intersection(preds_1, preds_2, x=1):
    """
    TXIOI
    combines the best of TRO with only top X interleaving with the rest of AOI.

    Args:
        preds_1: predictions from model 1
        preds_2: predictions from model 2
        x: top x which are interleaved from preds_1 and preds_2

    Returns:
        ensemble of preds, from preds_1 and preds_2
    """
    ensemble_preds = []
    for doc_ids_1, doc_ids_2 in zip(preds_1, preds_2):
        ensemble_pred = top_rank_order(preds_1=[doc_ids_1[:x]], preds_2=[doc_ids_2[:x]])[0]
        ensemble_pred.extend(all_ordered_intersection(preds_1=[doc_ids_1[:]], preds_2=[doc_ids_2[x:]])[0])
        ensemble_preds.append(ensemble_pred)

    return ensemble_preds


class EnsembleStrategy:
    AOI = all_ordered_intersection
    TRO = top_rank_order
    TXIOI = top_x_interleaved_with_ordered_intersection
