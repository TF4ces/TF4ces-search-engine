#!/usr/bin/env python3
# encoding: utf-8

"""
    Data dictionary handler.

    Author : TF4ces
"""


def get_ids_and_docs(dict_obj, key="document"):
    """
    Get doc ids and documents as tuples from dict or queries as per key.

    Args:
        dict_obj: dict of docs/queries
        key: query or document

    Returns:
        (doc_ids, docs) or (query_ids, queries) depending on key=document or key=query respectively.
    """
    return tuple(zip(*[
        (doc_id, doc_val[key])
        for doc_id, doc_val in dict_obj.items()
    ]))
