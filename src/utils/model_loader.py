#!/usr/bin/env python3
# encoding: utf-8

"""
    Model loader method.

    Author : TF4ces
"""

# Native import

# Third-party imports

# # User imports
from config.conf import __SENTENCE_TRANSFORMERS_MODELS__
from src.TF4ces_search_engine.model.tf_idf import TFIDF
from src.TF4ces_search_engine.model.bm25 import FastBM25
from src.TF4ces_search_engine.model.sentence_transformer import Transformer


def load_model(model_name, model_path, emb_path, bl_train):
    """
    Wrapper to help load the model class of TF4ces Search Engine.

    Args:
        model_name: Name of the Model
        model_path: used for tfidf only.
        emb_path: used for Transformers based model only
        bl_train: used of tfidf only.

    Returns:
        Model (TF4cesBaseModel)
    """
    if model_name == 'tfidf':
        return TFIDF(model_path=model_path, retrain=bl_train)

    elif model_name == 'bm25':
        return FastBM25()

    elif model_name in __SENTENCE_TRANSFORMERS_MODELS__:
        return Transformer(model_url=model_name, model_path=None, emb_path=emb_path)

    else:
        raise Exception(f"Unknown model name : {model_name}")