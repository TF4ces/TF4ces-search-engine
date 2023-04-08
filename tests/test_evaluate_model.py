#!/usr/bin/env python3
# encoding: utf-8

"""
    Main test script to test and eval model.

    Author : TF4ces
"""


# Native imports

# Third-party imports

# User imports
from config.conf import __WORKSPACE__, __ALL_MODELS__
from src.main import TF4cesFlow


# Param Configs
TOP_N = 10_000  # Retrieve top N documents for each query.
K = 1_000
VERSION = 'v0.0.1'
TEST_RUN = False


# Pre-processing configs
USE_CACHE = True
PREPROCESS_CACHE_DIR = __WORKSPACE__ / "dataset" / "preprocessed" / f"test_{VERSION}"  # pre processed data is stored here.


# Model configs
MODEL = __ALL_MODELS__[0] # tfidf, bm25, all-mpnet-base-v2, all-roberta-large-v1, Intel/ColBERT-NQ
MODEL_PATH = __WORKSPACE__ / "models"


if __name__ == '__main__':

    print(f"Model Selected : {MODEL}")

    pipeline = TF4cesFlow(
        model_name=MODEL,
        dataset_name="lotte",
        dataset_category="lifestyle",
        top_n=TOP_N,
        k=K,
        use_cache=USE_CACHE,
        version=VERSION,
        preprocess_cache_dir=PREPROCESS_CACHE_DIR,
        model_path=MODEL_PATH,
    )

    # Development data
    SPLIT = 'dev'
    print(f"Starting Pipeline for data : {SPLIT}")
    pipeline.gather_data(split=SPLIT)                   # Data gathering
    if TEST_RUN: pipeline.small_test(split=SPLIT)       # DEBUG ONLY
    pipeline.data_preprocessing(split=SPLIT)            # Data Preprocessing
    q_ids, gold_doc_ids, pred_doc_ids = pipeline.retrieval(split=SPLIT, bl_train=True)          # Retrieval
    pipeline.evaluate(gold_doc_ids=gold_doc_ids, pred_doc_ids=pred_doc_ids, k=5)                # Eval k=5

    # Test data
    SPLIT = 'test'
    print(f"Starting Pipeline for data : {SPLIT}")
    pipeline.gather_data(split=SPLIT)                   # Data gathering
    if TEST_RUN: pipeline.small_test(split=SPLIT)       # DEBUG ONLY
    pipeline.data_preprocessing(split=SPLIT)            # Data Preprocessing
    q_ids, gold_doc_ids, pred_doc_ids = pipeline.retrieval(split=SPLIT, bl_train=False)         # Retrieval
    pipeline.evaluate(gold_doc_ids=gold_doc_ids, pred_doc_ids=pred_doc_ids, k=5)                # Eval k=5
    pipeline.evaluate(gold_doc_ids=gold_doc_ids, pred_doc_ids=pred_doc_ids, k=10)               # Eval k=10
    pipeline.evaluate(gold_doc_ids=gold_doc_ids, pred_doc_ids=pred_doc_ids, k=100)              # Eval k=100

    print("DONE")
