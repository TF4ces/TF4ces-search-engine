#!/usr/bin/env python3
# encoding: utf-8

"""
    Main test script to test and eval model.

    Author : TF4ces
"""


# Native imports
import itertools

# Third-party imports
import numpy as np

# User imports
from config.conf import __WORKSPACE__
from src.TF4ces_search_engine.data.data_gathering import DataGathering
from src.TF4ces_search_engine.feature.data_preprocessing import DataPreprocessing
from src.TF4ces_search_engine.model.tf_idf import Tfidf, TFIDF
from src.TF4ces_search_engine.model.bm25 import BM25, FastBM25
from src.utils.evalutor import mean_recall_K, mean_precision_K


if __name__ == '__main__':

    TOP_N = 100  # Retrieve top 5 documents for each query.
    K = 10

    VERSION = 'v0.0.1'  #v0.0.1-small
    TEST_RUN = False

    # Pre-processing configs
    USE_CACHE = True
    PREPROCESS_CACHE_DIR = __WORKSPACE__ / "dataset" / "preprocessed" / f"test_{VERSION}"  # pre processed data is stored here.

    # Model configs
    MODEL = "bm25" # tfidf, bm25
    MODEL_PATH = __WORKSPACE__ / "models" / f"{MODEL}" / f"{MODEL}.{VERSION}.pkl"



    # Data Gathering
    data_gathering = DataGathering(dataset_name="lotte",)
    docs = data_gathering.get_documents(dataset_category="lifestyle", dataset_split="dev")
    queries = data_gathering.get_queries(dataset_category="lifestyle", dataset_split="dev")

    if TEST_RUN:
        docs = dict(itertools.islice(docs.items(), 1_000, 3_000))
        queries = dict(itertools.islice(queries.items(), 1_000))


    # Data Preprocessing
    print("Preprocessing in Progress...")
    data_preprocessing = DataPreprocessing(cache_dir=PREPROCESS_CACHE_DIR)
    docs, queries = data_preprocessing.pre_process(docs=docs, queries=queries, model_type=MODEL, use_cache=USE_CACHE)


    print(f"Retrieving top {TOP_N} for queries({len(queries)}) documents({len(docs)})...")
    if MODEL == "tfidf":
        RETRAIN = False

        model = TFIDF(model_path=MODEL_PATH, retrain=RETRAIN)
        pred_queries = model.retrieve_documents(
            docs_obj=docs,
            queries_obj=queries,
            top_n=TOP_N,
            train=RETRAIN,
        )

    if MODEL == "bm25":
        bm25 = FastBM25(model_path=MODEL_PATH, retrain=True)
        pred_queries = bm25.retrieve_documents(
            docs_obj=docs,
            queries_obj=queries,
            top_n=TOP_N,
            train=True,
        )


    # Evaluation
    print("Evaluation...")
    q_ids, gold_doc_ids, pred_doc_ids = zip(*pred_queries)
    recall_k = mean_recall_K(golds=gold_doc_ids, preds=pred_doc_ids, k=K)
    precision_k = mean_precision_K(golds=gold_doc_ids, preds=pred_doc_ids, k=K)

    print(f"Recall@{K} : {recall_k}")
    print(f"Precision@{K} : {precision_k}")

    print(f"Sample Doc Ids\nGold: {gold_doc_ids[:5]}\nPred: {pred_doc_ids[:5]}")

    print("DONE")
