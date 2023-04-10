#!/usr/bin/env python3
# encoding: utf-8

"""
    Test Script for TF4cesSearchEnsemble.

    Author : TF4ces
"""

# Native imports
from copy import deepcopy
import random
# Third-party imports

# User imports
from config.conf import __WORKSPACE__, __ALL_MODELS__
from src.TF4ces_search_engine.data.data_gathering import DataGathering
from src.main_ensemble import TF4cesSearchEnsemble
from src.utils.ensemble_strategy import EnsembleStrategy


# Model Names
TFIDF_MODEL = __ALL_MODELS__[0]
BM25_MODEL = __ALL_MODELS__[1]
MPNET_MODEL = __ALL_MODELS__[2]
ROBERTA_MODEL = __ALL_MODELS__[3]


# Dataset Configs
DATASET_NAME = 'lotte'
DATASET_CATEGORY = 'lifestyle'
SPLIT = 'test'


# Path Configs
TFIDF_MODEL_PATH = __WORKSPACE__ / "models/tfidf/lotte/lifestyle/tfidf.v0.0.1.pkl"
MPNET_EMB_PATH = __WORKSPACE__ / "dataset/embeddings_test/test_v0.0.1/all-mpnet-base-v2/lotte/lifestyle"
ROBERTA_EMB_PATH = __WORKSPACE__ / "dataset/embeddings_test/test_v0.0.1/all-roberta-large-v1/lotte/lifestyle"


if __name__ == '__main__':

    FILTER_TOP_N = 3_000
    VOTING_TOP_N = 100

    FILTER_MODEL = {
        #TFIDF_MODEL: {'model_path': TFIDF_MODEL_PATH},
        BM25_MODEL: {'model_path': None},
    }
    VOTER_MODELS = {
        MPNET_MODEL: {'emb_path': MPNET_EMB_PATH},
        ROBERTA_MODEL: {'emb_path': ROBERTA_EMB_PATH},
    }

    # Step 0 : Gather Data (Documents)
    data_gathering = DataGathering(dataset_name=DATASET_NAME, )
    docs_obj = data_gathering.get_documents(dataset_category=DATASET_CATEGORY, dataset_split=SPLIT)
    queries_obj = data_gathering.get_queries(dataset_category=DATASET_CATEGORY, dataset_split=SPLIT)
    del data_gathering

    # Step 1 : Gather queries
    SAMPLE_QUERIES_SIZE = 2    # Select first 100 queries for testing.
    queries_obj = {id: data for id, data in queries_obj.items() if id < SAMPLE_QUERIES_SIZE}

    # Step 2: Init the Ensemble
    ensemble = TF4cesSearchEnsemble(
        filter_model_dict=FILTER_MODEL,
        voter_models_dict=VOTER_MODELS,
        docs_obj=deepcopy(docs_obj),
        filter_top_n=FILTER_TOP_N,
        voting_top_n=VOTING_TOP_N,
    )
    ensemble.summary()

    # Step 3: Load models
    ensemble.load_filter_model(bl_train=True)
    ensemble.load_voter_models()

    # Step 4: Filter Model predictions
    q_ids, filtered_rel_doc_ids = ensemble.filter_docs(queries_obj=deepcopy(queries_obj))
    ensemble.evaluate(q_ids=q_ids, preds_doc_ids=filtered_rel_doc_ids, K=5, queries_obj=queries_obj, to_display="Filter Model")

    # Step 5: Voter Model predictions
    ensemble.load_embeddings(filtered_rel_doc_ids=filtered_rel_doc_ids)
    ensemble.find_most_relevant_docs(q_ids=q_ids, filtered_rel_doc_ids=filtered_rel_doc_ids, queries_obj=queries_obj)

    # Step 6: Evaluation
    preds = ensemble.ensemble_voting(ensemble_strategy=EnsembleStrategy.TXIOI)
    ensemble.evaluate(q_ids=q_ids, preds_doc_ids=preds, K=5, queries_obj=queries_obj, to_display="Voter Models")

    # Step 7: User queries
    user_queries = ["How to ride a cycle?", "Ways to build search Engine."]
    user_queries_obj = {q_id: {'query': query, 'rel_doc_ids': list()} for q_id, query in enumerate(user_queries)}
    q_ids, filtered_rel_doc_ids = ensemble.filter_docs(queries_obj=user_queries_obj)
    ensemble.load_embeddings(filtered_rel_doc_ids=filtered_rel_doc_ids)
    ensemble.find_most_relevant_docs(q_ids=q_ids, filtered_rel_doc_ids=filtered_rel_doc_ids, queries_obj=user_queries_obj)
    preds = ensemble.ensemble_voting(ensemble_strategy=EnsembleStrategy.TXIOI)
    top_5 = preds[0][:5], queries_obj[0]

    # ensemble.evaluate(q_ids=q_ids, preds_doc_ids=filtered_rel_doc_ids, K=5, queries_obj=queries_obj, to_display="Filter Model")

    del ensemble
    print("DONE")
