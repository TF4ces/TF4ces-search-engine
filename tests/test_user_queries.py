#!/usr/bin/env python3
# encoding: utf-8

"""
    Test Script for TF4cesSearchEnsemble with user queries.

    Author : TF4ces
"""

# Native imports
from copy import deepcopy

# Third-party imports
import pandas as pd

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
        TFIDF_MODEL: {'model_path': TFIDF_MODEL_PATH},
        # BM25_MODEL: {'model_path': None},
    }
    VOTER_MODELS = {
        MPNET_MODEL: {'emb_path': MPNET_EMB_PATH},
        ROBERTA_MODEL: {'emb_path': ROBERTA_EMB_PATH},
    }

    # Step 0 : Gather Data (Documents)
    data_gathering = DataGathering(dataset_name=DATASET_NAME, )
    docs_obj = data_gathering.get_documents(dataset_category=DATASET_CATEGORY, dataset_split=SPLIT)
    del data_gathering

    # Step 1 : Gather queries
    user_queries = [
        'are clear pomegranate seeds good to eat?',
        'do nutmeg and cinnamon go together?',
        'Does this Oat Milk have added sugar?',
        'How do I flip an egg when preparing it over easy?',
        'do you need to refrigerate homemade oil and vinegar dressing?',
    ]
    user_queries_obj = {q_id: {'query': query, 'rel_doc_ids': list()} for q_id, query in enumerate(user_queries)}

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
    ensemble.load_filter_model(bl_train=False)
    ensemble.load_voter_models()

    # Step 4: Filter Model predictions
    q_ids, filtered_rel_doc_ids = ensemble.filter_docs(queries_obj=deepcopy(user_queries_obj), bl_eval=False)

    # Step 5: Voter Model predictions
    ensemble.load_embeddings(filtered_rel_doc_ids=filtered_rel_doc_ids)
    ensemble.find_most_relevant_docs(q_ids=q_ids, filtered_rel_doc_ids=filtered_rel_doc_ids, queries_obj=user_queries_obj)

    # Step 6: Predictions
    preds = ensemble.ensemble_voting(ensemble_strategy=EnsembleStrategy.TXIOI)

    pred_dfs = dict()
    for (q_id, q_data), pred_doc_ids in zip(user_queries_obj.items(), preds):
        # print(q_id)
        # docs = list(map(lambda doc_id: docs_obj[doc_id]['document'], pred_doc_ids[:5]))
        pred_dfs[q_id] = pd.DataFrame(
            map(lambda doc_id: docs_obj[doc_id]['document'], pred_doc_ids[:5]),
            columns=[q_data['query']],
            index=pred_doc_ids[:5]
        )
        # pred_dfs[q_id] = df

    print(pred_dfs)
    del ensemble
    print("DONE")
