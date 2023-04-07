#!/usr/bin/env python3
# encoding: utf-8

"""
    Main test script to test and eval model.

    Author : TF4ces
"""

# Native imports
import sys
sys.path.append('/home/jovyan/teaching_material/IR/TF4ces-search-engine/') #Change the path if running on jupyter hub
import pickle

# Third-party imports
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers.util import cos_sim

# User imports
from config.conf import __WORKSPACE__, __ALL_MODELS__, __SENTENCE_TRANSFORMERS_MODELS__
from src.utils.model_loader import load_model
from src.TF4ces_search_engine.data.data_gathering import DataGathering
from src.TF4ces_search_engine.feature.data_preprocessing import DataPreprocessing
from src.utils.data_obj_handler import get_ids_and_docs
from src.utils.evalutor import mean_recall_K, mean_precision_K


TFIDF_MODEL = __ALL_MODELS__[0]
BM25_MODEL = __ALL_MODELS__[1]
MPNET_MODEL = __ALL_MODELS__[2]
ROBERTA_MODEL = __ALL_MODELS__[3]


class TF4cesSearchEnsemble:

    def __init__(self, filter_model_dict, voter_models_dict, docs_obj):
        self.filter_model_dict = filter_model_dict
        self.voter_models_dict = voter_models_dict
        self.all_models = list(filter_model_dict.keys()) + list(voter_models_dict.keys())

        self.filter_models = dict()
        self.voter_models = dict()

        self.docs_obj = docs_obj

        self.embeddings = {model_name: dict() for model_name in self.all_models}
        self.rel_doc_ids = {model_name: dict() for model_name in self.all_models}

    def load_filter_model(self, bl_train=False):
        for model_name, meta in self.filter_model_dict.items():
            self.filter_models[model_name] = load_model(
                model_name=model_name,
                model_path=meta['model_path'],
                emb_path=None,
                bl_train=bl_train
            )

    def load_voter_models(self):
        for model_name, meta in self.voter_models_dict.items():
            self.voter_models[model_name] = load_model(
                model_name=model_name,
                model_path=None,
                emb_path=meta['emb_path'],
                bl_train=False
            )

    def filter_docs(self, queries_obj, top_n=10_000):
        # Only the first filter model is used.
        model_name =  list(self.filter_models.keys())[0]
        model = list(self.filter_models.values())[0]

        # Data pre-processing
        data_preprocessing = DataPreprocessing(cache_dir=__WORKSPACE__ / "dataset" / "preprocessed" / "ensemble" / model_name)
        docs_obj, _ = data_preprocessing.pre_process(
            docs=self.docs_obj, queries={}, model_type=model_name, use_cache=True
        )
        queries_obj = data_preprocessing.pre_process_query(
            queries=queries_obj, model_type=model_name,
        )

        # Filtering
        print(f"Retrieving top {top_n} for queries({len(queries_obj)}) documents({len(docs_obj)}) using filter Model : '{model_name}'.")
        pred_queries = model.retrieve_documents(
            docs_obj=docs_obj,
            queries_obj=queries_obj,
            top_n=top_n,
            train=False,
        )
        q_ids, gold_doc_ids, pred_doc_ids = zip(*pred_queries)
        k = 100
        r = mean_recall_K(golds=gold_doc_ids, preds=pred_doc_ids, k=k)
        print(f"Recall@{k} with filter Model : {r}")

        self.q_ids = q_ids
        self.rel_doc_ids[model_name] = pred_doc_ids
        return self.q_ids, self.rel_doc_ids[model_name]

    def load_embeddings(self, filtered_rel_doc_ids):
        all_doc_ids = list(set(np.array(filtered_rel_doc_ids).reshape(-1)))
        for model_name, model in self.voter_models.items():
            emb_path = model.emb_path / 'docs'
            for id, emb in tqdm(
                zip(all_doc_ids, model.read_embeddings(doc_ids=all_doc_ids, emb_path=emb_path)),
                desc=f"Loading Embeddings for Model '{model_name}'",
                total=len(all_doc_ids)
            ):
                self.embeddings[model_name][id] = emb

    def find_most_relevant_docs(self, q_ids, filtered_rel_doc_ids, queries_obj, top_n=1_000):
        queries = [queries_obj[id]['query'] for id in q_ids]

        for model_name, model in self.voter_models.items():
            model.debug = False
            pred_doc_ids = list()
            for idx, (q_id, query) in tqdm(
                    enumerate(zip(q_ids, queries)),
                    desc=f"Retrive Top docs using Voter Model [{model_name}]",
                    total=len(q_ids),
            ):
                doc_ids = filtered_rel_doc_ids[idx]
                doc_embeddings = map(lambda id: self.embeddings[model_name][id], doc_ids)
                pred_queries = model.get_relevant_docs_using_embds(
                    query_ids=[q_id],
                    queries=[query],
                    doc_ids=doc_ids,
                    doc_embeddings=np.array(list(doc_embeddings)),
                    top_n=top_n
                )
                _, pred_doc_ids_ = zip(*pred_queries)
                pred_doc_ids.extend(pred_doc_ids_)
            self.rel_doc_ids[model_name] = pred_doc_ids

    def ensemble_voting(self, ensemble_strategy):
        model_1, model_2 = list(self.voter_models_dict.keys())[:2]
        return ensemble_strategy(
            ensemble.rel_doc_ids[model_1],
            ensemble.rel_doc_ids[model_2]
        )

    @staticmethod
    def evaluate(q_ids, preds_doc_ids, K=5, to_display=''):
        gold_doc_ids = [queries_obj[q_id]['rel_doc_ids'] for q_id in q_ids]
        recall_k = mean_recall_K(golds=gold_doc_ids, preds=preds_doc_ids, k=K)
        precision_k = mean_precision_K(golds=gold_doc_ids, preds=preds_doc_ids, k=K)
        print(f"{to_display} Recall@{K} : {recall_k}")
        print(f"{to_display} Precision@{K} : {precision_k}")


def all_ordered_intersection(preds_1, preds_2):
    """
    Finds the intersection of two predictions, aka. common predictions.
    Args:
        preds_1: predictions from model 1
        preds_2: predictions from model 2

    Returns:
        ensemble of preds, from preds_1 and preds_2
    """
    return [[doc_id for doc_id in doc_ids_1 if doc_id in doc_ids_2] for doc_ids_1, doc_ids_2 in zip(preds_1, preds_2)]


def top_rank_order(preds_1, preds_2):
    n = len(preds_1[0])

    ensemble_preds = []
    for pred_doc_ids_d1_, pred_doc_ids_d2_ in zip(preds_1, preds_2):
        ensemble_pred = []
        for pred_doc_id_d1_, pred_doc_id_d2_ in zip(pred_doc_ids_d1_, pred_doc_ids_d2_):
            if pred_doc_id_d1_ not in ensemble_pred:
                ensemble_pred.append(pred_doc_id_d1_)

            if pred_doc_id_d2_ not in ensemble_pred:
                ensemble_pred.append(pred_doc_id_d2_)

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

    Returns:
        ensemble of preds, from preds_1 and preds_2
    """
    ensemble_preds = []
    for doc_ids_1, doc_ids_2 in zip(preds_1, preds_2):
        ensemble_pred = top_rank_order(preds_1=[doc_ids_1[:x]], preds_2=[doc_ids_2[:x]])[0]
        ensemble_pred.extend(all_ordered_intersection(preds_1=[doc_ids_1[:]], preds_2=[doc_ids_2[x:]])[0])
        ensemble_preds.append(ensemble_pred)

    return ensemble_preds


if __name__ == '__main__':

    DATASET_NAME = 'lotte'
    DATASET_CATEGORY = 'lifestyle'
    SPLIT = 'test'

    TFIDF_MODEL_PATH = __WORKSPACE__ / "models/tfidf/lotte/lifestyle/tfidf.v0.0.1.pkl"
    MPNET_EMB_PATH = __WORKSPACE__ / "dataset/embeddings_test/test_v0.0.1/all-mpnet-base-v2/lotte/lifestyle"
    ROBERTA_EMB_PATH = __WORKSPACE__ / "dataset/embeddings_test/test_v0.0.1/all-roberta-large-v1/lotte/lifestyle"

    FILTER_MODEL = {
        TFIDF_MODEL: {'model_path': TFIDF_MODEL_PATH}
    }
    VOTER_MODELS = {
        MPNET_MODEL: {'emb_path': MPNET_EMB_PATH},
        ROBERTA_MODEL: {'emb_path': ROBERTA_EMB_PATH},
    }

    # Step 0 : Get Data
    data_gathering = DataGathering(dataset_name=DATASET_NAME, )
    docs_obj = data_gathering.get_documents(dataset_category=DATASET_CATEGORY, dataset_split=SPLIT)
    queries_obj = data_gathering.get_queries(dataset_category=DATASET_CATEGORY, dataset_split=SPLIT)
    del data_gathering

    # queries_obj = {id: data for id, data in queries_obj.items() if id < 3}

    # Step 1: Init the Ensemble
    ensemble = TF4cesSearchEnsemble(
        filter_model_dict=FILTER_MODEL,
        voter_models_dict=VOTER_MODELS,
        docs_obj=docs_obj,
    )

    # Step 2: Load models
    ensemble.load_filter_model()
    ensemble.load_voter_models()

    # Step 3: Filter Model predictions
    TOP_N = 10_000
    q_ids, filtered_rel_doc_ids = ensemble.filter_docs(queries_obj=queries_obj, top_n=TOP_N)
    ensemble.evaluate(q_ids=q_ids, preds_doc_ids=filtered_rel_doc_ids, K=5, to_display="Filter Model")

    # Step 4: Voter Model predictions
    TOP_N = 1_000
    ensemble.load_embeddings(filtered_rel_doc_ids=filtered_rel_doc_ids)
    ensemble.find_most_relevant_docs(q_ids=q_ids, filtered_rel_doc_ids=filtered_rel_doc_ids, queries_obj=queries_obj, top_n=TOP_N)


    # Step 5: Evaluation
    preds = ensemble.ensemble_voting(ensemble_strategy=top_x_interleaved_with_ordered_intersection)
    ensemble.evaluate(q_ids=q_ids, preds_doc_ids=preds, K=5, to_display="Voter Model")

    del ensemble
    print("DONE")



