#!/usr/bin/env python3
# encoding: utf-8

"""
    BM25 Model.

    Author : TF4ces
"""


# Native imports

# Third-party imports
from rank_bm25 import BM25Okapi
from fastbm25 import fastbm25
from tqdm.auto import tqdm
import numpy as np

# User imports
from src.TF4ces_search_engine.model.model_tf4ces import TF4cesBaseModel


class BM25(TF4cesBaseModel):

    def __init__(self, model_path=None, retrain=False):
        super().__init__(model_path=model_path)

        self.retrain = retrain
        self.model_path = model_path    # FIXME Not being used

        self.bm25 = None

    @staticmethod
    def init_model(docs):
        return BM25Okapi(corpus=docs)

    def get_scores(self, query):
        return self.bm25.get_scores(query)

    def train(self, docs):
        self.bm25 = self.init_model(docs=docs)
        print(f"Model [BM25] : Trained on {len(docs)} docs.")
        return None

    @staticmethod
    def tokenize(sentences):
        return [sentence.split() for sentence in sentences]

    def retrieve_documents(self, docs_obj, queries_obj, top_n, train=True):

        # Step 0 : get ids and docs
        doc_ids, docs, query_ids, queries = self.get_docs_n_queries(docs_obj=docs_obj, queries_obj=queries_obj)

        # TODO train and save bm25 weights, and seperate out prediction
        # Step 1 : Train BM25
        if self.retrain and train: self.train(docs=docs)

        # Step 2 : Get relevant docs
        score_matrix = list()
        print(f"Model [BM25] : Finding most relevant Docs for given Queries...")
        for query_id, query in tqdm(zip(query_ids, queries), total=len(query_ids), desc=f"Prediction"):
            score_matrix.append(self.get_scores(query))

        top_N_indexes = np.array(score_matrix).argsort(axis=1)[:, -top_n:][::-1]
        relevant_doc_ids = map(lambda indexes: np.array(doc_ids)[indexes], top_N_indexes)

        return (
            (query_id, np.array(queries_obj[query_id]['rel_doc_ids']), rel_doc_ids)
            for query_id, rel_doc_ids in zip(query_ids, relevant_doc_ids)
        )


class FastBM25(BM25):

    def __init__(self, model_path=None, retrain=False):
        super().__init__(model_path=model_path, retrain=retrain)

    @staticmethod
    def init_model(docs):
        return fastbm25(corpus=docs)

    def get_scores(self, query, top_n):
        pred = self.bm25.top_k_sentence(query, k=top_n)

        # If no prediction return empty set.
        if len(pred) == 0:
            return list()

        docs_, rel_doc_ids_, score_ = list(zip(*pred))
        return rel_doc_ids_

    def retrieve_documents(self, docs_obj, queries_obj, top_n, train=True):

        # Step 0 : get ids and docs
        doc_ids, docs, query_ids, queries = self.get_docs_n_queries(docs_obj=docs_obj, queries_obj=queries_obj)

        # TODO train and save bm25 weights, and seperate out prediction
        # Step 1 : Train BM25
        self.train(docs=docs)

        # Step 2 : Get relevant docs
        relevant_doc_ids = list()
        print(f"Model [BM25-Fast] : Finding most relevant Docs for given Queries...")
        for query_id, query in tqdm(zip(query_ids, queries), total=len(query_ids), desc=f"Prediction"):
            rel_doc_ids_ = self.get_scores(query=query, top_n=top_n)
            relevant_doc_ids.append(np.array(rel_doc_ids_))

        return (
            (query_id, np.array(queries_obj[query_id]['rel_doc_ids']), rel_doc_ids)
            for query_id, rel_doc_ids in zip(query_ids, relevant_doc_ids)
        )
