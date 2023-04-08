#!/usr/bin/env python3
# encoding: utf-8

"""
    Model Template.

    Author : TF4ces
"""


# Native imports

# Third-party imports
import numpy as np

# User imports
from src.utils.data_obj_handler import get_ids_and_docs


class TF4cesBaseModel:

    def __init__(self, model_path):

        self.model_path = model_path
        self.model = None

    def load_model(self, ):
        pass

    def save_model(self):
        if not self.model_path.parent.exists(): self.model_path.parent.mkdir(parents=True, exist_ok=True)
        pass

    def init_model(self):
        pass

    def train(self, docs):
        pass

    @staticmethod
    def tokenize(sentences):
        return sentences

    def get_docs_n_queries(self, docs_obj, queries_obj):

        # Step 0 : get ids and docs
        doc_ids, docs = get_ids_and_docs(dict_obj=docs_obj, key="document")
        query_ids, queries = get_ids_and_docs(dict_obj=queries_obj, key="query")

        # Step 1: Queries and docs to be tokenized.
        docs, queries = self.tokenize(sentences=docs), self.tokenize(sentences=queries)

        return doc_ids, docs, query_ids, queries

    def predict(self, query):
        pass

    def retrieve_documents(self, docs_obj, queries_obj, top_n, train=True):

        # Step 0 : get ids and docs
        doc_ids, docs, query_ids, queries = self.get_docs_n_queries(docs_obj=docs_obj, queries_obj=queries_obj)

        # Step 1 : Train Model
        if train: self.train(docs=docs)

        # Step 2 : Get relevant docs
        relevant_doc_ids = None # TODO predicted doc_ids

        return (
            (query_id, np.array(queries_obj[query_id]['rel_doc_ids']), rel_doc_ids)
            for query_id, rel_doc_ids in zip(query_ids, relevant_doc_ids)
        )
