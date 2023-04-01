#!/usr/bin/env python3
# encoding: utf-8

"""
    File Handlers.

    Author : TF4ces
"""

from rank_bm25 import BM25Okapi

class BM25():

    document_keys = []
    document_values = []

    queries_keys = []
    queries_values = []
    source_values = []
    rel_doc_ids_values = []

    retrieved_id = []

    bm25 = 0
    query_vec = 0

    def get_doc_items(self, docs):
        for key, value in docs.items():
            if isinstance(key, int):
                self.document_keys.append(key)
            if isinstance(value, dict):
                self.get_doc_items(value)
            else:
                if key == 'document':
                    self.document_values.append(value)

        return self.document_keys, self.document_values

    def get_query_items(self, queries):
        for key in queries:
            self.queries_keys.append(key)
            for inner_key in queries[key]:
                if inner_key == "query":
                    self.queries_values.append(queries[key][inner_key])

        return self.queries_keys, self.queries_values

    def doc_vectorize(self, docs):
        self.document_keys, self.document_values = self.get_doc_items(docs)
        self.bm25 = BM25Okapi(self.document_values)  #Todo fit transform for predictions.
        return self.bm25, self.document_keys

    def retrieve_documents(self, docs, queries, top_n):
        self.bm25, self.document_keys = self.doc_vectorize(docs)
        self.queries_keys, self.queries_values = self.get_query_items(queries)

        for id in self.queries_keys:
            query = self.queries_values[int(id)]
            score = self.bm25.get_scores(query)
            sorted_doc_ids = [doc_id for _, doc_id in sorted(zip(score, self.document_keys), reverse=True)]

            self.retrieved_id.append(sorted_doc_ids[:top_n])

        return self.queries_keys, self.retrieved_id


