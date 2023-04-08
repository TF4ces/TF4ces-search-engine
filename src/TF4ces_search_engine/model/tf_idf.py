#!/usr/bin/env python3
# encoding: utf-8

"""
    TF-IDF Model

    Author : TF4ces
"""

# Native imports
import pickle

# Third-party imports
from sklearn.exceptions import NotFittedError
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# User-module imports
from src.TF4ces_search_engine.model.model_tf4ces import TF4cesBaseModel


class TFIDF(TF4cesBaseModel):

    def __init__(self, model_path, retrain=False):
        super().__init__(model_path=model_path)

        self.retrain = retrain
        self.model_path = model_path

        self.tfidf_vectorizer = self.load_model()

    def load_model(self, ):

        if not self.retrain:
            tfidf = pickle.load(open(self.model_path, 'rb'))
            print(f"Model [TF-IDF] : Loaded with vocab_size ({len(tfidf.get_feature_names_out())}) from : '{self.model_path}'")
            return tfidf

        return TfidfVectorizer()

    def save_model(self, tfidf):
        if not self.model_path.parent.exists(): self.model_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(tfidf, open(self.model_path, 'wb'))
        print(f"Model [TF-IDF] : Saved at '{self.model_path}'")

    def train(self, docs):
        tfidf = self.tfidf_vectorizer.fit(raw_documents=docs)
        print(f"Model [TF-IDF] : Trained on {len(docs)} docs with vocab size : {len(self.tfidf_vectorizer.get_feature_names_out())}")
        self.save_model(tfidf=tfidf)
        return None

    def encode(self, raw_documents):
        try:
            return self.tfidf_vectorizer.transform(raw_documents=raw_documents)

        except NotFittedError:
            self.train(docs=raw_documents)
            return self.encode(raw_documents=raw_documents)

    def retrieve_documents(self, docs_obj, queries_obj, top_n, train=False):

        # Step 0 : get ids and docs
        doc_ids, docs, query_ids, queries = self.get_docs_n_queries(docs_obj=docs_obj, queries_obj=queries_obj)

        # Step 1 : Train tf-idf
        if self.retrain and train: self.train(docs=docs)

        # Step 2 : Vectorize docs, and queries
        doc_embeddings, query_embeddings = self.encode(raw_documents=docs), self.encode(raw_documents=queries)
        print(f"Model [TF-IDF] : Vector embeddings generated for queries({len(queries)}) and docs ({len(docs)})")

        # Step 3 : Find similarity b/w, query and docs, and top n relevant docs.
        #################################################################
        batch_size = 1000
        top_N_indexes = list()
        print(f"Model [TF-IDF] : Finding cosine similarities between Queries & Docs...")

        for i in range(0, len(query_ids), batch_size):
            top_N_indexes.extend(
                cosine_similarity(
                    query_embeddings[i:i+batch_size], doc_embeddings
                ).argsort(axis=1)[:, ::-1][:, :top_n]  # Argsort docs, and filter top_n, reverse for descending
            )

        relevant_doc_ids = map(lambda indexes: np.array(doc_ids)[indexes], top_N_indexes)
        #################################################################

        return (
            (query_id, np.array(queries_obj[query_id]['rel_doc_ids']), rel_doc_ids)
            for query_id, rel_doc_ids in zip(query_ids, relevant_doc_ids)
        )