#!/usr/bin/env python3
# encoding: utf-8

"""
    TF-IDF Model

    Author : TF4ces
"""

# Native imports
import pickle

# Third-party imports
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
            print(f"Loaded TFIDF model with vocab_size : {len(tfidf.get_feature_names_out())} from : '{self.model_path}'")
            return tfidf

        return TfidfVectorizer()

    def save_model(self, tfidf):
        if not self.model_path.parent.exists(): self.model_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(tfidf, open(self.model_path, 'wb'))
        print(f"TFIDF Model saved at '{self.model_path}'")

    def train(self, docs):
        tfidf = self.tfidf_vectorizer.fit(raw_documents=docs)
        print(f"TFIDF Model trained on {len(docs)} docs with vocab size : {len(self.tfidf_vectorizer.get_feature_names_out())}")
        self.save_model(tfidf=tfidf)
        return None

    def encode(self, raw_documents):
        return self.tfidf_vectorizer.transform(raw_documents=raw_documents)

    def retrieve_documents(self, docs_obj, queries_obj, top_n, train=False):

        # Step 0 : get ids and docs
        doc_ids, docs, query_ids, queries = self.get_docs_n_queries(docs_obj=docs_obj, queries_obj=queries_obj)

        # Step 1 : Train tf-idf
        if self.retrain and train: self.train(docs=docs)

        # Step 2 : Vectorize docs, and queries
        #doc_vecs, query_vecs = self.encode(raw_documents=docs), self.encode(raw_documents=queries)
        
        doc_embeddings, query_embeddings = self.encode(raw_documents=docs), self.encode(raw_documents=queries)
        print("Encoded")

        # Step 3 : Find similarity b/w, query and docs, and top n relevant docs.
        #################################################################
        batch_size = 1000  
        num_queries, num_docs = len(query_ids), len(doc_ids)
        top_N_indexes = np.zeros((num_queries, top_n), dtype=np.int32)

        for i in range(0, num_queries, batch_size):
            query_embeddings_batch = query_embeddings[i:i+batch_size]
            sim_matrix_batch = cosine_similarity(query_embeddings_batch, doc_embeddings)
            top_N_indexes_batch = sim_matrix_batch.argsort(axis=1)[:, -top_n:]
            top_N_indexes[i:i+batch_size] = top_N_indexes_batch
        ################################################################# 
        
        #top_N_indexes = cosine_similarity(query_vecs, doc_vecs).argsort(axis=1)[:, -top_n:]
        relevant_doc_ids = map(lambda indexes: np.array(doc_ids)[indexes], top_N_indexes)

        return (
            (query_id, np.array(queries_obj[query_id]['rel_doc_ids']), rel_doc_ids)
            for query_id, rel_doc_ids in zip(query_ids, relevant_doc_ids)
        )