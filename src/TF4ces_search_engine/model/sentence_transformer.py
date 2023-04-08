#!/usr/bin/env python3
# encoding: utf-8

"""
    Model Sentence Transformers.

    More info on the pre-trained models
        : https://www.sbert.net/docs/pretrained_models.html

    Author : TF4ces
"""


# Native imports
import multiprocessing
import gc

# Third-party imports
import torch
import numpy as np
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# User imports
from src.TF4ces_search_engine.model.model_tf4ces import TF4cesBaseModel


class Transformer(TF4cesBaseModel):

    def __init__(self, model_url, model_path, emb_path):
        super().__init__(model_path=model_path)
        self.model_url = model_url
        self.model_path = model_path
        self.device = None
        self.debug = True

        self.model = self.load_model()

        self.emb_path = emb_path

    def load_model(self, ):
        model = SentenceTransformer(self.model_url)
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        model.to(self.device)
        print(f"Model [{self.model_url}] : Loaded on '{self.device}'")
        return model

    def encode(self, raw_documents):
        return self.model.encode(raw_documents)

    def save_embeddings(self, embds, doc_ids, emb_path):
        for id, embd in zip(doc_ids, embds):
            path = emb_path / f"{id}.npy"
            np.save(path, embd)

    def read_embeddings(self, doc_ids, emb_path):
        embds = list()
        for id in doc_ids:
            path = emb_path / f"{id}.npy"
            try:
                embds.append(np.load(path))
            except FileNotFoundError:
                return None
        return embds

    def get_batched_embeddings(self, doc_ids, raw_documents, emb_path, cache=True):
        batch_embds = self.read_embeddings(doc_ids=doc_ids, emb_path=emb_path) if cache else None
        if batch_embds is None:
            batch_embds = self.encode(raw_documents)
            if cache: self.save_embeddings(embds=batch_embds, doc_ids=doc_ids, emb_path=emb_path)
        return batch_embds

    def get_embeddings(self, doc_ids, raw_documents, type='docs', cache=True):
        emb_path = self.emb_path / type
        #print(emb_path)
        if cache and not emb_path.exists(): emb_path.mkdir(parents=True, exist_ok=True)

        bs = 10
        pool_size = 1

        batched_data = list()
        for idx in range(0, len(doc_ids), bs):
            start_idx, end_idx = idx, idx+bs

            batch_doc_ids = doc_ids[start_idx:end_idx]
            batch_docs = raw_documents[start_idx:end_idx]

            batched_data.append([batch_doc_ids, batch_docs, emb_path, cache])

        if pool_size > 1 and self.device == "cpu":
            worker_pool = multiprocessing.Pool(pool_size)
            embeddings = np.array(worker_pool.starmap(
                func=self.get_batched_embeddings,
                iterable=tqdm(batched_data, total=len(batched_data), desc=f"Generating '{type}' embeddings [Batch Size: {bs}]"),
                chunksize=1
            ))
            worker_pool.close()
            worker_pool.join()
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
            
        elif self.debug == False:
            embeddings = list()
            for data in batched_data:
                embeddings.extend(self.get_batched_embeddings(*data))
            embeddings = np.array(embeddings)

        else:
            embeddings = list()
            for data in tqdm(batched_data, desc=f"Generating/Loading '{type}' embeddings [Batch Size: {bs}]"):
                embeddings.extend(self.get_batched_embeddings(*data))
            embeddings = np.array(embeddings)

        return embeddings

    def get_relevant_docs_using_embds(self, query_ids, queries, doc_ids, doc_embeddings, top_n):

        # Step 1 Generate embeddings for queries
        query_embeddings = self.get_embeddings(doc_ids=query_ids, raw_documents=queries, type="queries", cache=True)

        #################################################################
        batch_size = 1000
        top_N_indexes = list()

        # print(f"Model [{self.model_url}] : Finding cosine similarities between Queries & Docs...")
        for i in range(0, len(query_ids), batch_size):
            top_N_indexes.extend(cos_sim(query_embeddings[i:i+batch_size], doc_embeddings).argsort(axis=1, descending=True)[:, :top_n])
        #################################################################

        # Step 2 : Get relevant docs
        relevant_doc_ids = map(lambda indexes: np.array(doc_ids)[indexes], top_N_indexes)

        return (
            (query_id, rel_doc_ids)
            for query_id, rel_doc_ids in zip(query_ids, relevant_doc_ids)
        )

    def retrieve_documents(self, docs_obj, queries_obj, top_n, train=True):

        # Step 0 : get ids and docs
        doc_ids, docs, query_ids, queries = self.get_docs_n_queries(docs_obj=docs_obj, queries_obj=queries_obj)

        # Step 1 Generate embeddings
        doc_embeddings = self.get_embeddings(doc_ids=doc_ids, raw_documents=docs, type="docs", cache=True)
        query_embeddings = self.get_embeddings(doc_ids=query_ids, raw_documents=queries, type="queries", cache=True)

        #################################################################
        batch_size = 1000  
        top_N_indexes = list()

        print(f"Model [{self.model_url}] : Finding cosine similarities between Queries & Docs...")
        for i in range(0, len(query_ids), batch_size):
            top_N_indexes.extend(cos_sim(query_embeddings[i:i+batch_size], doc_embeddings).argsort(axis=1, descending=True)[:, :top_n])
        #################################################################

        # Step 2 : Get relevant docs
        relevant_doc_ids = map(lambda indexes: np.array(doc_ids)[indexes], top_N_indexes)

        return (
            (query_id, np.array(queries_obj[query_id]['rel_doc_ids']), rel_doc_ids)
            for query_id, rel_doc_ids in zip(query_ids, relevant_doc_ids)
        )
