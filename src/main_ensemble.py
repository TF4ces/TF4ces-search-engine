#!/usr/bin/env python3
# encoding: utf-8

"""
    Main file for TF4ces Search Ensemble Engine.

    Author : TF4ces
"""

# Native imports

# Third-party imports
import numpy as np
from tqdm.auto import tqdm

# User imports
from config.conf import __WORKSPACE__
from src.utils.model_loader import load_model
from src.TF4ces_search_engine.feature.data_preprocessing import DataPreprocessing
from src.utils.evalutor import calc_mean_recall_n_precision, plot_precision_and_recall_curve


class TF4cesSearchEnsemble:

    def __init__(self, filter_model_dict, voter_models_dict, docs_obj, filter_top_n=10_000, voting_top_n=1_000):
        self.filter_model_dict = filter_model_dict
        self.voter_models_dict = voter_models_dict
        self.all_models = list(filter_model_dict.keys()) + list(voter_models_dict.keys())

        self.filter_top_n = filter_top_n
        self.voting_top_n = voting_top_n

        self.filter_models = dict()
        self.voter_models = dict()

        self.docs_obj = docs_obj

        self.embeddings = {model_name: dict() for model_name in self.all_models}
        self.rel_doc_ids = {model_name: dict() for model_name in self.all_models}

    def summary(self):
        print('-'*70)
        print(f"\t\tTF4ces Search Engine")
        print('-' * 70)
        print(f"Filter Model\t: {list(self.filter_model_dict.keys())}")
        print(f"Voter Models\t: {list(self.voter_models_dict.keys())}")
        print(f"Num of Docs\t: {len(self.docs_obj)}")
        print(f"Filter Top N\t: {self.filter_top_n}")
        print(f"Voting Top N\t: {self.voting_top_n}")
        print('-' * 70, end='\n\n')

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

    def filter_docs(self, queries_obj, bl_eval=True):
        # Step 1 : Get only the first filter model.
        model_name =  list(self.filter_models.keys())[0]
        model = list(self.filter_models.values())[0]

        # Step 2 : Data pre-processing (Docs)
        data_preprocessing = DataPreprocessing(cache_dir=__WORKSPACE__ / "dataset" / "preprocessed" / "ensemble" / model_name)
        docs_obj, _ = data_preprocessing.pre_process(
            docs=self.docs_obj, queries={}, model_type=model_name, use_cache=True
        )

        # Step 3 : Data pre-processing (Queries)
        queries_obj = data_preprocessing.pre_process_query(
            queries=queries_obj, model_type=model_name,
        )

        # Step 4 : Filtering docs by Filter model's predictions on top_n relevant docs, for each query
        print(f"Filter Model [{model_name}] : Retrieving top {self.filter_top_n} Docs for queries({len(queries_obj)}) documents({len(docs_obj)}).")
        pred_queries = model.retrieve_documents(
            docs_obj=docs_obj,
            queries_obj=queries_obj,
            top_n=self.filter_top_n,
            train=False,
        )
        q_ids, gold_doc_ids, pred_doc_ids = zip(*pred_queries)

        # Step 5 : Evaluation
        k = self.filter_top_n
        if bl_eval: calc_mean_recall_n_precision(gold_doc_ids=gold_doc_ids, pred_doc_ids=pred_doc_ids, k=k, to_display=f'Filter Model', bl_print=True)

        self.q_ids = q_ids
        self.rel_doc_ids[model_name] = pred_doc_ids
        return self.q_ids, self.rel_doc_ids[model_name]

    def read_embeddings(self, doc_ids, emb_path, model_name):
        embds = list()
        if not emb_path.exists(): emb_path.mkdir(parents=True, exist_ok=True)

        # Load embeddings from disk, for given doc_ids.
        for id in tqdm(doc_ids, desc=f"Voting Model [{model_name}] : Loading Embeddings"):
            path = emb_path / f"{id}.npy"
            try:
                embds.append(np.load(path))

            except FileNotFoundError:
                raw_document = self.docs_obj[id]['document']
                self.voter_models[model_name].get_batched_embeddings(doc_ids=[id], raw_documents=[raw_document], emb_path=emb_path, cache=True)
                embds.append(np.load(path))

        return embds

    def load_embeddings(self, filtered_rel_doc_ids):

        # Step 1 : Get all doc ids relevant for all the queries to be loaded in memory.
        all_doc_ids = set([doc_id for doc_ids in filtered_rel_doc_ids for doc_id in doc_ids])

        # Step 2 : Load all_doc_ids embeddings, for each voter model.
        for model_name, model in self.voter_models.items():
            emb_path = model.emb_path / 'docs'

            # Step 3 : Read embeddings from disk and save in memory
            for id, emb in zip(all_doc_ids, self.read_embeddings(doc_ids=all_doc_ids, emb_path=emb_path, model_name=model_name)):
                self.embeddings[model_name][id] = emb

    def load_embeddings_if_not_present(self, filtered_rel_doc_ids):

        # Step 1 : Get all doc ids relevant for all the queries to be loaded in memory.
        all_doc_ids = set([doc_id for doc_ids in filtered_rel_doc_ids for doc_id in doc_ids])

        ids_not_present = list()
        # Step 2 : Load all_doc_ids embeddings, for each voter model.
        for model_name, model in self.voter_models.items():
            emb_path = model.emb_path / 'docs'

            # Step 3 : Read embeddings from disk and save in memory
            for id in all_doc_ids:
                emb = self.embeddings[model_name].get(id, None)
                if emb is None:
                    ids_not_present.append([id])

        if len(ids_not_present) > 0: self.load_embeddings(filtered_rel_doc_ids=ids_not_present)

    def find_most_relevant_docs(self, q_ids, filtered_rel_doc_ids, queries_obj):

        # Step 1 : get all queries as list of strings
        queries = [queries_obj[id]['query'] for id in q_ids]

        # Step 2: Iterate over each Voter model, and predict relevant documents for all queries
        for model_name, model in self.voter_models.items():
            model.debug = False
            pred_doc_ids = list()

            # Step 3: Iterate and predict relevant docs for each query.
            for idx, (q_id, query) in tqdm(
                    enumerate(zip(q_ids, queries)),
                    desc=f"Retrieve Top {self.voting_top_n} docs using Voter Model [{model_name}]",
                    total=len(q_ids),
            ):
                doc_ids = filtered_rel_doc_ids[idx]                                         # Get doc ids per query
                doc_embeddings = map(lambda id: self.embeddings[model_name][id], doc_ids)   # point to embeddings for doc_ids
                pred_queries = model.get_relevant_docs_using_embds(                         # Find relevant documents
                    query_ids=[q_id],
                    queries=[query],
                    doc_ids=doc_ids,
                    doc_embeddings=np.array(list(doc_embeddings)),
                    top_n=self.voting_top_n,
                )
                _, pred_doc_ids_ = zip(*pred_queries)
                pred_doc_ids.extend(pred_doc_ids_)

            # Step 4: update pred rel_doc_ids for each model.
            self.rel_doc_ids[model_name] = pred_doc_ids

    def ensemble_voting(self, ensemble_strategy):
        # Get two voting models, and merge there results using ensemble_strategy (TRO, AOI, TXIOI)
        model_1, model_2 = list(self.voter_models_dict.keys())[:2]
        return ensemble_strategy(
            self.rel_doc_ids[model_1],
            self.rel_doc_ids[model_2]
        )

    @staticmethod
    def evaluate(q_ids, queries_obj, preds_doc_ids, K=5, to_display=''):
        gold_doc_ids = [queries_obj[q_id]['rel_doc_ids'] for q_id in q_ids]
        return calc_mean_recall_n_precision(
            gold_doc_ids=gold_doc_ids,
            pred_doc_ids=preds_doc_ids,
            k=K,
            to_display=f"{to_display}",
            bl_print=True
        )

    @staticmethod
    def plot_eval_curve(q_ids, queries_obj, preds_doc_ids, max_k):
        gold_doc_ids = [queries_obj[q_id]['rel_doc_ids'] for q_id in q_ids]
        plot_precision_and_recall_curve(gold_doc_ids, preds_doc_ids, max_k=max_k)
