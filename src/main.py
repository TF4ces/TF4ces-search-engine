#!/usr/bin/env python3
# encoding: utf-8

"""
    Main file for search engine

    Author : TF4ces
"""


# Native imports
import itertools

# Third-party imports
import numpy as np

# User imports
from config.conf import __WORKSPACE__, __SENTENCE_TRANSFORMERS_MODELS__
from src.TF4ces_search_engine.data.data_gathering import DataGathering
from src.TF4ces_search_engine.feature.data_preprocessing import DataPreprocessing
from src.TF4ces_search_engine.model.tf_idf import TFIDF
from src.TF4ces_search_engine.model.sentence_transformer import Transformer
from src.TF4ces_search_engine.model.bm25 import BM25, FastBM25
from src.utils.evalutor import mean_recall_K, mean_precision_K


class TF4cesFlow:

    def __init__(
            self,
            model_name,
            dataset_name="lotte",
            dataset_category="lifestyle",
            top_n=5,
            k=5,
            use_cache=True,
            version='v0.0.1-test',
            preprocess_cache_dir=__WORKSPACE__ / "dataset" / "preprocessed" / f"test_",
            model_path=__WORKSPACE__ / "models"
    ):
        self.version = version
        self.dataset_name = dataset_name
        self.dataset_category = dataset_category

        self.top_n = top_n
        self.k = k

        # data gathering
        self.splits = ('dev', 'test')
        self.data = {
            split: {'docs': None, 'queries': None}
            for split in self.splits
        }

        # data pre-processing
        self.use_cache = use_cache
        self.preprocess_cache_dir = preprocess_cache_dir / model_name / self.dataset_name / self.dataset_category

        # model
        self.model = None
        self.model_name = model_name
        self.emb_path = __WORKSPACE__ / "dataset" / "embeddings" / f"test_{self.version}" / model_name / self.dataset_name / self.dataset_category
        self.model_path = model_path / self.model_name / self.dataset_name / self.dataset_category / f"{self.model_name}.{self.version}.pkl"

    def gather_data(self, split='dev'):
        data_gathering = DataGathering(dataset_name=self.dataset_name, )
        self.data[split]['docs'] = data_gathering.get_documents(dataset_category=self.dataset_category, dataset_split=split)
        self.data[split]['queries'] = data_gathering.get_queries(dataset_category=self.dataset_category, dataset_split=split)

    def small_test(self, split='dev'):
        self.data[split]['docs'] = dict(itertools.islice(self.data[split]['docs'].items(), 1_000, 3_000))
        self.data[split]['queries'] = dict(itertools.islice(self.data[split]['queries'].items(), 1_000))
        if '-small' not in self.preprocess_cache_dir.name:
            self.preprocess_cache_dir = self.preprocess_cache_dir.parent / f"{self.preprocess_cache_dir.name}-small"

    def data_preprocessing(self, split='dev'):
        data_preprocessing = DataPreprocessing(cache_dir=self.preprocess_cache_dir / split)
        self.data[split]['docs'], self.data[split]['queries'] = data_preprocessing.pre_process(
            docs=self.data[split]['docs'],
            queries=self.data[split]['queries'],
            model_type=self.model_name,
            use_cache=self.use_cache
        )

    def get_model(self, bl_train):
        if self.model_name == 'tfidf':
            self.model = TFIDF(model_path=self.model_path, retrain=bl_train)

        elif self.model_name == 'bm25':
            self.model = FastBM25()

        elif self.model_name in __SENTENCE_TRANSFORMERS_MODELS__:
            self.model = Transformer(model_url=self.model_name, model_path=self.model_path, emb_path=self.emb_path)
        else:
            raise Exception(f"Unknown model name : {self.model_name}")

    def retrieval(self, split="dev", bl_train=False):
        print(f"Retrieving top {self.top_n} for queries({len(self.data[split]['queries'])}) documents({len(self.data[split]['docs'])})...")
        self.get_model(bl_train=bl_train)
        pred_queries = self.model.retrieve_documents(
            docs_obj=self.data[split]['docs'],
            queries_obj=self.data[split]['queries'],
            top_n=self.top_n,
            train=bl_train,
        )
        q_ids, gold_doc_ids, pred_doc_ids = zip(*pred_queries)

        print(f"Evaluation..")
        recall_k = mean_recall_K(golds=gold_doc_ids, preds=pred_doc_ids, k=self.k)
        precision_k = mean_precision_K(golds=gold_doc_ids, preds=pred_doc_ids, k=self.k)

        print(f"Recall@{self.k} : {recall_k}")
        print(f"Precision@{self.k} : {precision_k}")
        print(f"Sample Doc Ids\nGold: {gold_doc_ids[:5]}\nPred: {pred_doc_ids[:5]}")
        return q_ids, gold_doc_ids, pred_doc_ids
