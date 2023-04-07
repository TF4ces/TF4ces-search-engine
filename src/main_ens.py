#!/usr/bin/env python3
# encoding: utf-8

"""
    Main file for Novelty

    Author : TF4ces
"""

# Native imports
import itertools
import pickle
import gc

# Third-party imports
import numpy as np

# User imports
from config.conf import __WORKSPACE__, __SENTENCE_TRANSFORMERS_MODELS__
from src.TF4ces_search_engine.feature.data_preprocessing import DataPreprocessing
from src.TF4ces_search_engine.model.sentence_transformer import Transformer
from src.TF4ces_search_engine.model.tf_idf import TFIDF


class Ensemble:

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
        
        self.model = None
        self.model_name = model_name

        # data pre-processing
        self.use_cache = use_cache
        self.preprocess_cache_dir = preprocess_cache_dir / self.model_name / self.dataset_name / self.dataset_category

        # model
        self.emb_path = __WORKSPACE__ / "dataset" / "embeddings" / f"test_{self.version}" / self.model_name / self.dataset_name / self.dataset_category
        self.emb_test_path = __WORKSPACE__ / "dataset" / "embeddings_test" / f"test_{self.version}" / self.model_name / self.dataset_name / self.dataset_category
        self.model_path = model_path / self.model_name / self.dataset_name / self.dataset_category / f"{self.model_name}.{self.version}.pkl"
        self.dict_path = __WORKSPACE__ / "dataset" / "dictionary" / self.dataset_category

    def base_gather_data(self, split='dev'):
        # data_gathering = DataGathering(dataset_name=self.dataset_name, )
        self.data[split]['docs'] = pickle.load(open(self.dict_path / str("docs." + split + ".pkl"), 'rb'))
        self.data[split]['queries'] = pickle.load(open(self.dict_path / str("queries." + split + ".pkl"), 'rb'))
        
    def get_specific_ids(self,ids,key,split):
        values_subset = {i:v for i, v in self.data[split][key].items() if i in ids}
        return values_subset
        
    def gather_data(self, query_ids, doc_ids, split='dev'):
        # data_gathering = DataGathering(dataset_name=self.dataset_name, )
        self.data[split]['docs'] = pickle.load(open(self.dict_path / str("docs." + split + ".pkl"), 'rb'))
        self.data[split]['queries'] = pickle.load(open(self.dict_path / str("queries." + split + ".pkl"), 'rb'))
        self.data[split]['docs'] = self.get_specific_ids(doc_ids,key='docs',split=split)
        self.data[split]['queries'] = self.get_specific_ids([query_ids],key='queries',split=split)

    def small_test(self, split='dev', nqueries=1):
        #self.data[split]['docs'] = dict(itertools.islice(self.data[split]['docs'].items(), 1_000, 3_000))
        self.data[split]['queries'] = dict(itertools.islice(self.data[split]['queries'].items(), nqueries))
        print(f"Selected Queries : {len(self.data[split]['queries'])}")
        if '-small' not in self.preprocess_cache_dir.name:
            self.preprocess_cache_dir = self.preprocess_cache_dir.parent / f"{self.preprocess_cache_dir.name}-small"
        
    def data_preprocessing(self, split='dev', use_cache=True, cache_sub_dir=None):
        if cache_sub_dir is not None:
            cache_dir_path = self.preprocess_cache_dir / split / cache_sub_dir
        else:
            cache_dir_path = self.preprocess_cache_dir / split
        data_preprocessing = DataPreprocessing(cache_dir=cache_dir_path)
        self.data[split]['docs'], _ = data_preprocessing.pre_process(
            docs=self.data[split]['docs'],
            queries=self.data[split]['queries'],
            model_type=self.model_name,
            use_cache=use_cache
        )
    
    def query_preporcessing(self, split='dev', use_cache=True, cache_sub_dir=None):
        if cache_sub_dir is not None:
            cache_dir_path = self.preprocess_cache_dir / split / cache_sub_dir
        else:
            cache_dir_path = self.preprocess_cache_dir / split
        data_preprocessing = DataPreprocessing(cache_dir=cache_dir_path)
        self.data[split]['queries'] = data_preprocessing.pre_process_query(
            queries=self.data[split]['queries'], 
            model_type=self.model_name, 
            use_cache=False,
        )
    
    
    def get_model(self, bl_train):
        if self.model_name == 'tfidf':
            self.model = TFIDF(model_path=self.model_path, retrain=bl_train)

        elif self.model_name == 'bm25':
            self.model = FastBM25()

        elif self.model_name in __SENTENCE_TRANSFORMERS_MODELS__:
            if bl_train == True:
                self.model = Transformer(model_url=self.model_name, model_path=self.model_path, emb_path=self.emb_path)
            if bl_train == False:
                self.model = Transformer(model_url=self.model_name, model_path=self.model_path, emb_path=self.emb_test_path)
        else:
            raise Exception(f"Unknown model name : {self.model_name}")
    
    def base_retrieval(self, split="dev", bl_train=False):
        print(f"Retrieving top {self.top_n} for queries({len(self.data[split]['queries'])}) documents({len(self.data[split]['docs'])})...")
        self.get_model(bl_train=bl_train)
        pred_queries = self.model.retrieve_documents(
            docs_obj=self.data[split]['docs'],
            queries_obj=self.data[split]['queries'],
            top_n=self.top_n,
            train=bl_train,
        )
        q_ids, gold_doc_ids, pred_doc_ids = zip(*pred_queries)
        
        return q_ids, gold_doc_ids, pred_doc_ids

    def retrieval(self, split="dev", bl_train=False):
        #print(f"Retrieving top {self.top_n} for queries({len(self.data[split]['queries'])}) documents({len(self.data[split]['docs'])})...")
        #self.get_model(bl_train=bl_train)
        pred_queries = self.model.retrieve_documents(
            docs_obj=self.data[split]['docs'],
            queries_obj=self.data[split]['queries'],
            top_n=self.top_n,
            train=bl_train,
        )
        q_ids, gold_doc_ids, pred_doc_ids = zip(*pred_queries)
        
        return q_ids, gold_doc_ids, pred_doc_ids