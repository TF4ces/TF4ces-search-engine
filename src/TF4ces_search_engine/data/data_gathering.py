#!/usr/bin/env python3
# encoding: utf-8

"""
    File to read dataset

    Author : TF4ces

    Troubleshooting :
        For windows the doc.iter might throw decoding error while reading tsv file,
        and might need to change the encoding in source files as per this issue
        https://github.com/allenai/ir_datasets/issues/208#issuecomment-1235944338

    Warnings :
        Use ir-datasets==0.4.1 version
"""

import ir_datasets
from pathlib import Path


class DataGathering():

    def __init__(self, dataset_name="lotte"):
        self.dataset_name = dataset_name

    def lotte_documents(self, dataset_category="lifestyle", dataset_split="dev"):

        # Step 1 : Load dataset
        dataset_name = Path(self.dataset_name) / dataset_category / dataset_split
        data = ir_datasets.load(str(dataset_name.as_posix()))

        # Step 2 : Get Documents
        documents = {}
        for item in data.docs_iter():
            doc_id = int(item.doc_id)
            doc = item.text
            documents[doc_id] = {"document": doc}

        return documents
        
    def lotte_queries(self, dataset_category="lifestyle", dataset_split="dev"):
        q_idx = 0
        query_sources = ['search', 'forum']
        queries = {}

        for query_source in query_sources:
            tmp_query_mappings = dict()

            dataset_name = Path(self.dataset_name) / dataset_category / dataset_split / query_source
            data = ir_datasets.load(str(dataset_name.as_posix()))

            for q_id, query in data.queries_iter():
                tmp_query_mappings[q_id] = q_idx
                queries[q_idx] = {'query': query, 'source': str(dataset_name / q_id), 'rel_doc_ids': list()}
                q_idx += 1

            for item in data.qrels_iter():
                rel_doc_id = int(item.doc_id)
                tmp_q_idx = tmp_query_mappings[item.query_id]
                queries[tmp_q_idx]['rel_doc_ids'] += [rel_doc_id]

            del tmp_query_mappings

        return queries
        
    def get_documents(self, dataset_category="lifestyle", dataset_split="dev"):
        """
        Help get documents as a Dict.
        
        Args: 
            dataset_type : train or test type
            dataset_category : sub-category for Lotte includes, ('all', 'lifestyle', ..)
            
            
        Return:
            Dict, of doc ids and docs
            
        Example : 
            {
                1: {"document": "document 1"},
                2: {"document": "document 2"},
            }
        """
        
        if self.dataset_name == "lotte":
            return self.lotte_documents(dataset_category=dataset_category, dataset_split=dataset_split)
        
        else: 
            raise Exception(f"unknown dataset_type given: {self.dataset_name}")

    def get_queries(self, dataset_category="lifestyle", dataset_split="dev"):
        """
        Help get queries as a Dict.
        
        Args: 
            dataset_type : train or test type
            dataset_category : sub-category for Lotte includes, ('all', 'lifestyle', ..)
            
            
        Return:
            Dict, of queries
            
        Example : 
            {
                1: {"query": "query 1..", "rel_doc_ids": [1, 2, 3]},
                2: {"query": "query 2..", "rel_doc_ids": [1, 5, 3]},
            }
        """
        if self.dataset_name == "lotte":
            return self.lotte_queries(dataset_category=dataset_category, dataset_split=dataset_split)
        
        else: 
            raise Exception(f"unknown dataset_type given: {self.dataset_name}")
