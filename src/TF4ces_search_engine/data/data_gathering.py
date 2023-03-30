#!/usr/bin/env python3
# encoding: utf-8

"""
    File to read dataset

    Author : TF4ces
"""


import ir_datasets

class DataGathering():

    def __init__(self, dataset_type="lotte", base_dataset_dir="search"):
        '''base_dataset_dir : forum or search'''
        self.dataset_type = dataset_type
        self.base_dataset_dir = base_dataset_dir

    
    def lotte_documents(self,dataset_category="lifestyle",dataset_split="dev"):
        # raise NotImplementedError
        # TODO return dict
        # if self.base_dataset_dir is None:
        #     path = self.dataset_type+'/'+dataset_category+'/'+dataset_split
        # else:
        #     path = self.dataset_type+'/'+dataset_category+'/'+dataset_split+'/'+self.base_dataset_dir

        path = self.dataset_type + '/' + dataset_category + '/' + dataset_split + '/' + self.base_dataset_dir
        data = ir_datasets.load(path)
        documents = {}
        for doc_id, doc in data.docs_iter():
            documents[doc_id] = doc

        return documents

        
    def lotte_queries(self,dataset_category="lifestyle",dataset_split="dev"):
        # raise NotImplementedError
        # TODO return dict
        # if self.base_dataset_dir==None:
        #     # print("Error: Base Directory not mentioned")
        #     break


        path = self.dataset_type+'/'+dataset_category+'/'+dataset_split+'/'+self.base_dataset_dir

        data = ir_datasets.load(path)

        queries = {}
        for q_id, query in data.queries_iter():
            queries[q_id] = query

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
                2: "document 2",
            }
        """
        
        if self.dataset_type == "lotte":
            return self.lotte_documents(dataset_category="lifestyle", dataset_split="dev")
        
        else: 
            raise Exception(f"unknown dataset_type given: {self.dataset_type}")

    def get_queries(self, dataset_category="lifestyle", dataset_split="dev"):
        """
        Help get queries as a Dict.
        
        Args: 
            dataset_type : train or test type
            dataset_category : sub-category for Lotte includes, ('all', 'lifestyle', ..)
            
            
        Return:
            Dict, TODO
            
        Example : 
            {
                1: {"query": "query 1..", "doc_ids": [1, 2, 3]},
                2: {"query": "query 2..", "doc_ids": [1, 5, 3]},
            }
        """
        if self.dataset_type == "lotte":
            return self.lotte_queries(dataset_category="lifestyle", dataset_split="dev")
        
        else: 
            raise Exception(f"unknown dataset_type given: {self.dataset_type}")
