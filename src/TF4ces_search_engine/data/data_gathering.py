#!/usr/bin/env python3
# encoding: utf-8

"""
    File to read dataset

    Author : TF4ces
"""




class DataGathering:
    
    def __init__(self, dataset_type="lotte", base_dataset_dir=None):
        self.dataset_type = dataset_type
        self.base_dataset_dir = base_dataset_dir
        
    
    def lotte_documents(self, dataset_split="train", dataset_category=None):
        raise NotImplementedError
        #TODO return dict
        
    def lotte_queries(self, dataset_split="train", dataset_category=None):
        raise NotImplementedError
        #TODO return dict
        
    def get_documents(self, dataset_split="train", dataset_category=None):
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
            return self.lotte_documents(dataset_split="train", dataset_category=None)
        
        else: 
            raise Exception (f"unknown dataset_type given: {self.dataset_type}")

    def get_queries(self, dataset_split="train", dataset_category=None):
        """
        Help get queries as a Dict.
        
        Args: 
            dataset_type : train or test type
            dataset_category : sub-category for Lotte includes, ('all', 'lifestyle', ..)
            
            
        Return:
            Dict, TODO
            
        Example : 
            {
                1: {"querry": "querry 1..", "doc_ids": [1, 2, 3]},
                2: {"querry": "querry 2..", "doc_ids": [1, 5, 3]},
            }
        """
        if self.dataset_type == "lotte":
            return self.lotte_queries(dataset_split="train", dataset_category=None)
        
        else: 
            raise Exception (f"unknown dataset_type given: {self.dataset_type}")
        
        
