#!/usr/bin/env python3
# encoding: utf-8

"""
    Main test script to test and eval model.

    Author : TF4ces
"""

# Native imports
import sys
sys.path.append('/home/jovyan/teaching_material/IR/TF4ces-search-engine/') #Change the path if running on jupyter hub
import pickle

# Third-party imports
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers.util import cos_sim

# User imports
from config.conf import __WORKSPACE__, __ALL_MODELS__
from src.main_ens import Ensemble
from src.utils.evalutor import mean_recall_K, mean_precision_K

if __name__ == '__main__':
    
    VERSION = 'v0.0.1'  # v0.0.1-small
    TEST_RUN = True

    # Pre-processing configs
    USE_CACHE = True
    PREPROCESS_CACHE_DIR = __WORKSPACE__ / "dataset" / "preprocessed" / f"test_{VERSION}"  # pre processed data is stored here.
     
    # Model configs
    MODEL_PATH = __WORKSPACE__ / "models"
    K=5
    
    MODEL_0 = __ALL_MODELS__[0]
    MODEL_1 = __ALL_MODELS__[2]  # tfidf, bm25, all-mpnet-base-v2, all-roberta-large-v1, Intel/ColBERT-NQ
    MODEL_2 = __ALL_MODELS__[3]  # tfidf, bm25, all-mpnet-base-v2, all-roberta-large-v1, Intel/ColBERT-NQ
   
    print(f"Base Model Selected : {MODEL_0}")
    print(f"Voter Models : {MODEL_1, MODEL_2}")
    
    TOP_N = 10000  # Retrieve top N documents for each query.
    SAMPLE_QUERIES = 3
    
    ###################################################### MODEl 0 ##################################################################
    pipeline = Ensemble(
        model_name=MODEL_0,
        dataset_name="lotte",
        dataset_category="lifestyle",
        top_n=TOP_N,
        k=K,
        use_cache=USE_CACHE,
        version=VERSION,
        preprocess_cache_dir=PREPROCESS_CACHE_DIR,
        model_path=MODEL_PATH,
    )

    split = 'dev'
    pipeline.base_gather_data(split=split)
    if TEST_RUN: pipeline.small_test(split=split, nqueries=SAMPLE_QUERIES)  # DEBUG ONLY
    pipeline.data_preprocessing(split=split, use_cache=True, cache_sub_dir=None)
    pipeline.query_preporcessing(split='dev', use_cache=False, cache_sub_dir='sub_dir')
    q_ids_d0, gold_doc_ids_d0, pred_doc_ids_d0 = pipeline.base_retrieval(split=split, bl_train=False)
    
    # # Test
    # split = 'test'
    # pipeline.gather_data(split=split)
    # if TEST_RUN: pipeline.small_test(split=split)  # DEBUG ONLY
    # pipeline.data_preprocessing(split=split)
    # q_ids_t0, gold_doc_ids_t0, pred_doc_ids_t0 = pipeline.retrieval(split=split, bl_train=False)
    
    
    TOP_N = 1000  # Retrieve top N documents for each query. 
    TEST_RUN = False
    ###################################################### MODEl 1 ##################################################################

    pipeline = Ensemble(
        model_name=MODEL_1,
        dataset_name="lotte",
        dataset_category="lifestyle",
        top_n=TOP_N,
        k=K,
        use_cache=USE_CACHE,
        version=VERSION,
        preprocess_cache_dir=PREPROCESS_CACHE_DIR,
        model_path=MODEL_PATH,
    )

    # split = 'test'
    q_ids_d1 = []
    gold_doc_ids_d1 = []
    pred_doc_ids_d1 = []
    pipeline.get_model(bl_train=True)
    pipeline.model.debug = False
    for i in tqdm(range(len(q_ids_d0))):
        pipeline.gather_data(query_ids=q_ids_d0[i], doc_ids=pred_doc_ids_d0[i], split=split)
        if TEST_RUN: pipeline.small_test(split=split)  # DEBUG ONLY
        #pipeline.data_preprocessing(split=split, use_cache=False, cache_sub_dir="subset")
        q_ids_d1_, gold_doc_ids_d1_, pred_doc_ids_d1_ = pipeline.retrieval(split=split, bl_train=True)
        q_ids_d1.extend(q_ids_d1_)
        gold_doc_ids_d1.extend(gold_doc_ids_d1_)
        pred_doc_ids_d1.extend(pred_doc_ids_d1_)
    

    # # Test
    # split = 'test'
    # pipeline.gather_data(split=split)
    # if TEST_RUN: pipeline.small_test(split=split)  # DEBUG ONLY
    # pipeline.data_preprocessing(split=split)
    # q_ids_t1, gold_doc_ids_t1, pred_doc_ids_t1 = pipeline.retrieval(split=split, bl_train=False)
    
    
    ###################################################### MODEl 2 ##################################################################
    
    pipeline = Ensemble(
        model_name=MODEL_2,
        dataset_name="lotte",
        dataset_category="lifestyle",
        top_n=TOP_N,
        k=K,
        use_cache=USE_CACHE,
        version=VERSION,
        preprocess_cache_dir=PREPROCESS_CACHE_DIR,
        model_path=MODEL_PATH,
    )

    # split = 'test'
    q_ids_d2 = []
    gold_doc_ids_d2 = []
    pred_doc_ids_d2 = []
    pipeline.get_model(bl_train=True)
    pipeline.model.debug = False
    for i in tqdm(range(len(q_ids_d0))):
        pipeline.gather_data(query_ids=q_ids_d0[i], doc_ids=pred_doc_ids_d0[i], split=split)
        if TEST_RUN: pipeline.small_test(split=split)  # DEBUG ONLY
        #pipeline.data_preprocessing(split=split, use_cache=False, cache_sub_dir="subset")
        q_ids_d2_, gold_doc_ids_d2_, pred_doc_ids_d2_ = pipeline.retrieval(split=split, bl_train=True)
        q_ids_d2.extend(q_ids_d2_)
        gold_doc_ids_d2.extend(gold_doc_ids_d2_)
        pred_doc_ids_d2.extend(pred_doc_ids_d2_)
    

#     # Test
#     split = 'test'
#     pipeline.gather_data(split=split)
#     if TEST_RUN: pipeline.small_test(split=split)  # DEBUG ONLY
#     pipeline.data_preprocessing(split=split)
#     q_ids_t2, gold_doc_ids_t2, pred_doc_ids_t2 = pipeline.retrieval(split=split, bl_train=False)

    
    ###################################################### EVALUATION ##################################################################
    K = 10
        
    # print(f"{pred_doc_ids_d1=}")
    # print(f"{pred_doc_ids_d2=}")
    
    def priority_intersection_m1(pred_doc_ids_d1, pred_doc_ids_d2):
        return [[i for i in sublist_a if i in sublist_b] for sublist_a, sublist_b in zip(pred_doc_ids_d1, pred_doc_ids_d2)]
    
    def top_sorted_rank_order(pred_doc_ids_d1, pred_doc_ids_d2):
        print(f"{np.array(pred_doc_ids_d1).shape}")
        print(f"{np.array(pred_doc_ids_d2).shape}")
        dev_intersection = []
        for pred_doc_ids_d1_, pred_doc_ids_d2_ in zip(pred_doc_ids_d1, pred_doc_ids_d2):
            tmp_preds = []
            for pred_doc_id_d1_, pred_doc_id_d2_ in zip(pred_doc_ids_d1_, pred_doc_ids_d2_):
                if pred_doc_id_d1_ not in tmp_preds:
                    tmp_preds.append(pred_doc_id_d1_)

                if pred_doc_id_d2_ not in tmp_preds:
                    tmp_preds.append(pred_doc_id_d2_)

                if len(tmp_preds) >= K:
                    break
            dev_intersection.append(tmp_preds)
            
        print(f"{np.array(dev_intersection).shape=}")
        return dev_intersection
    
    dev_intersection = priority_intersection_m1(pred_doc_ids_d1, pred_doc_ids_d2)
    #dev_intersection = top_sorted_rank_order(pred_doc_ids_d1, pred_doc_ids_d2)
            
    # print(f"{dev_intersection=}")
    # print(f"{gold_doc_ids_d1=}")
    # print(f"{gold_doc_ids_d2=}")
    
    
    #test_intersection = [[i for i in sublist_a if i in sublist_b] for sublist_a, sublist_b in zip(pred_doc_ids_t1, pred_doc_ids_t2)]
    
    # for sublist in dev_intersection:
    #     while len(sublist) < (TOP_N/2):
    #         sublist.append(0)
    # for sublist in test_intersection:
    #     while len(sublist) < (TOP_N/2):
    #         sublist.append(0)
    
    #dev_intersection = np.array(dev_intersection, dtype=object)
    #test_intersection = np.array(test_intersection)
    
    print()
    print(f"Dev Evaluation..")
    recall_k = mean_recall_K(golds=gold_doc_ids_d1, preds=dev_intersection, k=K)
    precision_k = mean_precision_K(golds=gold_doc_ids_d1, preds=dev_intersection, k=K)
    print(f"Recall@{K} : {recall_k}")
    print(f"Precision@{K} : {precision_k}")
    
    # print()
    # print(f"Test Evaluation..")
    # recall_k = mean_recall_K(golds=gold_doc_ids_t1, preds=test_intersection, k=K)
    # precision_k = mean_precision_K(golds=gold_doc_ids_t1, preds=test_intersection, k=K)
    # print(f"Recall@{K} : {recall_k}")
    # print(f"Precision@{K} : {precision_k}")


    
#     pred = {}
#     for query_id, gold_doc_id, rel_doc_id in zip(q_ids_1, gold_doc_ids_1, pred_doc_ids_1):
#         pred[query_id] = {"gold":gold_doc_id, "pred":rel_doc_id}
#     PREDICTIONS_CACHE_DIR = __WORKSPACE__ / "predictions" / f"test_{VERSION}" / MODEL / "lotte" / "lifestyle" / split  
#     predictions_doc_file = PREDICTIONS_CACHE_DIR / str("pred_docs." + split + ".pkl")
#     if not PREDICTIONS_CACHE_DIR.exists(): PREDICTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
#     pickle.dump(pred, open(predictions_doc_file, 'wb'))
    
#     pred = {}
#     for query_id, gold_doc_id, rel_doc_id in zip(q_ids_1, gold_doc_ids_1, pred_doc_ids_1):
#         pred[query_id] = {"gold":gold_doc_id, "pred":rel_doc_id}
#     PREDICTIONS_CACHE_DIR = __WORKSPACE__ / "predictions" / f"test_{VERSION}" / MODEL / "lotte" / "lifestyle" / split  
#     predictions_doc_file = PREDICTIONS_CACHE_DIR / str("pred_docs." + split + ".pkl")
#     if not PREDICTIONS_CACHE_DIR.exists(): PREDICTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
#     pickle.dump(pred, open(predictions_doc_file, 'wb'))