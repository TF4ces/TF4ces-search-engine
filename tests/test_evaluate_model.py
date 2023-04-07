#!/usr/bin/env python3
# encoding: utf-8

"""
    Main test script to test and eval model.

    Author : TF4ces
"""


# Native imports
import sys
sys.path.append('/home/jovyan/teaching_material/IR/TF4ces-search-engine/')

# Third-party imports

# User imports
from config.conf import __WORKSPACE__, __ALL_MODELS__
from src.main import TF4cesFlow


if __name__ == '__main__':

    TOP_N = 10_000  # Retrieve top N documents for each query.
    K = 1_000

    VERSION = 'v0.0.1'  #v0.0.1-small
    TEST_RUN = False

    # Pre-processing configs
    USE_CACHE = True
    PREPROCESS_CACHE_DIR = __WORKSPACE__ / "dataset" / "preprocessed" / f"test_{VERSION}"  # pre processed data is stored here.

    # Model configs
    MODEL = __ALL_MODELS__[0] # tfidf, bm25, all-mpnet-base-v2, all-roberta-large-v1, Intel/ColBERT-NQ
    MODEL_PATH = __WORKSPACE__ / "models"

    print(f"Model Selected : {MODEL}")

    pipeline = TF4cesFlow(
        model_name=MODEL,
        dataset_name="lotte",
        dataset_category="lifestyle",
        top_n=TOP_N,
        k=K,
        use_cache=USE_CACHE,
        version=VERSION,
        preprocess_cache_dir=PREPROCESS_CACHE_DIR,
        model_path=MODEL_PATH,
    )

    # split = 'dev'
    # pipeline.gather_data(split=split)
    # if TEST_RUN: pipeline.small_test(split=split)  # DEBUG ONLY
    # pipeline.data_preprocessing(split=split)
    # pipeline.retrieval(split=split, bl_train=True)

    # Test
    split = 'test'
    pipeline.gather_data(split=split)
    if TEST_RUN: pipeline.small_test(split=split)  # DEBUG ONLY
    pipeline.data_preprocessing(split=split)
    pipeline.retrieval(split=split, bl_train=False)

    print("DONE")
