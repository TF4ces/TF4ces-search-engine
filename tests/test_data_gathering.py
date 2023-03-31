#!/usr/bin/env python3
# encoding: utf-8

"""
    Test file for data_gathering.py

    Author : TF4ces
"""

from config.conf import __WORKSPACE__
from src.TF4ces_search_engine.data.data_gathering import DataGathering



if __name__ == '__main__':
    print("Test Start")

    base_dataset_dir = __WORKSPACE__ / "dataset"

    data_gathering = DataGathering(
        dataset_name="lotte",
    )

    docs = data_gathering.get_documents(dataset_category="lifestyle", dataset_split="dev")
    queries = data_gathering.get_queries(dataset_category="lifestyle", dataset_split="dev")
    print(f"{len(docs)=}")
    print(f"{len(queries)=}")

    print(f"Done")