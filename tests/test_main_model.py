from src.TF4ces_search_engine.data.data_gathering import DataGathering
from src.TF4ces_search_engine.feature.data_preprocessing import Preprocessing
from src.TF4ces_search_engine.model.tf_idf import Tfidf
from src.TF4ces_search_engine.model.bm25 import BM25

import itertools

if __name__ == '__main__':

    model = "bm25"

    data_gathering = DataGathering(dataset_name="lotte",)

    docs = data_gathering.get_documents(dataset_category="lifestyle", dataset_split="dev")
    queries = data_gathering.get_queries(dataset_category="lifestyle", dataset_split="dev")

    docs_N = dict(itertools.islice(docs.items(), 100))
    queries_N = dict(itertools.islice(queries.items(), 5))

    dp = Preprocessing()

    print("Preprocessing in Progress...")
    docs_N, queries_N = dp.pre_process(docs_N, queries_N, model)

    top_n = 5  # Retrieve top 5 documents for each query.
    print("Retrieving top 5 documents...")

    if model == "tfidf":
        tfidf = Tfidf()
        query_keys, retrieved_id = tfidf.retrieve_documents(docs_N, queries_N, top_n)

    if model == "bm25":
        bm25 = BM25()
        query_keys, retrieved_id = bm25.retrieve_documents(docs_N, queries_N, top_n)

    original_id = []
    for key in queries_N:
        for inner_key in queries[key]:
            if inner_key == "rel_doc_ids":
                original_id.append(queries[key][inner_key])

    print("Original documents", original_id)
    print("Retrieved documents", retrieved_id)








