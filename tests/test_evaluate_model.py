from src.TF4ces_search_engine.data.data_gathering import DataGathering
from src.TF4ces_search_engine.feature.data_preprocessing import Preprocessing
from src.TF4ces_search_engine.model.tf_idf import Tfidf
from src.TF4ces_search_engine.model.bm25 import BM25

import itertools
import numpy as np

if __name__ == '__main__':

    model = "bm25"

    data_gathering = DataGathering(dataset_name="lotte",)

    docs = data_gathering.get_documents(dataset_category="lifestyle", dataset_split="dev")
    queries = data_gathering.get_queries(dataset_category="lifestyle", dataset_split="dev")

    dp = Preprocessing()

    print("Preprocessing in Progress...")
    docs, queries = dp.pre_process(docs, queries, model)

    top_n = 5  # Retrieve top 5 documents for each query.
    print("TFIDF: Retrieving top 5 documents...")

    if model == "tfidf":
        tfidf = Tfidf()
        query_keys, retrieved_id = tfidf.retrieve_documents(docs, queries, top_n)

    if model == "bm25":
        bm25 = BM25()
        query_keys, retrieved_id = bm25.retrieve_documents(docs, queries, top_n)

    original_id = []
    for key in queries:
        for inner_key in queries[key]:
            if inner_key == "rel_doc_ids":
                original_id.append(queries[key][inner_key])

    # Calculate performance for retrieved documents
    original_n = len(original_id)
    retrieved_n = len(retrieved_id)

    # Create an empty confusion matrix
    confusion_matrix = np.zeros((original_n, retrieved_n))

    # Populate the confusion matrix
    for i in range(original_n):
        for j in range(retrieved_n):
            intersection = set(original_id[i]).intersection(set(retrieved_id[j]))
            if intersection:
                confusion_matrix[i, j] = len(intersection)/len(set(original_id[i]))
            else:
                confusion_matrix[i, j] = 0

    # Print the confusion matrix
    TP = np.diag(confusion_matrix)
    FP = np.sum(confusion_matrix, axis=0) - TP
    FN = np.sum(confusion_matrix, axis=1) - TP
    TN = np.sum(confusion_matrix) - (TP + FP + FN)

    # Calculate precision, recall, and F1-score

    def precision(TP, FP):
        result = []
        for i in range(len(TP)):
            if TP[i] + FP[i] == 0:
                result.append(0)
            else:
                result.append(TP[i] / (TP[i] + FP[i]))
        return result


    def recall(TP, FN):
        result = []
        for i in range(len(TP)):
            if TP[i] + FN[i] == 0:
                result.append(0)
            else:
                result.append(TP[i] / (TP[i] + FN[i]))
        return result


    def f1_score(precision_vals, recall_vals):
        result = []
        for i in range(len(precision_vals)):
            precision_val = precision_vals[i]
            recall_val = recall_vals[i]
            if precision_val + recall_val == 0:
                result.append(0)
            else:
                result.append(2 * precision_val * recall_val / (precision_val + recall_val))
        return result

    precision_vals = precision(TP, FP)
    recall_vals = recall(TP, FN)
    f1_vals = f1_score(precision_vals, recall_vals)

    # Print the results
    print("Precision:", precision_vals)
    print("Recall:", recall_vals)
    print("F1-score:", f1_vals)
    print()
    print("Average Precision:", np.average(precision_vals))
    print("Average Recall:", np.average(recall_vals))
    print("Average F1-score:", np.average(f1_vals))










