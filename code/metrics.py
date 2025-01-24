import numpy as np
import pandas as pd

import world


def calculate_precision(top_items, true_items, k):
    return len(set(top_items[:k]).intersection(true_items)) / k


def calculate_recall(top_items, true_items):
    return len(set(top_items).intersection(true_items)) / len(true_items) if true_items else 0


def calculate_ndcg(top_items, true_items, k):
    dcg_score = sum(1 / np.log2(idx + 2) for idx, item in enumerate(top_items[:k]) if item in true_items)
    idcg_score = sum(1 / np.log2(idx + 2) for idx in range(min(len(true_items), k)))
    return dcg_score / idcg_score if idcg_score > 0 else 0


def calculate_map(top_items, true_items, k):
    ap = 0
    correct = 0
    for idx, item in enumerate(top_items[:k]):
        if item in true_items:
            correct += 1
            ap += correct / (idx + 1)
    return ap / min(k, len(true_items)) if true_items else 0


def evaluate_metrics(rating_K, label, k_list, epoch):

    results = {}
    label_set = [set(l) for l in label]
    reciprocal_ranks = []

    for i, ranked_items in enumerate(rating_K):
        true_items = label_set[i]
        rank = next((idx + 1 for idx, item in enumerate(ranked_items) if item in true_items), 0)
        reciprocal_ranks.append(1 / rank if rank > 0 else 0)

    metrics_data = []

    for k in k_list:
        if k > rating_K.shape[1]:
            raise ValueError(f"k={k} lớn hơn số phần tử trong rating_K")

        top_k = rating_K[:, :k]
        precision, recall, ndcg, map_score, hit_rate = [], [], [], [], []

        for i, top_items in enumerate(top_k):
            true_items = label_set[i]
            precision.append(calculate_precision(top_items, true_items, k))
            recall.append(calculate_recall(top_items, true_items))
            ndcg.append(calculate_ndcg(top_items, true_items, k))
            map_score.append(calculate_map(top_items, true_items, k))
            hit_rate.append(1 if len(set(top_items).intersection(true_items)) > 0 else 0)

        results[k] = {
            "Precision@K": np.mean(precision),
            "Recall@K": np.mean(recall),
            "HitRate@K": np.mean(hit_rate),
            "NDCG@K": np.mean(ndcg),
            "MAP@K": np.mean(map_score),
        }
        metrics_data.append(
            [k, np.mean(precision), np.mean(recall), np.mean(hit_rate), np.mean(ndcg), np.mean(map_score)])

    mrr = np.mean(reciprocal_ranks)
    results["MRR"] = mrr

    columns = ["K", "Precision@K", "Recall@K", "HitRate@K", "NDCG@K", "MAP@K"]
    df = pd.DataFrame(metrics_data, columns=columns)
    print(df)
    print(f"\nMean Reciprocal Rank (MRR): {mrr:.4f}")

    epoch_data = {"Epoch": epoch, "MRR": mrr}
    for k in k_list:
        epoch_data[f"Precision@{k}"] = results[k]["Precision@K"]
        epoch_data[f"Recall@{k}"] = results[k]["Recall@K"]
        epoch_data[f"HitRate@{k}"] = results[k]["HitRate@K"]
        epoch_data[f"NDCG@{k}"] = results[k]["NDCG@K"]
        epoch_data[f"MAP@{k}"] = results[k]["MAP@K"]

