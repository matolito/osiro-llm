import numpy as np


def precision_at_k(recommended_items: list, relevant_items: list, k: int) -> float:
    """
    Computes Precision@k.
    """
    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    true_positives = len(set(recommended_k) & relevant_set)
    return true_positives / k


def recall_at_k(recommended_items: list, relevant_items: list, k: int) -> float:
    """
    Computes Recall@k.
    """
    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    true_positives = len(set(recommended_k) & relevant_set)
    return true_positives / len(relevant_set) if relevant_set else 0


def dcg_at_k(recommended_items: list, relevant_items: list, k: int) -> float:
    """
    Computes Discounted Cumulative Gain (DCG@k).
    """
    recommended_k = recommended_items[:k]
    relevance = [1 if item in set(relevant_items) else 0 for item in recommended_k]
    return np.sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance)])


def ndcg_at_k(recommended_items: list, relevant_items: list, k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (nDCG@k).
    """
    dcg = dcg_at_k(recommended_items, relevant_items, k)
    ideal_dcg = dcg_at_k(list(relevant_items), relevant_items, k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0
