"""
Shared evaluation utils for all baseline experiments.
Provides data loading, retrieval metric computation, and result saving.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_data(preprocessed_path: str):
    """
    Load patents, products, mappings and build the positive map.
    Returns patents, products, all_patent_nums, all_product_ids, positive_map.
    """
    print(f"Loading data from {preprocessed_path}...")
    with open(preprocessed_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    patents = data["patents"]
    products = data["products"]
    mappings = data["mappings"]
    all_patent_nums = list(patents.keys())
    all_product_ids = list(products.keys())

    positive_map: Dict[str, List[str]] = {}
    for m in mappings:
        positive_map.setdefault(m["patent_number"], []).append(m["product_id"])

    avg_pairs = len(mappings) / max(len(all_patent_nums), 1)
    print(f"Patents: {len(all_patent_nums)}")
    print(f"Products: {len(all_product_ids)}")
    print(f"Mappings: {len(mappings)} (avg {avg_pairs:.2f} pairs/patent)")

    return patents, products, all_patent_nums, all_product_ids, positive_map


def compute_retrieval_metrics(
    query_patent_nums: List[str],
    all_scores: np.ndarray,
    corpus_product_ids: List[str],
    positive_map: Dict[str, List[str]],
    k_values: List[int] = [5, 10, 20],
) -> Dict:
    """Compute MRR, MAP@k and nDCG@k for a set of queries against a corpus."""
    mrrs: List[float] = []
    maps = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}

    for i, patent_num in enumerate(query_patent_nums):
        pos_ids = positive_map.get(patent_num, [])
        if not pos_ids:
            continue

        row = all_scores[i]
        ranked_ids = [corpus_product_ids[j] for j in np.argsort(row)[::-1]]
        relevant_set = set(pos_ids)

        # MRR
        mrr = 0.0
        for rank, item_id in enumerate(ranked_ids, 1):
            if item_id in relevant_set:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)

        for k in k_values:
            top_k = ranked_ids[:k]

            # MAP@k
            precisions = []
            num_relevant = 0
            for rank, item_id in enumerate(top_k, 1):
                if item_id in relevant_set:
                    num_relevant += 1
                    precisions.append(num_relevant / rank)
            maps[k].append(
                sum(precisions) / min(len(relevant_set), k) if relevant_set else 0.0
            )

            # nDCG@k
            dcg = sum(
                1.0 / np.log2(rank + 1)
                for rank, item_id in enumerate(top_k, 1)
                if item_id in relevant_set
            )
            idcg = sum(
                1.0 / np.log2(rank + 1)
                for rank in range(1, min(k, len(relevant_set)) + 1)
            )
            ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)

    metrics: Dict = {}
    if mrrs:
        metrics["mrr"] = float(np.mean(mrrs))
        for k in k_values:
            metrics[f"map@{k}"] = float(np.mean(maps[k]))
            metrics[f"ndcg@{k}"] = float(np.mean(ndcgs[k]))
    return metrics


def save_results(results: Dict, output_path: Path) -> None:
    """Save results dict as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


def print_metrics(metrics: Dict, k_values: List[int] = [5, 10, 20]) -> None:
    """Print metrics in a compact format."""
    print(
        f"MRR={metrics.get('mrr', 0):.4f}  "
        + "  ".join(f"nDCG@{k}={metrics.get(f'ndcg@{k}', 0):.4f}" for k in k_values)
    )
