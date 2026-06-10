"""
BM25Okapi retrieval baseline for patent-product matching.
Index is built once over the full product corpus.
"""

import string
import numpy as np
from pathlib import Path
from typing import Dict, List
from rank_bm25 import BM25Okapi
from experiments.src.baselines.utils import get_patent_text, get_product_text
from experiments.src.baselines.metrics import (
    load_data,
    compute_retrieval_metrics,
    save_results,
    print_metrics,
)


# Tokenizer (TF-IDF uses its own built-in)
_PUNCT = str.maketrans("", "", string.punctuation)


def _tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    return text.lower().translate(_PUNCT).split()


def run(
    preprocessed_path: str,
    output_dir: str,
    masked: bool = True,
    k_values: List[int] = [5, 10, 20],
) -> Dict:
    """BM25Okapi retrieval baseline."""
    print("\nEXPERIMENT: BM25")
    print(f"Masked: {masked}")

    patents, products, all_patent_nums, all_product_ids, positive_map = load_data(
        preprocessed_path
    )

    # Build BM25 index once over the full product corpus
    print(f"Building BM25 index over {len(all_product_ids)} products...")
    product_token_lists = [
        _tokenize(get_product_text(products[pid], masked=masked))
        for pid in all_product_ids
    ]
    bm25 = BM25Okapi(product_token_lists)

    # Score all patents against the full product corpus
    print(
        f"Scoring {len(all_patent_nums)} patents against {len(all_product_ids)} products..."
    )
    all_scores = np.zeros(
        (len(all_patent_nums), len(all_product_ids)), dtype=np.float32
    )
    for i, pnum in enumerate(all_patent_nums):
        query_tokens = _tokenize(get_patent_text(patents[pnum], masked=masked))
        all_scores[i] = bm25.get_scores(query_tokens)

    metrics = compute_retrieval_metrics(
        query_patent_nums=all_patent_nums,
        all_scores=all_scores,
        corpus_product_ids=all_product_ids,
        positive_map=positive_map,
        k_values=k_values,
    )
    print_metrics(metrics, k_values)

    results = {
        "config": {"method": "bm25", "masked": masked},
        "metrics": metrics,
    }
    save_results(results, Path(output_dir) / "bm25.json")
    return results
