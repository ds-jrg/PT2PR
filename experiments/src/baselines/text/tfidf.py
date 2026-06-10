"""
TF-IDF cosine similarity retrieval baseline for patent-product matching.
Vocabulary is fitted on the full product corpus so IDF reflects the retrieval setting.
"""

from pathlib import Path
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from experiments.src.baselines.utils import get_patent_text, get_product_text
from experiments.src.baselines.metrics import (
    load_data,
    compute_retrieval_metrics,
    save_results,
    print_metrics,
)


def run(
    preprocessed_path: str,
    output_dir: str,
    masked: bool = True,
    k_values: List[int] = [5, 10, 20],
) -> Dict:
    """TF-IDF cosine similarity retrieval baseline."""
    print("\nEXPERIMENT: TF-IDF")
    print(f"Masked: {masked}")

    patents, products, all_patent_nums, all_product_ids, positive_map = load_data(
        preprocessed_path
    )

    # Encode full product corpus; IDF fitted here reflects retrieval setting
    all_product_texts = [
        get_product_text(products[pid], masked=masked) for pid in all_product_ids
    ]
    print(f"Fitting TF-IDF on {len(all_product_ids)} products...")
    vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 1),
    )
    vectorizer.fit(all_product_texts)
    product_matrix = vectorizer.transform(all_product_texts)

    # Score all patents against the full product corpus
    print(
        f"Scoring {len(all_patent_nums)} patents against {len(all_product_ids)} products..."
    )
    patent_texts = [get_patent_text(patents[p], masked=masked) for p in all_patent_nums]
    query_matrix = vectorizer.transform(patent_texts)
    scores = cosine_similarity(query_matrix, product_matrix)

    metrics = compute_retrieval_metrics(
        query_patent_nums=all_patent_nums,
        all_scores=scores,
        corpus_product_ids=all_product_ids,
        positive_map=positive_map,
        k_values=k_values,
    )
    print_metrics(metrics, k_values)

    results = {
        "config": {"method": "tfidf", "masked": masked},
        "metrics": metrics,
    }
    save_results(results, Path(output_dir) / "tfidf.json")
    return results
