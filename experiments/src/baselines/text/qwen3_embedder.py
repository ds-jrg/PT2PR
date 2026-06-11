"""
Qwen3-Embedding dense retrieval baseline for patent-product matching.
"""

from pathlib import Path
from typing import Dict, List
from sentence_transformers import SentenceTransformer
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
    model_name: str = "Qwen/Qwen3-Embedding-8B",
    masked: bool = True,
    batch_size: int = 1,
    k_values: List[int] = [5, 10, 20],
) -> Dict:
    """Qwen3-Embedding dense retrieval baseline."""
    print("\nEXPERIMENT: QWEN3-EMBEDDING")
    print(f"Model: {model_name}  Masked: {masked}")

    patents, products, all_patent_nums, all_product_ids, positive_map = load_data(
        preprocessed_path
    )

    print(f"Loading model {model_name}...")
    model = SentenceTransformer(
        model_name, device="cuda", tokenizer_kwargs={"padding_side": "left"}
    )

    # Encode full product corpus as passages (no prompt)
    print(f"Encoding {len(all_product_ids)} products...")
    product_texts = [
        get_product_text(products[pid], masked=masked) for pid in all_product_ids
    ]
    product_embeddings = model.encode(
        product_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        prompt="",
    )

    # Encode all patents as queries with task-specific prompt
    print(f"Encoding {len(all_patent_nums)} patents...")
    patent_texts = [get_patent_text(patents[p], masked=masked) for p in all_patent_nums]
    patent_embeddings = model.encode(
        patent_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        prompt="Retrieve relevant products for the given patent.",
    )

    # Cosine similarity: embeddings are L2-normalised so dot product is sufficient
    print("Computing similarity scores...")
    scores = patent_embeddings @ product_embeddings.T

    metrics = compute_retrieval_metrics(
        query_patent_nums=all_patent_nums,
        all_scores=scores,
        corpus_product_ids=all_product_ids,
        positive_map=positive_map,
        k_values=k_values,
    )
    print_metrics(metrics, k_values)

    model_slug = model_name.split("/")[-1].lower().replace("-", "_")
    results = {
        "config": {"method": model_slug, "model": model_name, "masked": masked},
        "metrics": metrics,
    }
    save_results(results, Path(output_dir) / f"{model_slug}.json")
    return results
