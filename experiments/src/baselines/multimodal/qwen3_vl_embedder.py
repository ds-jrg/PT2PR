"""
Qwen3-VL-Embedding retrieval baseline for multimodal patent-product matching.
"""

from pathlib import Path
from typing import Dict, List, Literal
from sentence_transformers import SentenceTransformer
from experiments.src.baselines.utils import (
    get_patent_text,
    get_product_text,
    get_patent_image_path,
    get_product_image_path,
)
from experiments.src.baselines.metrics import (
    load_data,
    compute_retrieval_metrics,
    save_results,
    print_metrics,
)


def run(
    preprocessed_path: str,
    output_dir: str,
    visual_assets_dir: str,
    mode: Literal["text", "image", "multimodal"] = "multimodal",
    model_name: str = "Qwen/Qwen3-VL-Embedding-8B",
    masked: bool = True,
    batch_size: int = 1,
    k_values: List[int] = [5, 10, 20],
) -> Dict:
    """Qwen3-VL-Embedding retrieval baseline."""
    print(f"\nEXPERIMENT: QWEN3-VL ({mode})")
    print(f"Model: {model_name}  Masked: {masked}")

    patents, products, all_patent_nums, all_product_ids, positive_map = load_data(
        preprocessed_path
    )
    assets_dir = Path(visual_assets_dir)

    print(f"Loading model {model_name}...")
    model = SentenceTransformer(
        model_name, device="cuda", tokenizer_kwargs={"padding_side": "left"}
    )

    def _build_inputs(entity_ids, get_text_fn, get_image_path_fn, label: str):
        """
        Build the input list for a set of entities.
        - text mode: plain text string
        - image mode: image path string
        - multimodal mode: {"text": ..., "image": ...} dict
        """
        inputs = []
        for eid in entity_ids:
            if mode == "text":
                inputs.append(get_text_fn(eid))
            else:
                path = get_image_path_fn(eid)
                if path is None:
                    raise RuntimeError(
                        f"Image not found for {label} entity '{eid}'. "
                        f"The multimodal dataset should guarantee visual assets "
                        f"for all pairs; check that the visual assets directory "
                        f"is accessible and matches the path used during preprocessing."
                    )
                if mode == "image":
                    inputs.append(str(path))
                else:  # multimodal
                    inputs.append({"text": get_text_fn(eid), "image": str(path)})
        return inputs

    print(f"Building patent inputs ({mode})...")
    patent_inputs = _build_inputs(
        all_patent_nums,
        lambda p: get_patent_text(patents[p], masked=masked),
        lambda p: get_patent_image_path(p, assets_dir),
        "patents",
    )

    print(f"Building product inputs ({mode})...")
    product_inputs = _build_inputs(
        all_product_ids,
        lambda pid: get_product_text(products[pid], masked=masked),
        lambda pid: get_product_image_path(pid, assets_dir),
        "products",
    )

    print(f"Encoding {len(patent_inputs)} patents...")
    patent_embeddings = model.encode(
        patent_inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        prompt="Retrieve relevant products for the given patent.",
    )

    print(f"Encoding {len(product_inputs)} products...")
    product_embeddings = model.encode(
        product_inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        prompt="",
    )

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

    results = {
        "config": {
            "method": "qwen3_vl",
            "model": model_name,
            "mode": mode,
            "masked": masked,
        },
        "metrics": metrics,
    }
    save_results(results, Path(output_dir) / f"qwen3_vl_{mode}.json")
    return results
