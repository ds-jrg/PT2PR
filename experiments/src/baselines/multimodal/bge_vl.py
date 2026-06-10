"""
BGE-VL-MLLM multimodal retrieval baseline for patent-product matching.
"""

import torch
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

# Task-specific prompts per mode (query side only)
_PROMPTS = {
    "text": "Retrieve the most relevant product for the given patent: ",
    "image": "Retrieve the most relevant product image for the given patent drawing: ",
    "multimodal": "Retrieve the most relevant product for the given patent, considering both the patent text and drawings: ",
}


def _print_seq_lengths(inputs, label: str, model):
    """Print token sequence length statistics for a sample of inputs."""
    lengths = []
    for inp in inputs[:10]:
        text = inp.get("text", inp) if isinstance(inp, dict) else inp
        if isinstance(text, str):
            tokens = model.tokenizer(text, return_tensors="pt")
            lengths.append(tokens["input_ids"].shape[1])
    if lengths:
        print(
            f"{label} sequence lengths (first {len(lengths)}): "
            f"min={min(lengths)}, max={max(lengths)}, avg={sum(lengths) / len(lengths):.0f}"
        )


def run(
    preprocessed_path: str,
    output_dir: str,
    visual_assets_dir: str,
    mode: Literal["text", "image", "multimodal"] = "multimodal",
    model_name: str = "BAAI/BGE-VL-v1.5-mmeb",
    masked: bool = True,
    batch_size: int = 1,
    max_text_length: int = 8192,
    k_values: List[int] = [5, 10, 20],
) -> Dict:
    """BGE-VL-MLLM retrieval baseline."""
    print(f"\nEXPERIMENT: BGE-VL-MLLM ({mode})")
    print(f"Model: {model_name}  Masked: {masked}  Max text length: {max_text_length}")

    patents, products, all_patent_nums, all_product_ids, positive_map = load_data(
        preprocessed_path
    )
    assets_dir = Path(visual_assets_dir)

    print(f"Loading model {model_name}...")
    model = SentenceTransformer(
        model_name,
        model_kwargs={
            "torch_dtype": torch.float16,
            "device_map": "auto",
        },
    )

    def _truncate(text: str) -> str:
        """
        Truncate text to max_text_length tokens before passing to the model.
        Pre-truncation avoids internal truncation cutting through image token
        placeholders in multimodal mode, which causes a token count mismatch.
        """
        ids = model.tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][
            0
        ]
        if len(ids) > max_text_length:
            ids = ids[:max_text_length]
            return model.tokenizer.decode(ids, skip_special_tokens=True)
        return text

    def _build_inputs(entity_ids, get_text_fn, get_image_path_fn, label: str):
        """Build the input list for a set of entities."""
        inputs = []
        for eid in entity_ids:
            if mode == "text":
                inputs.append(_truncate(get_text_fn(eid)))
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
                    inputs.append(
                        {
                            "text": _truncate(get_text_fn(eid)),
                            "image": str(path),
                        }
                    )
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

    _print_seq_lengths(patent_inputs, "patents", model)
    _print_seq_lengths(product_inputs, "products", model)

    # Patents encoded with task-specific prompt; products encoded without prompt
    print(f"Encoding {len(patent_inputs)} patents...")
    patent_embeddings = model.encode(
        patent_inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        prompt=_PROMPTS[mode],
    )

    print(f"Encoding {len(product_inputs)} products...")
    product_embeddings = model.encode(
        product_inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
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
        "config": {
            "method": model_slug,
            "model": model_name,
            "mode": mode,
            "masked": masked,
            "max_text_length": max_text_length,
        },
        "metrics": metrics,
    }
    save_results(results, Path(output_dir) / f"{model_slug}_{mode}.json")
    return results
