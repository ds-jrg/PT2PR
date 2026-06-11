"""
Baseline experiments for patent-product retrieval.
"""

import argparse
from pathlib import Path
from experiments.src.baselines.text import (
    tfidf,
    bm25,
    qwen3_embedder,
    patentsberta,
    bge,
    all_mpnet,
)
from experiments.src.baselines.multimodal import qwen3_vl_embedder as qwen3_vl, bge_vl

# Default data root; individual paths are resolved from --dataset and --setting.
_DATA_ROOT = "experiments/data"
_RESULTS_ROOT = "experiments/results"


def _resolve_paths(dataset: str, setting: str, visual_assets_dir: str | None):
    """
    Return (preprocessed_path, output_dir, visual_assets_dir) for a given
    dataset and setting, using the standard directory layout.
    """
    data_dir = f"{_DATA_ROOT}/{dataset}"
    if setting == "multimodal":
        preprocessed = f"{data_dir}/masked_multimodal_data.json"
    else:
        preprocessed = f"{data_dir}/masked_data.json"
    output_dir = f"{_RESULTS_ROOT}/{dataset}/{setting}_setting"
    assets_dir = str(Path(visual_assets_dir) / dataset) if visual_assets_dir else None
    return preprocessed, output_dir, assets_dir


def run_text_setting(dataset: str, no_mask: bool):
    print(f"TEXT SETTING [{dataset}]")
    preprocessed, output_dir, _ = _resolve_paths(dataset, "text", None)
    shared = dict(
        preprocessed_path=preprocessed,
        output_dir=output_dir,
        masked=not no_mask,
    )
    tfidf.run(**shared)
    bm25.run(**shared)
    all_mpnet.run(**shared)
    patentsberta.run(**shared)
    bge.run(**shared)
    qwen3_embedder.run(**shared)


def run_multimodal_setting(dataset: str, visual_assets_dir: str, no_mask: bool):
    print(f"MULTIMODAL SETTING [{dataset}]")
    preprocessed, output_dir, assets_dir = _resolve_paths(
        dataset, "multimodal", visual_assets_dir
    )

    text_shared = dict(
        preprocessed_path=preprocessed,
        output_dir=output_dir,
        masked=not no_mask,
    )
    mm_shared = dict(
        preprocessed_path=preprocessed,
        output_dir=output_dir,
        visual_assets_dir=assets_dir,
        masked=not no_mask,
    )

    # Text-only references on the multimodal subset
    bge.run(**text_shared)
    qwen3_embedder.run(**text_shared)

    # BGE-VL: text, image, multimodal
    bge_vl.run(**mm_shared, mode="text")
    bge_vl.run(**mm_shared, mode="image")
    bge_vl.run(**mm_shared, mode="multimodal")

    # Qwen3-VL: text, image, multimodal
    qwen3_vl.run(**mm_shared, mode="text")
    qwen3_vl.run(**mm_shared, mode="image")
    qwen3_vl.run(**mm_shared, mode="multimodal")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline experiments for patent-product retrieval"
    )
    parser.add_argument(
        "--dataset",
        choices=["amazon", "esci", "both"],
        default="both",
        help=(
            "Dataset to run experiments on. 'both' runs amazon then esci. "
            "Default: both."
        ),
    )
    parser.add_argument(
        "--setting",
        choices=["text", "multimodal", "both"],
        default="both",
        help=(
            "Experiment setting to run. 'text' uses the full dataset; "
            "'multimodal' uses the visual-asset subset; "
            "'both' runs text then multimodal. Default: both."
        ),
    )
    parser.add_argument(
        "--visual-assets-dir",
        default=None,
        help=(
            "Root directory of downloaded visual assets. Required when "
            "--setting is 'multimodal' or 'both'. The path depends on where "
            "the assets are stored and is not set by default."
        ),
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Use plain (unmasked) text fields instead of *_masked variants.",
    )

    args = parser.parse_args()

    if args.setting in ("multimodal", "both") and not args.visual_assets_dir:
        parser.error(
            "--visual-assets-dir is required when --setting is 'multimodal' or 'both'"
        )

    datasets = ["amazon", "esci"] if args.dataset == "both" else [args.dataset]

    for dataset in datasets:
        if args.setting in ("text", "both"):
            run_text_setting(dataset, args.no_mask)

        if args.setting in ("multimodal", "both"):
            assets_path = Path(args.visual_assets_dir) / dataset
            if not assets_path.exists():
                print(f"[INFO] Skipping multimodal for '{dataset}': {assets_path} not found.")
            else:
                run_multimodal_setting(dataset, args.visual_assets_dir, args.no_mask)

    print("\nAll experiments complete.")
    print(f"Results written under: {_RESULTS_ROOT}/")


if __name__ == "__main__":
    main()
