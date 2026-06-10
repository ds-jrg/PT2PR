"""
Data preprocessing script for patent-product pairs.
Deduplicates entities and extracts only necessary textual fields.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def is_design_patent(pair: Dict[str, Any]) -> bool:
    """Check if pair contains a design patent (patent_number starts with 'D')."""
    try:
        patent_number = pair.get("patent", {}).get("patent_number", "")
        return patent_number.upper().startswith("D")
    except Exception:
        return False


def has_title(text_dict: Dict[str, Any], side: str) -> bool:
    """Check that a patent or product has a non-empty title."""
    if side == "patent":
        patent = text_dict.get("patent_data", text_dict)
        title = patent.get("bibliographic", {}).get("DC.title", "").strip()
    else:
        title = text_dict.get("title", "")
        if isinstance(title, list):
            title = " ".join(title)
        title = title.strip()
    return bool(title)


def has_patent_visuals(pub_number: str, visual_assets_dir: Path) -> bool:
    """Check that at least one hi-res image exists for the patent."""
    hi_res = visual_assets_dir / "patents" / pub_number / "hi_res"
    return hi_res.exists() and any(hi_res.iterdir())


def has_product_visuals(id: str, visual_assets_dir: Path) -> bool:
    """Check that at least one large or hi-res image exists for the product."""
    base = visual_assets_dir / "products" / id / "images"
    return any(
        (base / sub).exists() and any((base / sub).iterdir())
        for sub in ("hi_res", "large")
    )


def extract_product_fields(product: Dict[str, Any]) -> Dict[str, Any]:
    """Extract necessary product fields."""
    return {
        "main_category": product.get("main_category"),
        "title": product.get("title"),
        "features": product.get("features", []),
        "description": product.get("description", []),
        "store": product.get("store"),
        "categories": product.get("categories", []),
        "details": product.get("details", {}),
    }


def extract_patent_fields(patent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract necessary patent fields, excluding unnecessary metadata."""
    patent = patent_data["patent_data"] if "patent_data" in patent_data else patent_data

    biblio = patent.get("bibliographic", {})
    filtered_biblio = {
        k: v
        for k, v in biblio.items()
        if k
        not in [
            "id",
            "full",
            "figurePage",
            "label",
            "left",
            "top",
            "type",
            "Additional",
            "right",
            "bottom",
            "IsCPC",
            "Leaf",
            "FirstCode",
            "num_attr",
            "thisCountry",
            "isPatent",
            "isScholar",
            "scholarID",
            "events",
        ]
    }

    title = biblio.get("DC.title", "").strip()

    return {
        "bibliographic": filtered_biblio,
        "title": title,
        "abstract": patent.get("abstract", ""),
        "description": patent.get("description", ""),
        "claims": patent.get("claims", []),
    }


def get_patent_number(patent_data: Dict[str, Any]) -> str:
    """Extract patent number from patent data."""
    patent = patent_data["patent_data"] if "patent_data" in patent_data else patent_data
    return patent.get("bibliographic", {}).get("publicationNumber", "")


def preprocess_dataset(
    input_path: str,
    output_path: str,
    setting: str,
    visual_assets_dir: Optional[Path],
) -> Dict[str, Any]:
    print(f"Loading data from {input_path}...")

    if input_path.endswith(".jsonl"):
        data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of patent-product pairs")

    total_pairs = len(data)
    print(f"Total pairs in input: {total_pairs}")

    unique_patents = {}
    unique_products = {}
    mappings = []

    skipped_design = 0
    skipped_no_title = 0
    skipped_no_visuals = 0
    processed_pair_count = 0

    for idx, pair in enumerate(data):
        product_id = pair.get("product_id", "")
        product_data = pair.get("product", {})

        if product_id and product_id not in unique_products:
            if has_title(product_data, "product"):
                if setting == "text":
                    unique_products[product_id] = extract_product_fields(product_data)
                elif setting == "multimodal":
                    # For multimodal, products must have visual assets to be encodeable
                    if product_data.get("images") and has_product_visuals(
                        product_id, visual_assets_dir
                    ):
                        unique_products[product_id] = extract_product_fields(
                            product_data
                        )

        # Pair-level filtering (determines query patents and mappings)

        # Text setting: exclude design patents from query side
        if setting == "text" and is_design_patent(pair):
            skipped_design += 1
            continue

        try:
            patent_number = get_patent_number(pair.get("patent", {}))

            if not patent_number or not product_id:
                print(f"Warning: Missing patent number or product ID in pair {idx}")
                skipped_no_title += 1
                continue

            # Both sides must have a title
            if not has_title(pair.get("patent", {}), "patent"):
                skipped_no_title += 1
                continue
            if not has_title(product_data, "product"):
                skipped_no_title += 1
                continue

            # Multimodal setting: both sides must have visual assets on disk.
            # Stage 1: fast pre-filter on in-memory data before touching the filesystem.
            # Stage 2: confirm the downloaded files are actually present on disk.
            if setting == "multimodal":
                if not pair.get("patent", {}).get("patent_data", {}).get("images"):
                    skipped_no_visuals += 1
                    continue
                if not product_data.get("images"):
                    skipped_no_visuals += 1
                    continue
                if not has_patent_visuals(patent_number, visual_assets_dir):
                    skipped_no_visuals += 1
                    continue
                if not has_product_visuals(product_id, visual_assets_dir):
                    skipped_no_visuals += 1
                    continue

            if patent_number not in unique_patents:
                unique_patents[patent_number] = extract_patent_fields(
                    pair.get("patent", {})
                )

            # Only add mapping if product made it into the pool
            if product_id in unique_products:
                mappings.append(
                    {"patent_number": patent_number, "product_id": product_id}
                )
                processed_pair_count += 1

        except Exception as e:
            print(f"Warning: Error processing pair {idx}: {e}")
            skipped_no_title += 1

    output_data = {
        "patents": unique_patents,
        "products": unique_products,
        "mappings": mappings,
        "metadata": {
            "setting": setting,
            "total_input_pairs": total_pairs,
            "design_patents_skipped": skipped_design,
            "no_title_skipped": skipped_no_title,
            "no_visuals_skipped": skipped_no_visuals,
            "valid_pairs_processed": processed_pair_count,
            "unique_patents": len(unique_patents),
            "unique_products": len(unique_products),
        },
    }

    print(f"\nSaving preprocessed data to {output_path}...")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    stats = output_data["metadata"]
    print(f"Setting: {stats['setting']}")
    print(f"Total input pairs: {stats['total_input_pairs']}")
    print(f"Design patents skipped: {stats['design_patents_skipped']}")
    print(f"No title skipped: {stats['no_title_skipped']}")
    print(f"No visuals skipped: {stats['no_visuals_skipped']}")
    print(f"Valid pairs processed: {stats['valid_pairs_processed']}")
    print(f"Unique patents: {stats['unique_patents']}")
    print(f"Unique products: {stats['unique_products']}")
    print(f"Patent-Product mappings: {len(mappings)}")
    print(f"Output saved to: {output_path}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess patent-product pairs dataset with deduplication"
    )
    parser.add_argument(
        "--dataset",
        choices=["amazon", "esci", "both"],
        required=True,
        help="Dataset to preprocess: 'amazon', 'esci', or both.",
    )
    parser.add_argument(
        "--setting",
        choices=["text", "multimodal", "both"],
        default="text",
        help=(
            "'text' filters design patents; 'multimodal' keeps them but requires visual assets; "
            "'both' runs both settings and saves separate outputs."
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Override the input path. By default, resolved automatically from --dataset as "
            "data/processed/<dataset>/full_patent_product_dataset_US_only.jsonl."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Override the output path. By default, resolved automatically from "
            "--dataset and --setting as "
            "experiments/data/<dataset>/preprocessed_data.json (text) or "
            "experiments/data/<dataset>/preprocessed_multimodal_data.json (multimodal)."
        ),
    )
    parser.add_argument(
        "--visual-assets-dir",
        default=None,
        help="Root directory of visual assets. Expected structure: <dir>/<dataset>/patents/ and <dir>/<dataset>/products/.",
    )

    args = parser.parse_args()

    datasets = ["amazon", "esci"] if args.dataset == "both" else [args.dataset]
    settings = ["text", "multimodal"] if args.setting == "both" else [args.setting]

    for dataset in datasets:
        for setting in settings:
            if setting == "multimodal" and not args.visual_assets_dir:
                parser.error(
                    "--visual-assets-dir is required when --setting includes multimodal"
                )

            input_path = (
                args.input
                or f"data/processed/{dataset}/full_patent_product_dataset_US_only.jsonl"
            )

            # Skip if input doesn't exist
            if not Path(input_path).exists():
                print(
                    f"Skipping {dataset}/{setting}: input file not found at {input_path}"
                )
                continue

            # Skip multimodal if no visual assets exist for this dataset
            if setting == "multimodal":
                assets_dir = Path(args.visual_assets_dir) / dataset
                if not assets_dir.exists() or not any(assets_dir.iterdir()):
                    print(
                        f"Skipping {dataset}/multimodal: no visual assets found at {assets_dir}"
                    )
                    continue

            if args.output:
                output_path = args.output
            elif setting == "multimodal":
                output_path = (
                    f"experiments/data/{dataset}/preprocessed_multimodal_data.json"
                )
            else:
                output_path = f"experiments/data/{dataset}/preprocessed_data.json"

            visual_assets_dir = (
                Path(args.visual_assets_dir) / dataset
                if args.visual_assets_dir
                else None
            )

            preprocess_dataset(
                input_path=input_path,
                output_path=output_path,
                setting=setting,
                visual_assets_dir=visual_assets_dir,
            )
