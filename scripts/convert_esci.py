"""
Convert the ESCI Shopping Queries product catalog into a general format (.jsonl.gz)
which is compatible with the Amazon Reviews 2023 dataset metadata format
so that the pipeline's step 1 (raw data extraction) can process it without modification.

Only English (product_locale == "us") products are included.
Duplicate product_ids are dropped, keeping the first occurrence.
"""

import argparse
import gzip
import json
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def _split_bullets(text) -> list:
    """Split bullet point string on newlines, dropping empty entries."""
    if not text or not isinstance(text, str):
        return []
    return [b.strip() for b in text.split("\n") if b.strip()]


def _wrap_description(text) -> list:
    """Wrap description string in a list for format compatibility."""
    if not text or not isinstance(text, str):
        return []
    return [text.strip()]


def _build_record(row, id_field: str) -> dict:
    """Convert a single ESCI product row to Amazon Reviews 2023 format."""
    details = {}
    if pd.notna(row.get("product_brand")) and row["product_brand"]:
        details["Brand"] = str(row["product_brand"]).strip()
    if pd.notna(row.get("product_color")) and row["product_color"]:
        details["Color"] = str(row["product_color"]).strip()

    return {
        id_field: str(row["product_id"]),
        "title": str(row["product_title"]).strip()
        if pd.notna(row.get("product_title"))
        else "",
        "description": _wrap_description(row.get("product_description")),
        "features": _split_bullets(row.get("product_bullet_point")),
        "store": None,  # Not available in ESCI (same as all empty lists or None values below)
        "categories": [],
        "details": details,
        "images": [],
        "videos": [],
        "price": None,
        "average_rating": None,
        "rating_number": None,
        "main_category": None,
        "bought_together": None,
    }


def convert(
    products_path: str, output_path: str, id_field: str = "parent_asin"
) -> None:
    logger.info(f"Loading products from {products_path}...")
    df = pd.read_parquet(products_path)
    logger.info(f"Total products: {len(df)}")

    df = df[df["product_locale"] == "us"]
    logger.info(f"English (US) products: {len(df)}")

    before = len(df)
    df = df.drop_duplicates(subset="product_id", keep="first")
    logger.info(f"After deduplication: {len(df)} (dropped {before - len(df)})")

    if id_field == "product_id":
        logger.info("ID field: product_id (no renaming)")
    else:
        logger.info("ID field: rename to product_id")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing to {output_path}...")
    written = 0
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = _build_record(row, id_field)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Done. {written} products written to {output_path} ({size_mb:.1f} MB)")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ESCI product catalog parquet to .jsonl.gz format."
    )
    parser.add_argument(
        "--products",
        required=True,
        help="Path to shopping_queries_dataset_products.parquet.",
    )
    parser.add_argument(
        "--output",
        default="data/external/esci/products.jsonl.gz",
        help="Output .jsonl.gz path (default: data/external/esci/products.jsonl.gz).",
    )
    parser.add_argument(
        "--id-field",
        default="product_id",
        choices=["product_id", "parent_asin"],
        help=(
            "Output field name for the product identifier. "
            "'product_id' (default) is the pipeline's canonical field name. "
            "'parent_asin' uses the Amazon Reviews 2023 schema name instead."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    convert(
        products_path=args.products,
        output_path=args.output,
        id_field=args.id_field,
    )
