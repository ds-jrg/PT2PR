"""
Merging step 01: building the full patent-product dataset.

Joins the cleaned patent-product pairs (with full patent metadata) against
the raw patented products file (which holds product text, images, etc.) on
product_id. Produces one record per (product_id, patent_number) pair plus
coverage stats. Also produces a US-only subset automatically.
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List
from pipeline.utils.io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


def _load_patents(path: str) -> Dict[str, List[dict]]:
    """Group patent records by product_id."""
    by_product: Dict[str, List[dict]] = defaultdict(list)
    for line_num, record in enumerate(read_jsonl(path), 1):
        pid = record.get("product_id")
        if not pid:
            raise ValueError(
                f"Missing product_id in patent record at line {line_num} of '{path}'."
            )
        by_product[pid].append(record)
    total = sum(len(v) for v in by_product.values())
    logger.info(f"Loaded {total} patent records for {len(by_product)} unique products.")
    return dict(by_product)


def _load_products(path: str) -> Dict[str, dict]:
    """Index product records by product_id."""
    by_id: Dict[str, dict] = {}
    for line_num, record in enumerate(read_jsonl(path), 1):
        pid = record.get("product_id")
        if not pid:
            raise ValueError(
                f"Missing product_id in product record at line {line_num} of '{path}'."
            )
        by_id[pid] = record
    logger.info(f"Loaded {len(by_id)} unique products.")
    return by_id


def _merge(
    patents_by_product: Dict[str, List[dict]],
    products_by_id: Dict[str, dict],
) -> List[dict]:
    """Produce one merged record per (product_id, patent_number) pair."""
    common_ids = set(patents_by_product) & set(products_by_id)
    logger.info(
        f"Merging: {len(common_ids)} product_ids have both patent and product data."
    )

    merged: List[dict] = []
    seen_pairs: set = set()
    skipped_no_title = 0
    skipped_duplicates = 0

    for pid in common_ids:
        product = products_by_id[pid]
        for patent in patents_by_product[pid]:
            title = (
                patent.get("patent_data", {}).get("bibliographic", {}).get("DC.title")
            )
            if not title or not str(title).strip():
                skipped_no_title += 1
                continue

            patent_number = patent.get("patent_number")
            if not patent_number:
                continue

            pair_key = (pid, patent_number)
            if pair_key in seen_pairs:
                skipped_duplicates += 1
                continue
            seen_pairs.add(pair_key)

            merged.append(
                {
                    "product_id": pid,
                    "product": product,
                    "patent": patent,
                }
            )

    logger.info(f"Created {len(merged)} merged patent-product pairs.")
    logger.info(f"Skipped {skipped_no_title} records with no patent title.")
    logger.info(
        f"Skipped {skipped_duplicates} duplicate (product_id, patent_number) pairs."
    )
    return merged


def _nonempty(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def _is_active(patent: dict) -> bool:
    status = (
        patent.get("patent_data", {}).get("bibliographic", {}).get("legalStatusIfi", "")
    )
    return any(w in status.lower() for w in ("active", "granted"))


def _compute_pair_stats(records: List[dict]) -> dict:
    total = len(records)
    patent_countries: Counter = Counter()
    patent_fields = defaultdict(int)
    product_fields = defaultdict(int)
    multimodal = defaultdict(int)

    for r in records:
        pat = r.get("patent", {})
        prod = r.get("product", {})
        pd = pat.get("patent_data") or {}

        patent_countries[pat.get("country_code", "UNKNOWN")] += 1

        if _nonempty(pd.get("abstract")):
            patent_fields["has_abstract"] += 1
        if _nonempty(pd.get("description")):
            patent_fields["has_description"] += 1
        if _nonempty(pd.get("claims")):
            patent_fields["has_claims"] += 1
        if _nonempty(pd.get("images")):
            patent_fields["has_patent_images"] += 1

        if _nonempty(prod.get("title")):
            product_fields["has_title"] += 1
        if _nonempty(prod.get("description")):
            product_fields["has_description"] += 1
        if _nonempty(prod.get("features")):
            product_fields["has_features"] += 1
        if _nonempty(prod.get("images")):
            product_fields["has_product_images"] += 1

        if _nonempty(pd.get("abstract")) and _nonempty(prod.get("title")):
            multimodal["text_text"] += 1
        if _nonempty(pd.get("images")) and _nonempty(prod.get("images")):
            multimodal["image_image"] += 1
        if _nonempty(pd.get("abstract")) and _nonempty(prod.get("images")):
            multimodal["patent_text_product_image"] += 1

    return {
        "total": total,
        "unique_patents": len(
            {
                (
                    r["patent"].get("country_code", ""),
                    r["patent"].get("patent_number", ""),
                    r["patent"].get("kind_code", ""),
                )
                for r in records
            }
        ),
        "unique_products": len({r["product_id"] for r in records}),
        "patent": {**dict(patent_fields), "patent_countries": patent_countries},
        "product": dict(product_fields),
        "multimodal": dict(multimodal),
    }


def _compute_entity_stats(records: List[dict], entity: str) -> dict:
    seen: Dict[tuple, dict] = {}
    for r in records:
        if entity == "patent":
            pat = r["patent"]
            key = (
                pat.get("country_code", ""),
                pat.get("patent_number", ""),
                pat.get("kind_code", ""),
            )
        else:
            key = (r["product_id"],)
        if all(key) and key not in seen:
            seen[key] = r
    return {"total": len(seen), "stats": _compute_pair_stats(list(seen.values()))}


def _save_stats(stats: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, Counter):
            return dict(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_convert(stats), f, indent=2, ensure_ascii=False)
    logger.info(f"Stats written to '{path}'.")


def _print_stats(stats: dict) -> None:
    for category, data in stats.items():
        total = data.get("total", 0) or data.get("stats", {}).get("total", 0)
        print(f"{category.replace('_', ' ').upper()} (n={total:,})")

        inner = data.get("stats", data)
        for key, value in inner.items():
            if key == "total":
                continue
            if isinstance(value, Counter):
                print(f"{key}:")
                for k, v in value.most_common():
                    print(f"{k}: {v:,} ({v / total * 100:.1f}%)")
            elif isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    if isinstance(v, int) and total:
                        print(f"{k}: {v:,} ({v / total * 100:.1f}%)")
                    else:
                        print(f"{k}: {v}")
            elif isinstance(value, int) and total:
                print(f"{key}: {value:,} ({value / total * 100:.1f}%)")
            else:
                print(f"{key}: {value}")


def _process_us_subset(
    merged: List[dict], merged_output_path: str, stats_output_path: str
) -> None:
    us_records = [r for r in merged if r["patent"].get("country_code") == "US"]
    if not us_records:
        logger.info("No US patents found; skipping US subset.")
        return

    us_output = merged_output_path.replace(".jsonl", "_US_only.jsonl")
    write_jsonl(us_records, us_output)
    logger.info(f"US-only subset: {len(us_records)} records -> '{us_output}'.")

    us_stats = {
        "all_records": _compute_pair_stats(us_records),
        "unique_patents": _compute_entity_stats(us_records, "patent"),
        "unique_products": _compute_entity_stats(us_records, "product"),
        "unique_active_patents": _compute_entity_stats(
            [r for r in us_records if _is_active(r["patent"])], "patent"
        ),
    }

    us_stats_path = stats_output_path.replace(".json", "_US_only.json")
    _save_stats(us_stats, us_stats_path)


def build_full_dataset(
    patents_input: str,
    products_input: str,
    output_path: str,
    stats_output_path: str,
) -> List[dict]:
    """
    Merge cleaned patent info with product data, compute and save stats,
    and produce a US-only subset.
    """
    patents_by_product = _load_patents(patents_input)
    products_by_id = _load_products(products_input)

    merged = _merge(patents_by_product, products_by_id)

    write_jsonl(merged, output_path)
    logger.info(f"Merged dataset written to '{output_path}'.")

    stats = {
        "all_records": _compute_pair_stats(merged),
        "unique_patents": _compute_entity_stats(merged, "patent"),
        "unique_products": _compute_entity_stats(merged, "product"),
        "unique_active_patents": _compute_entity_stats(
            [r for r in merged if _is_active(r["patent"])], "patent"
        ),
    }

    _print_stats(stats)
    _save_stats(stats, stats_output_path)

    _process_us_subset(merged, output_path, stats_output_path)

    return merged


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Merging step 01: building full patent-product dataset."
    )
    parser.add_argument(
        "--patents-input",
        required=True,
        help="Step 05 output: patent-product pairs with full patent metadata.",
    )
    parser.add_argument(
        "--products-input",
        required=True,
        help="Step 01 output: raw patented products JSONL.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--stats-output",
        required=True,
        help="Output JSON path for coverage stats.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    build_full_dataset(
        patents_input=args.patents_input,
        products_input=args.products_input,
        output_path=args.output,
        stats_output_path=args.stats_output,
    )
