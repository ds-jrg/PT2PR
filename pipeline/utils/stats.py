"""
Unified statistics module for the patent-product pipeline.
"""

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def _patent_key(record: Dict) -> Optional[Tuple[str, str, str]]:
    """
    Return (country_code, patent_number, kind_code) as the unique patent identity key,
    or None if patent_number is missing/empty.
    """
    country = str(record.get("country_code") or "").strip()
    number = str(record.get("patent_number") or "").strip()
    kind = str(record.get("kind_code") or "").strip()
    if not number:
        return None
    return (country or "UNKNOWN", number, kind)


def _compute_country_stats(records: List[Dict]) -> List[Dict]:
    """Compute per-country pair/product/patent counts for steps 02 and 03."""
    # country -> set of product_ids
    country_products: Dict[str, Set[str]] = defaultdict(set)
    # country -> set of (country, patent_number, kind_code) keys
    country_patents: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
    # total row count per country (including empty-patent-number rows)
    country_rows: Dict[str, int] = defaultdict(int)

    global_products: Set[str] = set()
    global_patents: Set[Tuple[str, str, str]] = set()
    total_rows = 0

    for record in records:
        product_id = str(record.get("product_id") or "").strip()
        if not product_id:
            continue

        country = str(record.get("country_code") or "UNKNOWN").strip()
        pk = _patent_key(record)

        country_rows[country] += 1
        total_rows += 1

        if product_id:
            country_products[country].add(product_id)
            global_products.add(product_id)

        if pk:
            country_patents[country].add(pk)
            global_patents.add(pk)

    all_countries = sorted(set(country_products) | set(country_patents))
    rows = []
    for country in all_countries:
        rows.append(
            {
                "country": country,
                "total_pairs": country_rows[country],
                "unique_products": len(country_products[country]),
                "unique_patents": len(country_patents[country]),
            }
        )

    rows.append(
        {
            "country": "TOTAL",
            "total_pairs": total_rows,
            "unique_products": len(global_products),
            "unique_patents": len(global_patents),
        }
    )
    return rows


def _compute_raw_extraction_stats(records: List[Dict]) -> Optional[Dict]:
    """
    Stats for step 01 output: per-source-file product/match counts.
    Only produced if records have a 'source_file' field.
    """
    if not records or "source_file" not in records[0]:
        return None

    by_source: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"products": 0, "patented": 0}
    )
    for r in records:
        src = r.get("source_file", "unknown")
        by_source[src]["products"] += 1
        if r.get("extracted_patents"):
            by_source[src]["patented"] += 1

    rows = []
    totals = {"products": 0, "patented": 0}
    for src, counts in sorted(by_source.items()):
        rows.append(
            {
                "source_file": src,
                "number_of_products": counts["products"],
                "number_of_patented_products_found": counts["patented"],
            }
        )
        totals["products"] += counts["products"]
        totals["patented"] += counts["patented"]

    rows.append(
        {
            "source_file": "TOTAL",
            "number_of_products": totals["products"],
            "number_of_patented_products_found": totals["patented"],
        }
    )
    return {"type": "raw_extraction", "rows": rows}


def _compute_patent_info_stats(records: List[Dict]) -> Optional[Dict]:
    """Stats for step 04 output: metadata coverage per unique patent by country."""
    if not records or "patent_data" not in records[0]:
        return None

    # Deduplicate by full (country, number, kind) triple
    seen: Set[str] = set()
    unique: List[Dict] = []
    for r in records:
        key = (
            str(r.get("country_code") or ""),
            str(r.get("patent_number") or ""),
            str(r.get("kind_code") or ""),
        )
        key_str = "\x00".join(key)
        if key_str not in seen:
            seen.add(key_str)
            unique.append(r)

    by_country: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    global_counts: Dict[str, int] = defaultdict(int)

    for r in unique:
        country = str(r.get("country_code") or "UNKNOWN")
        pd = r.get("patent_data")

        # Failed fetch: patent_data is None, or is a dict with an 'error' key
        fetch_failed = pd is None or (isinstance(pd, dict) and "error" in pd)

        for scope in (by_country[country], global_counts):
            scope["total"] += 1
            if fetch_failed:
                scope["fetch_failed"] += 1
                continue
            pd_dict = pd or {}
            if _is_nonempty(pd_dict.get("abstract")):
                scope["has_abstract"] += 1
            if _is_nonempty(pd_dict.get("description")):
                scope["has_description"] += 1
            if _is_nonempty(pd_dict.get("claims")):
                scope["has_claims"] += 1
            if _is_nonempty(pd_dict.get("images")):
                scope["has_images"] += 1

    stat_fields = [
        "total",
        "fetch_failed",
        "has_abstract",
        "has_description",
        "has_claims",
        "has_images",
    ]
    rows = []
    for country in sorted(by_country):
        row = {"country": country}
        row.update({f: by_country[country].get(f, 0) for f in stat_fields})
        rows.append(row)
    global_row = {"country": "TOTAL"}
    global_row.update({f: global_counts.get(f, 0) for f in stat_fields})
    rows.append(global_row)

    return {"type": "patent_info_coverage", "rows": rows}


def compute_and_save_stats(
    records: List[Dict],
    output_jsonl_path: str,
    step: str,
    stats_suffix: str = "_stats",
) -> None:
    """
    Compute stats appropriate for the given step and write them to a CSV
    file alongside the output JSONL.
    """
    if not records:
        logger.warning(f"[{step}] No records: SKIPPING stats.")
        return

    out_path = Path(output_jsonl_path)
    stats_path = out_path.parent / f"{out_path.stem}{stats_suffix}.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    raw_stats = _compute_raw_extraction_stats(records)
    patent_info_stats = _compute_patent_info_stats(records)

    if raw_stats:
        primary = raw_stats
    elif patent_info_stats:
        primary = patent_info_stats
    else:
        primary = {"type": "country", "rows": _compute_country_stats(records)}

    _write_csv(primary["rows"], str(stats_path))
    logger.info(f"[{step}] Stats -> {stats_path}")


def _write_csv(rows: List[Dict], path: str) -> None:
    """Write a list of dicts to a CSV file."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
