"""
I/O utils and product schema normalization.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# Fields that are part of the product schema.
# Any source field NOT in this set is moved to `details`.
CANONICAL_FIELDS = {
    "product_id",
    "title",
    "description",
    "features",
    "images",
    "videos",
    "price",
    "rating",
    "details",
    "source_file",
    "extracted_patents",
}

# Source field names that map to `product_id` automatically, if present, in order of precedence.
AUTO_ID_ALIASES = ["parent_asin", "asin"]  # "product_id" handled separately below


def read_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    """Yield parsed records from a JSONL file, skipping blank lines."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at {path}:{line_num} -- {e}")


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    """Write a list of records to a JSONL file, overwriting if it exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: Dict[str, Any], path: str) -> None:
    """Append a single record to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_product_record(
    record: Dict[str, Any],
    product_id_field: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize a raw source record into the pipeline schema."""
    record = dict(record)  # shallow copy not to mutate the original

    # Resolve product_id
    id_value: Optional[str] = None

    if product_id_field and product_id_field in record:
        id_value = record.pop(product_id_field)
    elif "product_id" in record:
        id_value = record["product_id"]
    else:
        for alias in AUTO_ID_ALIASES:
            if alias in record:
                id_value = record.pop(alias)
                break

    if id_value is None:
        raise ValueError(
            f"Cannot resolve product_id. Tried explicit field "
            f"'{product_id_field}', and aliases {sorted(AUTO_ID_ALIASES)}. "
            f"Available fields: {list(record.keys())}"
        )

    record["product_id"] = str(id_value)

    # Move unknown fields to `details`
    details = record.pop("details", {}) or {}
    extra_keys = [k for k in list(record.keys()) if k not in CANONICAL_FIELDS]
    for key in extra_keys:
        details[key] = record.pop(key)

    if details:
        record["details"] = details

    return record


def count_lines(path: str) -> int:
    """Count non-empty lines in a file (fast, no JSON parsing)."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count
