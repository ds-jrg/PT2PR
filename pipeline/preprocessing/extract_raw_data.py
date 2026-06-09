"""
Step 01: extracting patented products from raw source catalog.

Scans compressed JSONL catalog files (.jsonl.gz) for products that mention
a patent number in their title, description, or features fields.
"""

import argparse
import gzip
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set
from tqdm import tqdm
from pipeline.utils.io import append_jsonl, normalize_product_record
from pipeline.utils.stats import compute_and_save_stats

logger = logging.getLogger(__name__)


# High-confidence patent-phrase pattern.
# A match here is sufficient without requiring a nearby number.
_PHRASE_PATTERN = re.compile(
    r"(?i)\b(?:patent\s+numbers?|patent\s+no\.?s?|patent\s+num\.?s?|pat\.\s*no\.?s?|pat\.\s*num\.?s?)"
)

# Broader keyword pattern: only qualifies when a patent number is also nearby.
_KEYWORD_PATTERN = re.compile(r"(?i)\b(?:patent|pat\.?s?(?=\W|$))")

# Patent number formats (with and without thousands-separator commas).
_NUMBER_PLAIN = re.compile(r"([A-Z]{1,3})?(\d{5,14})([A-Z]\d?$)?", re.I)
_NUMBER_COMMA = re.compile(r"([A-Z]{1,3})?(\d{1,3}(?:,\d{3})+)([A-Z]\d?)?", re.I)

# Characters to look back/forward from a keyword match when checking for a number.
_SPAN_BEFORE = 200
_SPAN_AFTER = 200


def _text_from_field(value) -> str:
    """Flatten a field value (str or list) to a single string."""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value) if value else ""


def _has_patent_reference(record: Dict) -> bool:
    """Return True if the record's text includes a qualifying patent reference."""
    for field in ("title", "description", "features"):
        raw = record.get(field)
        if not raw:
            continue
        text = _text_from_field(raw)

        if _PHRASE_PATTERN.search(text):
            return True

        for match in _KEYWORD_PATTERN.finditer(text):
            start = max(0, match.start() - _SPAN_BEFORE)
            end = min(len(text), match.end() + _SPAN_AFTER)
            span = text[start:end]
            if _NUMBER_COMMA.search(span) or _NUMBER_PLAIN.search(span):
                return True

    return False


def _record_hash(record: Dict) -> str:
    """MD5 hash of the full JSON content used for within-run deduplication."""
    return hashlib.md5(
        json.dumps(record, sort_keys=True, default=str).encode()
    ).hexdigest()


def _iter_gz_jsonl(file_path: str) -> Generator[Dict, None, None]:
    """Yield parsed records from a gzipped JSONL file."""
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Malformed JSON at {file_path}:{line_num} -- {e}")
    except Exception as e:
        logger.error(f"Cannot open {file_path}: {e}")


def extract_patented_products(
    input_dir: str,
    output_path: str,
    product_id_field: Optional[str] = None,
) -> List[Dict]:
    """
    Scan all .jsonl.gz files in input_dir, filter for patent-mentioning products,
    normalize to the canonical schema, write to output_path, and return all records.
    """
    input_path = Path(input_dir)
    gz_files = sorted(input_path.glob("*.jsonl.gz"))

    if not gz_files:
        raise FileNotFoundError(f"No .jsonl.gz files found in '{input_dir}'.")

    logger.info(f"Found {len(gz_files)} source file(s) in '{input_dir}'.")

    # Clear output file before starting
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    seen_hashes: Set[str] = set()
    all_records: List[Dict] = []

    total_scanned = 0
    total_matched = 0

    for gz_file in tqdm(gz_files, desc="Scanning source files"):
        file_matched = 0

        for raw_record in _iter_gz_jsonl(str(gz_file)):
            total_scanned += 1

            if not _has_patent_reference(raw_record):
                continue

            # Deduplication before normalization (raw content hash)
            h = _record_hash(raw_record)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # Normalize to canonical schema
            try:
                record = normalize_product_record(raw_record, product_id_field)
            except ValueError as e:
                logger.warning(f"Skipping record; normalization failed: {e}")
                continue

            record["extracted_patents"] = True
            record["source_file"] = gz_file.name

            append_jsonl(record, output_path)
            all_records.append(record)
            file_matched += 1
            total_matched += 1

            if total_scanned % 50_000 == 0:
                logger.info(f"Scanned {total_scanned:,} | matched {total_matched:,}")

        logger.info(f"{gz_file.name}: {file_matched} patent-mentioning products found.")

    logger.info(
        f"Done. Scanned {total_scanned:,} records total, "
        f"wrote {total_matched:,} to '{output_path}'."
    )

    compute_and_save_stats(all_records, output_path, step="01_extract_raw_data")

    return all_records


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step 01: extracting patent-mentioning products from source catalog."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing .jsonl.gz source catalog files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for extracted patented products.",
    )
    parser.add_argument(
        "--product-id-field",
        default=None,
        help=(
            "Source field name that holds the product identifier. "
            "Defaults to auto-detection (parent_asin, asin, product_id). "
            "Only needed if the product catalog uses a non-standard field name."
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
    extract_patented_products(
        input_dir=args.input_dir,
        output_path=args.output,
        product_id_field=args.product_id_field,
    )
