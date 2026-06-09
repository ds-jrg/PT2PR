"""
Step 05: cleaning patent claims.

Removes partial duplicate claims: cases where the same claim number appears
more than once and one version is a fragment or prefix of the other. The
longest (most complete) version is kept.
"""

import argparse
import logging
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from pipeline.utils.io import read_jsonl, write_jsonl
from pipeline.utils.stats import compute_and_save_stats

logger = logging.getLogger(__name__)


def _is_fragment(claim1: Dict[str, Any], claim2: Dict[str, Any]) -> bool:
    """Return True if claim1's text is a substring/prefix of claim2's text."""
    t1 = claim1.get("text", "").strip()
    t2 = claim2.get("text", "").strip()
    if not t1 or not t2 or t1 == t2:
        return False
    if len(t1) < len(t2):
        return t2.startswith(t1) or t1 in t2
    return False


def _remove_partial_duplicates(
    claims: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Group claims by claim_number, keep the longest version when duplicates
    exist and one is a fragment of the other.
    """
    if not claims:
        return claims, 0

    by_number: Dict[Any, List[Dict]] = {}
    for claim in claims:
        num = claim.get("claim_number")
        by_number.setdefault(num, []).append(claim)

    cleaned: List[Dict] = []
    removed = 0

    for num, group in by_number.items():
        if len(group) == 1:
            cleaned.append(group[0])
            continue

        group_sorted = sorted(group, key=lambda c: len(c.get("text", "")), reverse=True)
        longest = group_sorted[0]
        has_fragments = False
        for other in group_sorted[1:]:
            if _is_fragment(other, longest):
                has_fragments = True
                removed += 1
        if has_fragments:
            cleaned.append(longest)
        else:
            cleaned.extend(group)

    cleaned.sort(
        key=lambda c: c.get("claim_number")
        if c.get("claim_number") is not None
        else float("inf")
    )
    return cleaned, removed


def _clean_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    patent_data = record.get("patent_data")
    if not patent_data or not patent_data.get("claims"):
        return record, 0
    cleaned, removed = _remove_partial_duplicates(patent_data["claims"])
    record["patent_data"]["claims"] = cleaned
    return record, removed


def clean_patent_info(input_path: str, output_path: str) -> List[Dict]:
    """Read step 04 output, remove partial duplicate claims, write cleaned records."""
    records = list(read_jsonl(input_path))
    total = len(records)
    logger.info(f"Processing {total} records from '{input_path}'.")

    cleaned_records: List[Dict] = []
    total_removed = 0
    affected = 0

    for record in tqdm(records, desc="Cleaning claims"):
        cleaned, removed = _clean_record(record)
        if removed > 0:
            total_removed += removed
            affected += 1
        cleaned_records.append(cleaned)

    write_jsonl(cleaned_records, output_path)
    compute_and_save_stats(cleaned_records, output_path, step="05_clean_patent_info")

    logger.info(f"Records affected: {affected}")
    logger.info(f"Partial duplicates removed: {total_removed}")
    logger.info(f"Output written to: '{output_path}'")

    return cleaned_records


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step 05: removing partial duplicate patent claims."
    )
    parser.add_argument("--input", required=True, help="Step 04 output JSONL.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
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
    clean_patent_info(input_path=args.input, output_path=args.output)
