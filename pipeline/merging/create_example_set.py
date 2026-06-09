"""
Merging step 02: creating a random example set for manual verification.

Samples 20 unique patents from the merged dataset and writes them to a
pretty-printed JSON file for human review.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from pipeline.utils.io import read_jsonl

logger = logging.getLogger(__name__)


def _patent_key(record: Dict[str, Any]) -> Optional[str]:
    """Return a unique key for the patent in this record, or None if fields are missing."""
    patent = record.get("patent", {})
    country = patent.get("country_code", "").strip()
    number = patent.get("patent_number", "").strip()
    kind = patent.get("kind_code", "").strip()
    if not all((country, number, kind)):
        return None
    return f"{country}{number}{kind}"


def create_example_set(
    input_path: str,
    output_path: str,
    n: int = 20,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample N unique patents from the merged dataset and write to JSON."""
    patent_map: Dict[str, Dict[str, Any]] = {}
    total_lines = 0
    skipped = 0

    for record in read_jsonl(input_path):
        total_lines += 1
        key = _patent_key(record)
        if not key:
            skipped += 1
            continue
        if key not in patent_map:
            patent_map[key] = record

    logger.info(f"Processed {total_lines} lines.")
    logger.info(f"Found {len(patent_map)} unique patents.")
    if skipped:
        logger.info(f"Skipped {skipped} records with incomplete patent fields.")

    if not patent_map:
        raise ValueError("No valid patent records found in input.")

    sample_size = min(n, len(patent_map))
    if sample_size < n:
        logger.warning(
            f"Requested {n} examples but only {len(patent_map)} unique patents available. "
            f"Returning all {sample_size}."
        )

    random.seed(seed)
    sampled_keys = random.sample(list(patent_map.keys()), sample_size)
    sampled = [patent_map[k] for k in sampled_keys]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=4, ensure_ascii=False)

    logger.info(f"Wrote {len(sampled)} examples to '{output_path}'.")
    return sampled


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Merging step 02: sampling example records for manual verification."
    )
    parser.add_argument("--input", required=True, help="Merged dataset JSONL.")
    parser.add_argument(
        "--output", required=True, help="Output JSON path for examples."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of unique patents to sample (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
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
    create_example_set(
        input_path=args.input,
        output_path=args.output,
        n=args.n,
        seed=args.seed,
    )
