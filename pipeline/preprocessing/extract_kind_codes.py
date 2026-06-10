"""
Step 03: fetching kind codes from Google Patents.

Kind codes (A1, B1, B2, …) identify the publication stage of a patent and are
required by EPO and USPTO APIs downstream. This step fetches them by scraping
the Google Patents page for each unique (country, patent_number) pair.

After this step, a manual review is performed (or awaited for new
datasets). The reviewed changes correct wrong kind codes and remove pairs where
alignment between the patent and the product has been verified as spurious.
"""

import argparse
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from pipeline.utils.manual_review import REVIEW_STOP, apply_reviewed_changes
from pipeline.utils.io import read_jsonl, write_jsonl
from pipeline.utils.stats import compute_and_save_stats

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 300
_DEFAULT_MAX_WORKERS = 5
_REQUEST_TIMEOUT = 20
_RETRY_DELAY = 2
_MAX_RETRIES = 3
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def _parse_pubstring(pubstr: str) -> Optional[Tuple[str, str, str]]:
    if not pubstr:
        return None
    s = pubstr.strip().replace(" ", "").upper()
    m = re.search(r"\b([A-Z]{1,3})(\d{4,})([A-Z]\d{0,2})\b", s)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def _extract_kind_from_html(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    if title:
        parsed = _parse_pubstring(title)
        if parsed:
            return parsed[2]
    return None


def _fetch_kind_code(
    patent_number: str,
    country: str,
    retry_count: int = 0,
) -> Optional[str]:
    google_id = f"{country}{patent_number}".replace(".", "")
    url = f"https://patents.google.com/patent/{google_id}"
    headers = {"User-Agent": _USER_AGENT}

    try:
        resp = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)

        if resp.status_code == 200:
            return _extract_kind_from_html(resp.text)

        if resp.status_code == 429 and retry_count < _MAX_RETRIES:
            wait = _RETRY_DELAY * (2**retry_count)
            logger.warning(f"Rate limited for {google_id}; retrying in {wait}s.")
            time.sleep(wait)
            return _fetch_kind_code(patent_number, country, retry_count + 1)

        logger.warning(f"HTTP {resp.status_code} for {google_id}.")
        return None

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout for {google_id}.")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request error for {google_id}: {e}")
        return None


def _enrich_record(record: Dict) -> Dict:
    patent_number = record.get("patent_number")
    country = record.get("country_code")

    if not patent_number or not country:
        record["kind_code"] = record.get("kind_code") or None
        return record

    # Always fetch from Google Patents to get the authoritative kind code,
    # even if a kind code was already extracted from the text span in step 02.
    # Step 03 is the verification/correction step.
    kind_code = _fetch_kind_code(patent_number, country)
    record["kind_code"] = kind_code
    logger.debug(
        f"{'OK' if kind_code else 'MISS'} {country}{patent_number} -> {kind_code}"
    )
    return record


def _process_batch(batch: List[Dict], max_workers: int) -> List[Dict]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_enrich_record, rec): rec for rec in batch}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                rec = futures[future]
                logger.error(f"Batch error for {rec.get('patent_number')}: {e}")
                rec["kind_code"] = None
                results.append(rec)
    return results


def extract_kind_codes(
    input_path: str,
    output_path: str,
    reviewed_file_path: Optional[str] = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> List[Dict]:
    """
    Fetch kind codes for all patent-product pairs that are missing them,
    apply the manual changes, and write results.
    """
    records = list(read_jsonl(input_path))
    logger.info(f"Loaded {len(records)} pairs from '{input_path}'.")

    # Process in batches
    enriched: List[Dict] = []
    total = len(records)
    succeeded = 0

    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        logger.info(
            f"Processing batch {i // batch_size + 1} "
            f"({i + 1} - {min(i + batch_size, total)} of {total})"
        )
        batch_result = _process_batch(batch, max_workers)
        enriched.extend(batch_result)
        succeeded += sum(1 for r in batch_result if r.get("kind_code"))

    logger.info(
        f"Kind code fetch complete: {succeeded}/{total} records have a kind code."
    )

    # Apply manual review
    if reviewed_file_path:
        result = apply_reviewed_changes(
            enriched, reviewed_file_path, step_label="03_extract_kind_codes"
        )
        if result is REVIEW_STOP:
            logger.error(
                "\n"
                "MANUAL REVIEW REQUIRED: Step 03\n"
                f"Intermediate output written to: {output_path}\n"
                f"Expected reviewed file: {reviewed_file_path}\n\n"
                "Review the intermediate output, save the corrected file as\n"
                "the reviewed file at the path above, then re-run the pipeline.\n"
            )
            write_jsonl(enriched, output_path)
            sys.exit(1)

        # Save pre-review stats so the automatic fetch results are also recorded
        compute_and_save_stats(
            enriched,
            output_path,
            step="03_extract_kind_codes_auto",
            stats_suffix="_auto_stats",
        )
        enriched = result

    write_jsonl(enriched, output_path)
    compute_and_save_stats(enriched, output_path, step="03_extract_kind_codes")
    logger.info(f"Wrote {len(enriched)} records to '{output_path}'.")
    return enriched


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step 03: fetching kind codes from Google Patents."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Step 02 output JSONL (manually checked initial pairs).",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--manual-review",
        default=None,
        help="Path to manual reviewed patch JSONL. Omit to skip review application.",
    )
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=_DEFAULT_MAX_WORKERS)
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
    extract_kind_codes(
        input_path=args.input,
        output_path=args.output,
        reviewed_file_path=args.manual_review,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )
