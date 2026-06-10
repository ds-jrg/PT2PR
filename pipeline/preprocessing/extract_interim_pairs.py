"""
Step 02: extracting patent-product pairs from raw extracted products.

Parses patent number mentions in product text fields (title, description,
features) and produces one record per (product_id, patent_number) pair,
with extracted country code, kind code, and the surrounding text span.

After this step, a manual review is performed (or awaited for new datasets).
The manually reviewed file corrects extraction errors: wrong country codes, missed numbers,
spurious pairs, and so on.
"""

import argparse
import logging
import re
import sys
from typing import Dict, List, Optional
from pipeline.utils.manual_review import REVIEW_STOP, apply_reviewed_changes
from pipeline.utils.io import read_jsonl, write_jsonl
from pipeline.utils.stats import compute_and_save_stats

logger = logging.getLogger(__name__)


_PATENT_KEYWORD = re.compile(
    r"(?i)\b(?:patent\s+numbers?|patent\s+no\.?s?|patent\s+num\.?s?|"
    r"pat\.\s*no\.?s?|pat\.\s*num\.?s?|patent|pat\.?s?(?=\W|$))"
)

_NUMBER_PLAIN = re.compile(r"([A-Z]{1,3})?(\d{5,14})([A-Z]\d?$)?", re.I)
_NUMBER_COMMA = re.compile(r"([A-Z]{1,3})?(\d{1,3}(?:,\d{3})+)([A-Z]\d?)?", re.I)

_US_HINT = re.compile(r"(?i)\b(?:US|U\.S\.|U\. S\.|U\.S)")
_KIND_CODE = re.compile(
    r"(?i)\s*(A\d?|A[1-4,8-9]|B\d?|B[1-4,8-9]|S\d?|S[1-2]|\.[X0-9])\b"
)

_SPAN_BEFORE = 200
_SPAN_AFTER = 200


def _normalize(text) -> str:
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"\s+", " ", text).strip()


def _flatten_field(value) -> str:
    if isinstance(value, list):
        return " ".join(_normalize(str(v)) for v in value if v)
    return _normalize(value or "")


def _apply_prefix_rules(
    prefix: str, number: str, suffix: str, kind_code: str
) -> Dict[str, str]:
    prefix = prefix.upper() if prefix else ""
    suffix = suffix or ""

    special_cases = {"ZL": ("CN", ""), "RE": ("US", "RE"), "D": ("US", "D")}
    if prefix in special_cases:
        country, kept_prefix = special_cases[prefix]
    elif len(prefix) == 3:
        country, kept_prefix = prefix[:2], prefix[2]
    elif len(prefix) == 2:
        country, kept_prefix = prefix, ""
    elif len(prefix) == 1:
        country, kept_prefix = "", prefix
    else:
        country, kept_prefix = "", ""

    patent_number = f"{kept_prefix}{number}{suffix}"
    if kind_code and kind_code.startswith("."):
        country = "CN"

    return {
        "patent_number": patent_number,
        "country_code": country,
        "kind_code": kind_code,
    }


def _extract_numbers(text: str, pattern) -> List:
    results = []
    for match in pattern.finditer(text):
        prefix, number, suffix = match.groups(default="")
        number = number.replace(",", "")
        after = text[match.end() : match.end() + 5]
        kind_match = _KIND_CODE.match(after)
        kind_code = kind_match.group(1).upper() if kind_match else ""
        results.append((prefix, number, suffix, kind_code))
    return results


def _extract_patents_from_span(span: str) -> List[Dict[str, str]]:
    results = []
    for prefix, number, suffix, kind_code in _extract_numbers(span, _NUMBER_COMMA):
        results.append(_apply_prefix_rules(prefix, number, suffix, kind_code))
    for prefix, number, suffix, kind_code in _extract_numbers(span, _NUMBER_PLAIN):
        results.append(_apply_prefix_rules(prefix, number, suffix, kind_code))
    return results


def _process_product(record: Dict) -> List[Dict]:
    product_id = str(record.get("product_id", ""))
    seen = set()
    results = []

    text_fields = [
        _flatten_field(record.get(f, "")) for f in ("title", "description", "features")
    ]

    for text in text_fields:
        for match in _PATENT_KEYWORD.finditer(text):
            start = max(0, match.start() - _SPAN_BEFORE)
            end = min(len(text), match.end() + _SPAN_AFTER)
            span = text[start:end].strip()

            before_text = text[max(0, match.start() - 20) : match.start()]
            has_us_hint = bool(_US_HINT.search(before_text)) or bool(
                _US_HINT.search(span)
            )

            patents = _extract_patents_from_span(span)

            if patents:
                for pat in patents:
                    country = pat["country_code"] or ("US" if has_us_hint else "")
                    key = (product_id, pat["patent_number"])
                    if key not in seen:
                        seen.add(key)
                        results.append(
                            {
                                "product_id": product_id,
                                "patent_number": pat["patent_number"],
                                "country_code": country,
                                "kind_code": pat["kind_code"],
                                "span_text": span,
                            }
                        )
            else:
                results.append(
                    {
                        "product_id": product_id,
                        "patent_number": "",
                        "country_code": "US" if has_us_hint else "",
                        "kind_code": "",
                        "span_text": span,
                    }
                )

    return results


def extract_interim_pairs(
    input_path: str,
    output_path: str,
    reviewed_file_path: Optional[str] = None,
) -> List[Dict]:
    """
    Extract (product_id, patent_number, country_code, kind_code, span_text) pairs
    from the patented products JSONL, apply manual corrections, and
    write the result.
    """
    records = list(read_jsonl(input_path))
    logger.info(f"Loaded {len(records)} products from '{input_path}'.")

    pairs: List[Dict] = []
    for record in records:
        pairs.extend(_process_product(record))

    logger.info(f"Extracted {len(pairs)} initial pairs.")

    # Apply manual review changes if a reviewed file path is provided
    if reviewed_file_path:
        result = apply_reviewed_changes(
            pairs, reviewed_file_path, step_label="02_extract_interim_pairs"
        )
        if result is REVIEW_STOP:
            logger.error(
                "\n"
                "MANUAL REVIEW REQUIRED: Step 02\n"
                f"Intermediate output written to: {output_path}\n"
                f"Expected reviewed file: {reviewed_file_path}\n\n"
                "Please review the intermediate output, make corrections,\n"
                "and save the corrected file as the reviewed file at the path\n"
                "above. See README.md for instructions.\n"
                "Then re-run the pipeline.\n"
            )
            # Still write intermediate output so the user has something to review
            write_jsonl(pairs, output_path)
            sys.exit(1)

        # Save pre-review stats so the automatic extraction is also recorded
        compute_and_save_stats(
            pairs,
            output_path,
            step="02_extract_interim_pairs_auto",
            stats_suffix="_auto_stats",
        )
        pairs = result

    write_jsonl(pairs, output_path)
    compute_and_save_stats(pairs, output_path, step="02_extract_interim_pairs")
    logger.info(f"Wrote {len(pairs)} pairs to '{output_path}'.")
    return pairs


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step 02: extracting patent-product pairs from patented products JSONL."
    )
    parser.add_argument("--input", required=True, help="Step 01 output JSONL.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--manual-review",
        default=None,
        help="Path to manual reviewed patch JSONL. Omit to skip review application.",
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
    extract_interim_pairs(
        input_path=args.input,
        output_path=args.output,
        reviewed_file_path=args.manual_review,
    )
