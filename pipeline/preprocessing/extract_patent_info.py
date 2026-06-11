"""
Step 04: fetching full patent metadata from Google Patents.

For each unique patent in the pairs file, fetches bibliographic data, abstract,
description, claims, and image references. Results are cached in a local cached
database so the step is resumable and re-runnable without re-fetching.
"""

import argparse
import json
import logging
import re
import diskcache
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from ftfy import fix_text
from tqdm import tqdm
from pipeline.utils.io import read_jsonl
from pipeline.utils.stats import compute_and_save_stats

logger = logging.getLogger(__name__)


_DEFAULT_BATCH_SIZE = 300
_DEFAULT_MAX_WORKERS = 5
_DEFAULT_REQUEST_DELAY = 2
_REQUEST_TIMEOUT = 20
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def _clean_text(text: str) -> Optional[str]:
    if not text:
        return None
    text = fix_text(text)
    text = text.replace("\xa0", " ").strip()
    return re.sub(r"\s+", " ", text)


def _normalize_patent_id(country: str, number: str, kind: str) -> str:
    return f"{country.upper()}{number}{kind.upper()}"


def _fetch_html(url: str) -> Tuple[str, Dict[str, Any]]:
    diag = {"url": url, "status_code": None, "ok": False, "error": None}
    try:
        resp = requests.get(
            url, headers={"User-Agent": _USER_AGENT}, timeout=_REQUEST_TIMEOUT
        )
        diag["status_code"] = resp.status_code
        if resp.status_code != 200:
            diag["error"] = f"HTTP {resp.status_code}"
            return "", diag
        diag["ok"] = True
        return resp.text, diag
    except requests.exceptions.RequestException as e:
        diag["error"] = f"{type(e).__name__}: {e}"
        return "", diag


def _remove_initial_full_claim(claims: List[str]) -> List[str]:
    if not claims or len(claims) == 1:
        return claims
    first = claims[0].lower()
    others = [c.lower() for c in claims[1:]]
    if all(o in first for o in others):
        return claims[1:]
    return claims


def _extract_patent_data(patent_id: str) -> Dict[str, Any]:
    url = f"https://patents.google.com/patent/{patent_id}/en"
    html, diag = _fetch_html(url)

    if not html:
        logger.warning(f"Failed to fetch {patent_id}: {diag.get('error')}")
        return {"error": "Failed to load Google Patents page", "diagnostics": diag}

    soup = BeautifulSoup(html, "html.parser")
    data: Dict[str, Any] = {
        "bibliographic": {},
        "abstract": "",
        "description": "",
        "claims": [],
        "images": [],
        "source": "google",
        "diagnostics": {"google_patents": diag},
    }

    # Bibliographic metadata
    for meta_name in ("DC.title", "DC.identifier", "DC.creator"):
        tag = soup.find("meta", {"name": meta_name})
        if tag and tag.get("content"):
            data["bibliographic"][meta_name] = tag["content"]

    for tag in soup.select("dd[itemprop], meta[itemprop]"):
        key = tag.get("itemprop")
        if key:
            value = (
                tag.get("content")
                if tag.name == "meta"
                else _clean_text(tag.get_text())
            )
            if value:
                data["bibliographic"][key] = value

    # Abstract
    abstract_tag = soup.find("div", {"itemprop": "abstract"})
    if abstract_tag:
        data["abstract"] = _clean_text(abstract_tag.get_text())
    elif meta := soup.find("meta", {"name": "DC.description"}):
        data["abstract"] = meta.get("content", "")

    # Description
    desc_tag = soup.find("div", {"itemprop": "description"})
    if desc_tag:
        data["description"] = _clean_text(desc_tag.get_text())
    elif fallback := soup.select_one("section#description, div.description"):
        data["description"] = fallback.get_text(" ", strip=True)

    # Claims
    claims: List[str] = []
    claims_section = soup.find("section", {"itemprop": "claims"})
    if claims_section:
        raw_claims = claims_section.find_all(["claim-text", "div"], recursive=True)

        for c in raw_claims:
            txt = _clean_text(c.get_text())
            if not txt or len(txt) < 10:
                continue
            if "what is claimed" in txt.lower():
                continue
            if re.match(r"^(claim\s*)?\d+[\.\:\-]\s+", txt, re.IGNORECASE):
                claims.append(txt)

        seen: set = set()
        claims = [cl for cl in claims if not (cl in seen or seen.add(cl))]
        claims = _remove_initial_full_claim(claims)

        if claims:
            structured = []
            for c in claims:
                num_m = re.match(r"^(claim\s*)?(\d+)", c, re.IGNORECASE)
                claim_number = int(num_m.group(2)) if num_m else None
                body = re.sub(r"^(claim\s*)?\d+[\.\:\-]\s*", "", c, flags=re.IGNORECASE)
                dep_m = re.search(r"claim\s+(\d+)", body, re.IGNORECASE)
                depends_on = int(dep_m.group(1)) if dep_m else None
                structured.append(
                    {
                        "text": c,
                        "claim_number": claim_number,
                        "dependent": depends_on is not None,
                        "depends_on": depends_on,
                    }
                )
            data["claims"] = structured
        else:
            full_text = _clean_text(claims_section.get_text())
            if full_text:
                lines = [l.strip() for l in full_text.splitlines() if l.strip()]
                s: set = set()
                lines = [l for l in lines if not (l in s or s.add(l))]
                data["claims"] = [
                    {
                        "text": l,
                        "claim_number": None,
                        "dependent": False,
                        "depends_on": None,
                    }
                    for l in lines
                ]

    # Images
    for img in soup.select("img"):
        src = img.get("src")
        if src and "patent" in src.lower():
            data["images"].append(src)

    return data


def _cached_fetch(patent_id: str, cache: diskcache.Cache) -> Dict[str, Any]:
    if patent_id in cache:
        return cache[patent_id]
    data = _extract_patent_data(patent_id)
    cache[patent_id] = data
    return data


def _extract_record(record: Dict, cache: diskcache.Cache) -> Dict:
    country = record.get("country_code", "")
    number = record.get("patent_number", "")
    kind = record.get("kind_code", "")

    if not (country and number and kind):
        record["patent_data"] = None
        record.pop("span_text", None)
        return record

    patent_id = _normalize_patent_id(country, number, kind)
    try:
        record["patent_data"] = _cached_fetch(patent_id, cache)
    except Exception as e:
        logger.error(f"Failed to extract {patent_id}: {e}")
        record["patent_data"] = None

    record.pop("span_text", None)
    return record


def _process_batch(
    batch: List[Dict], cache: diskcache.Cache, max_workers: int
) -> List[Dict]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_record, rec, cache): rec for rec in batch}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                rec = futures[future]
                logger.error(f"Batch error for {rec.get('patent_number')}: {e}")
                rec["patent_data"] = None
                rec.pop("span_text", None)
                results.append(rec)
    return results


def extract_patent_info(
    input_path: str,
    output_path: str,
    cache_path: str,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    request_delay: float = _DEFAULT_REQUEST_DELAY,
) -> List[Dict]:
    """Add full patent metadata fetched from Google Patents to the dataset."""
    records = list(read_jsonl(input_path))
    total = len(records)
    logger.info(f"Loaded {total} records from '{input_path}'.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    all_extracted: List[Dict] = []

    with diskcache.Cache(cache_path) as cache, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for i in tqdm(range(0, total, batch_size), desc="Fetching patent data"):
            batch = records[i : i + batch_size]
            extracted_batch = _process_batch(batch, cache, max_workers)

            for rec in extracted_batch:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            all_extracted.extend(extracted_batch)

            if i + batch_size < total:
                time.sleep(request_delay)

    compute_and_save_stats(all_extracted, output_path, step="04_extract_patent_info")
    logger.info(f"Wrote {len(all_extracted)} records to '{output_path}'.")
    return all_extracted


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step 04: fetching full patent metadata from Google Patents."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Step 03 output JSONL (manually checked pairs with kind codes).",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--cache",
        required=True,
        help="Path to the diskcache directory (e.g. data/interim/patent_cache).",
    )
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=_DEFAULT_MAX_WORKERS)
    parser.add_argument("--request-delay", type=float, default=_DEFAULT_REQUEST_DELAY)
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
    extract_patent_info(
        input_path=args.input,
        output_path=args.output,
        cache_path=args.cache,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        request_delay=args.request_delay,
    )
