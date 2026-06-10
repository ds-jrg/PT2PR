"""
Masking for patent-product pairs operating on the preprocessed datasets.
"""

import json
import re
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any
from rapidfuzz import fuzz as _fuzz
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _STOPWORDS


# Character-level similarity threshold for fuzzy candidate matching.
# Applied after the corpus-frequency filter, so the NER gate from the
# previous approach is replaced by the distinctiveness filter here.
_FUZZY_THRESHOLD = 70

# Tokens appearing in more than this fraction of corpus documents are
# considered too common to be distinctive entity identifiers.
_DOC_FREQ_THRESHOLD = 0.10

# Stricter threshold for single-word candidates from multi-word brand names.
# "Night" from "Night Flyer" or "buckle" from "Buckle-Down" are common English
# words, so they must not match freely across text. Tokens appearing in more than
# 2% of corpus documents are excluded from single-word candidacy when the source
# brand name has more than one word component.
_SINGLE_TOKEN_FREQ_THRESHOLD = 0.02

_COMMON_TOKENS: set[str] = set()
_TOKEN_DOC_FREQS: dict[str, float] = {}


def _build_token_stats(texts: list[str]) -> None:
    """
    Compute per-token document frequency across the corpus.
    Tokens appearing in more than _DOC_FREQ_THRESHOLD of documents land in
    _COMMON_TOKENS (filtered from all candidates).  All per-token frequencies
    are stored in _TOKEN_DOC_FREQS for the stricter single-token candidacy
    check in _generate_candidates.
    """
    global _COMMON_TOKENS, _TOKEN_DOC_FREQS
    n = len(texts)
    if n == 0:
        return
    doc_freq: Counter = Counter()
    for t in texts:
        tokens = set(re.findall(r"[A-Za-z0-9]+", t.lower()))
        doc_freq.update(tokens)
    _TOKEN_DOC_FREQS = {tok: c / n for tok, c in doc_freq.items()}
    _COMMON_TOKENS = {
        tok for tok, freq in _TOKEN_DOC_FREQS.items() if freq > _DOC_FREQ_THRESHOLD
    }


def _is_noise_token(tok: str) -> bool:
    tl = tok.lower()
    return tl in _STOPWORDS or tl in _COMMON_TOKENS


PATENT_MASK = (
    ""  # No placeholder text to avoid introducing shared tokens across documents
)
ENTITY_MASK = ""

_PATENT_KEYWORD_STR = (
    r"(?:"
    r"patents?\s+numbers?"  # patent/patents number/numbers
    r"|patents?\s+num\.?s?"  # patent/patents num/nums
    r"|patents?\s+nos?\.?"  # patent/patents no/nos.
    r"|pats?\.?\s*nos?\.?"  # pat./pats. no/nos.
    r"|pats?\.?\s*num\.?s?"  # pat./pats. num/nums
    r"|U\.?\s*S\.?\s*Pats?\.?\s*Nos?\.?"  # U.S. Pat./Pats. No/Nos.
    r"|U\.?\s*S\.?\s*Patents?\s+Nos?\.?"  # U.S. Patent/Patents Nos.
    r"|Ser\.?\s*Nos?\.?"  # Ser. No/Nos.
    r"|applications?\s+Ser\.?\s*Nos?\.?"  # Application(s) Ser. No.
    r"|Reg\.?\s*Des\.?\s*Nos?\.?"  # Reg. Des. No.
    r"|Reg\.?\s*Nos?\.?"  # Reg. No. (trademark / design registration)
    r"|(?:provisional\s+)?patents?\s+applications?\s+nos?\.?"  # (Provisional) Patent(s) Application(s) No.
    r"|(?:pat(?:ent)?s?\.?\s+)?app(?:lication)?s?\.?\s+pub(?:lication)?s?\.?\s*nos?\.?"  # Pat(s). App(s). Pub(s). No.
    r"|pub(?:lication)?s?\.?\s*nos?\.?"  # Pub./Pubs./Publication(s). No.
    r"|app(?:lication)?s?\.?\s*nos?\.?"  # App./Apps./Application(s). No.
    r"|designs?\s+nos?\.?"  # Design(s) No.
    r"|patented"  # "Patented D772596"
    r"|designs?\s+patents?"  # "Design Patent(s)"
    r"|patents?"  # kept last so specific forms match first
    r")"
)
_SEP = r"[,\.]"  # space-only separator removed; \s* after sep handles "6, 585, 212"
_NUM_WITH_SEP = r"\d{1,3}(?:" + _SEP + r"\s*\d{3})+"
_NUM_PLAIN = r"\d{5,14}"
_NUM_SLASH = r"\d{2,4}/\d{3,9}(?:[,\.]\s?\d{3})?"
_NUM_HYPHEN = r"[A-Z]{2}\s*\d[\d\s\-]{4,20}\d"
_KIND = (
    r"(?:"
    r"\s+(?-i:[A-Z]{1,2})\d{0,2}(?!\w)"  # space-preceded UPPERCASE kind: " B1", " S2"
    r"|[A-Z]{1,2}\d{0,2}"  # attached kind (case-insensitive via flag)
    r"|\.[A-Z0-9]{1,2}"  # dot-kind: .B, .X3
    r"|-\d{1,6}(?:/\d+)?"  # hyphen-suffix: -0001/2
    r")?"
)
_COUNTRY_OPT = r"(?:[A-Z]{1,3}\s?)?"
_DESIGN = r"D(?:\d{6,7}|\d{1,3}[,.]\s*\d{3})"

_ANY_NUM_TOKEN = (
    r"(?:"
    + "|".join([_NUM_HYPHEN, _NUM_SLASH, _NUM_WITH_SEP, _NUM_PLAIN, _DESIGN])
    + r")"
)
_NUM_SPACE_HYP_OUTER = r"\d+(?:[\s\-]\d+){1,6}"
_ANY_NUM_TOKEN_SCOPE = (
    r"(?:"
    + "|".join(
        [
            _NUM_HYPHEN,
            _NUM_SLASH,
            _NUM_WITH_SEP,
            _NUM_SPACE_HYP_OUTER,
            _NUM_PLAIN,
            _DESIGN,
        ]
    )
    + r")"
)
_KEYWORD_SCOPE_RE = re.compile(
    r"\b" + _PATENT_KEYWORD_STR
    # Zero-or-more separators: handles ", &" (comma then ampersand), "& ," etc.
    + r"(?:\s*(?:(?:[,;:|&]|\band\b|nos?\.)\s*)*"  # any number of separator chars
    + r"(?:Nr\.?\s*)?"  # optional German Nr. prefix
    + r"#?"  # optional hash prefix (#5123456)
    + _COUNTRY_OPT
    + _ANY_NUM_TOKEN_SCOPE  # includes space-separated (Germany)
    # Kind code: (?!\w) prevents consuming the start of regular words like "Better"
    + r"(?:\s+(?-i:[A-Z]{1,2})\d{0,2}(?!\w)"  # space + UPPERCASE kind: " B2", " S1"
    + r"|(?-i:[A-Z]{1,2})\d{0,2}(?!\w)"  # attached UPPERCASE kind: "9,474,336B2"
    + r"|\.(?-i:[A-Z0-9])(?![a-z])"  # dot-kind: .X .3, 1 char, blocks .Ot
    + r"|-\d{1,6}(?:/\d+)?"  # hyphen-suffix: -0001 or -0001/2
    + r")?"
    + r"){1,30}",
    re.IGNORECASE,
)
_NUM_SCOPE_TOKEN_RE = re.compile(
    r"(?<!\w)#?" + _COUNTRY_OPT + _ANY_NUM_TOKEN + _KIND + r"(?!\w)",
    re.IGNORECASE,
)
_NUM_SCOPE_LOOSE_RE = re.compile(
    r"(?<!\w)" + _COUNTRY_OPT + r"(?:\d+(?:[\s\-]\d+){1,6})" + r"(?!\w)",
    re.IGNORECASE,
)

# Global patterns
_GLOBAL_DESIGN_COUNTRY_RE = re.compile(
    r"(?<!\w)U\.?S\.?\s+" + _DESIGN + _KIND + r"(?!\w)",
)
_GLOBAL_US_DESIGN_PADDED_RE = re.compile(
    r"(?<!\w)US\d{0,2}" + _DESIGN + _KIND + r"(?!\w)",
)
_GLOBAL_EP_WO_RE = re.compile(
    r"(?<!\w)[EW][PO]\s?\d{6,10}" + _KIND + r"(?!\w)",
)
_GLOBAL_DESIGN_RE = re.compile(
    r"(?<!\w)" + _DESIGN + r"(?:[A-Za-z]\d{0,2})?(?!\w)",
)
_GLOBAL_USD_DESIGN_RE = re.compile(
    r"(?<!\w)USD(?:\d{6,7}|\d{1,3}(?:[,.]\s*\d{3})+)(?:[A-Za-z]\d{0,2})?(?!\w)",
)
_GLOBAL_HASH_RE = re.compile(
    r"(?<!\w)#\s*(?:[A-Za-z]{0,2}\s?)?(?:\d{1,3}(?:[,.]\s*\d{3})+|\d{5,14})(?:[A-Za-z]\d{0,2})?(?!\w)",
)
_PRODUCT_NUM_RE = re.compile(
    r"\b"
    r"(?:design\s+)?(?:patent|trademark)"
    r"(?:\s+\w+){0,5}?"
    r"\s+numbers?\s+(?:is\s+)?"
    + r"(?:"
    + _ANY_NUM_TOKEN_SCOPE
    + r")"
    + r"(?:\s*(?:[,&]|\band\b)\s*(?:"
    + _ANY_NUM_TOKEN_SCOPE
    + r"))*",
    re.IGNORECASE,
)
_PRODUCT_NUM_INNER_RE = re.compile(
    r"(?<!\w)" + _ANY_NUM_TOKEN_SCOPE + r"(?!\w)", re.IGNORECASE
)


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def _comma_format(digits: str) -> str:
    if len(digits) <= 3:
        return digits
    result, i = "", len(digits)
    while i > 0:
        start = max(0, i - 3)
        result = digits[start:i] + ("," if result else "") + result
        i = start
    return result


def _dot_format(digits: str) -> str:
    return _comma_format(digits).replace(",", ".")


def _space_format(digits: str) -> str:
    return _comma_format(digits).replace(",", ", ")


def _build_known_number_re(raw: str) -> re.Pattern | None:
    raw = raw.strip()
    if not raw:
        return None

    if "/" in raw:
        parts = raw.split("/", 1)
        dl = _digits_only(parts[0])
        dr = _digits_only(parts[1])
        if dl and dr:
            alts = "|".join(
                sorted(
                    {
                        re.escape(f"{dl}/{dr}"),
                        re.escape(f"{dl}/{_comma_format(dr)}"),
                        re.escape(dl + dr),
                    },
                    key=len,
                    reverse=True,
                )
            )
            try:
                return re.compile(
                    r"(?<!\w)(?:[A-Z]{1,3}\s?)?(?:"
                    + alts
                    + r")(?:[A-Za-z]\d{0,2})?(?!\w)",
                    re.IGNORECASE,
                )
            except re.error:
                return None

    digits = _digits_only(raw)
    if len(digits) < 5:
        return None

    surface_forms = {
        digits,
        _comma_format(digits),
        _dot_format(digits),
        _space_format(digits),
    }
    alts = "|".join(
        sorted({re.escape(f) for f in surface_forms}, key=len, reverse=True)
    )

    try:
        return re.compile(
            r"(?<!\w)(?:[A-Z]{1,3}\s?)?(?:" + alts + r")(?:[A-Za-z]\d{0,2})?(?!\w)",
            re.IGNORECASE,
        )
    except re.error:
        return None


def build_patent_number_patterns(biblio: dict) -> list[re.Pattern]:
    raw_numbers: set[str] = set()
    for key in ("publicationNumber", "numberWithoutCodes", "applicationNumber"):
        val = biblio.get(key)
        if val:
            raw_numbers.add(val.strip())
    patterns = []
    for raw in raw_numbers:
        p = _build_known_number_re(raw)
        if p:
            patterns.append(p)
    return patterns


_PLACEHOLDER_NAMES = {"individual", "unknown", "n/a", ""}


def parse_name_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = re.split(r";|\band\b", raw, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def collect_entity_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        name = name.strip()
        if not name or name.lower() in _PLACEHOLDER_NAMES:
            continue
        if name.lower() not in seen:
            seen.add(name.lower())
            result.append(name)
    return sorted(result, key=len, reverse=True)


def get_patent_entity_names(biblio: dict) -> list[str]:
    raw: list[str] = []
    for key in ("inventor", "assigneeCurrent", "assigneeOriginal"):
        raw.extend(parse_name_list(biblio.get(key)))
    return collect_entity_names(raw)


def get_product_entity_names(product: dict) -> list[str]:
    raw: list[str] = []
    details = product.get("details", {})
    for key in ("Brand", "Manufacturer"):
        val = details.get(key)
        if val and isinstance(val, str):
            raw.append(val)
    store = product.get("store")
    if store and isinstance(store, str):
        raw.append(store)
    return collect_entity_names(raw)


def _replace_keyword_scoped(text: str) -> str:
    def _replace_span(m: re.Match) -> str:
        span = _NUM_SCOPE_TOKEN_RE.sub(PATENT_MASK, m.group())
        span = _NUM_SCOPE_LOOSE_RE.sub(PATENT_MASK, span)
        return span

    return _KEYWORD_SCOPE_RE.sub(_replace_span, text)


def mask_patent_numbers(text: str, known_patterns: list[re.Pattern]) -> str:
    # keyword-scoped
    text = _replace_keyword_scoped(text)
    # product-listing style: "patent/trademark number is XXXXXXX [and XXXXXXX]"
    text = _replace_product_citations(text)
    # targeted global patterns safe enough to run unconditionally
    text = _GLOBAL_DESIGN_COUNTRY_RE.sub(PATENT_MASK, text)
    text = _GLOBAL_US_DESIGN_PADDED_RE.sub(PATENT_MASK, text)
    text = _GLOBAL_USD_DESIGN_RE.sub(PATENT_MASK, text)
    text = _GLOBAL_HASH_RE.sub(PATENT_MASK, text)
    text = _GLOBAL_EP_WO_RE.sub(PATENT_MASK, text)
    text = _GLOBAL_DESIGN_RE.sub(PATENT_MASK, text)
    # known numbers from bibliographic data
    for pat in known_patterns:
        text = pat.sub(PATENT_MASK, text)
    return text


def _replace_product_citations(text: str) -> str:
    """Finds all number tokens within each match and replaces them."""

    def _sub(m: re.Match) -> str:
        return _PRODUCT_NUM_INNER_RE.sub(PATENT_MASK, m.group())

    return _PRODUCT_NUM_RE.sub(_sub, text)


def _name_variants(name: str) -> list[str]:
    """
    Return the name plus a no-space concatenated variant for multi-word names.
    Handles brand names that appear without spaces in product text, e.g.
    "KneeGuard Kids" also matches "KneeGuardKids".
    """
    words = name.split()
    if len(words) < 2:
        return [name]
    return [name, "".join(words)]


def _generate_candidates(name: str) -> list[str]:
    """
    Generate word-level n-gram candidates from an entity name, keeping only
    those built entirely from distinctive tokens (not stopwords, not corpus-
    common). Longer candidates are returned first so more specific matches
    are attempted before shorter sub-spans.
    """
    tokens = re.findall(r"[A-Za-z0-9]+", name)
    is_multiword = len(tokens) > 1
    filtered = [t for t in tokens if not _is_noise_token(t)]
    if not filtered:
        return []
    candidates: set[str] = set()
    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered) + 1):
            span_tokens = filtered[i:j]
            if len(span_tokens) == 1 and is_multiword:
                tok = span_tokens[0].lower()
                # Single-word candidates from multi-word brand names require
                # the token to appear in fewer than _SINGLE_TOKEN_FREQ_THRESHOLD
                # of corpus documents. This rarity criterion is the sole gate:
                # a token that is genuinely distinctive will be rare regardless
                # of its capitalisation.
                if _TOKEN_DOC_FREQS.get(tok, 0.0) > _SINGLE_TOKEN_FREQ_THRESHOLD:
                    continue
            candidates.add(" ".join(span_tokens))
    return sorted(candidates, key=len, reverse=True)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent character spans into their union."""
    if not spans:
        return []
    merged = [min(spans, key=lambda s: s[0])]
    for start, end in sorted(spans):
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _fuzzy_mask_candidates(text: str, entity_names: list[str]) -> str:
    """
    For each entity name, generate filtered candidate anchors and slide same-
    length windows over the text, masking windows whose character-level
    similarity to the candidate meets _FUZZY_THRESHOLD.
    """
    word_matches = list(re.finditer(r"[A-Za-z0-9]+", text))
    if not word_matches:
        return text

    words = [m.group() for m in word_matches]
    offsets = [(m.start(), m.end()) for m in word_matches]
    spans_to_mask: list[tuple[int, int]] = []

    for name in entity_names:
        for candidate in _generate_candidates(name):
            cand_words = candidate.split()
            n = len(cand_words)
            if n > len(words):
                continue
            for i in range(len(words) - n + 1):
                window = " ".join(words[i : i + n])
                if _fuzz.ratio(candidate.lower(), window.lower()) >= _FUZZY_THRESHOLD:
                    spans_to_mask.append((offsets[i][0], offsets[i + n - 1][1]))

    # Merge overlapping spans before applying so that offset corruption cannot
    # occur when a shorter candidate matches inside a longer candidate's span
    for start, end in sorted(_merge_spans(spans_to_mask), reverse=True):
        text = text[:start] + ENTITY_MASK + text[end:]

    return text


def mask_entities(text: str, entity_names: list[str]) -> str:
    # Longest names first so "Omelia Systems Inc." is masked before "Omelia"
    entity_names = sorted(entity_names, key=len, reverse=True)

    # Expand with no-space variants before exact matching so concatenated brand
    # names are caught in Pass 1 (e.g. "KneeGuard Kids" -> also "KneeGuardKids")
    seen: set[str] = set()
    names_for_exact: list[str] = []
    for name in entity_names:
        for variant in _name_variants(name):
            if variant.lower() not in seen:
                seen.add(variant.lower())
                names_for_exact.append(variant)
    names_for_exact.sort(key=len, reverse=True)

    # Pass 1: exact match, case-insensitive, word-boundary checked
    for name in names_for_exact:
        if not name:
            continue
        pattern = re.compile(
            r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        text = pattern.sub(ENTITY_MASK, text)

    # Pass 2: corpus-frequency-aware fuzzy matching catches partial names and
    # abbreviations that exact matching misses even after no-space expansion
    if entity_names:
        text = _fuzzy_mask_candidates(text, entity_names)

    return text


def _normalize_title(title: Any) -> str:
    """Collapse internal whitespace and strip. Handles list-valued titles."""
    if isinstance(title, list):
        title = " ".join(str(t) for t in title if t)
    return re.sub(r"\s+", " ", str(title)).strip()


def mask_patent_titles(text: str, patent_titles: list[str]) -> str:
    """
    Mask patent titles appearing verbatim in product text.
    Uses normalised exact matching: case-insensitive, whitespace-collapsed.
    """
    for title in patent_titles:
        normalized = _normalize_title(title)
        if not normalized:
            continue
        # Split first, escape each word, then join with \\s+ so minor
        # whitespace differences in the source text are tolerated
        words = [re.escape(w) for w in re.split(r"\s+", normalized)]
        pattern = re.compile(
            r"(?<![A-Za-z0-9])" + r"\s+".join(words) + r"(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        text = pattern.sub(PATENT_MASK, text)
    return text


def _apply_title_masking_to_field(field: Any, patent_titles: list[str]) -> Any:
    """Apply patent title masking to a field that may be str or list of str."""
    if isinstance(field, str):
        return _cleanup(mask_patent_titles(field, patent_titles))
    if isinstance(field, list):
        return [
            _cleanup(mask_patent_titles(item, patent_titles))
            if isinstance(item, str)
            else item
            for item in field
        ]
    return field


def _cleanup(text: str) -> str:
    # Collapse multiple spaces left by deletions
    text = re.sub(r" {2,}", " ", text)
    # Remove space that appeared before punctuation after a deletion
    # e.g. "developed by ," -> "developed by,"
    # This is a simplified approach, doesn't cover all cases but works for common patterns
    text = re.sub(r" ([,;:.!?])", r"\1", text)
    return text.strip()


def mask_text(
    text: str, known_patterns: list[re.Pattern], entity_names: list[str]
) -> str:
    text = mask_patent_numbers(text, known_patterns)
    text = mask_entities(text, entity_names)
    return _cleanup(text)


def mask_field(
    field: Any, known_patterns: list[re.Pattern], entity_names: list[str]
) -> Any:
    if isinstance(field, str):
        return mask_text(field, known_patterns, entity_names)
    if isinstance(field, list):
        return [
            mask_text(item, known_patterns, entity_names)
            if isinstance(item, str)
            else item
            for item in field
        ]
    return field


def mask_claims(
    claims: list[dict],
    known_patterns: list[re.Pattern],
    entity_names: list[str],
) -> list[dict]:
    result = []
    for claim in claims:
        text = claim.get("text")

        new_claim = {}

        if isinstance(text, str):
            new_claim["text"] = text
            new_claim["text_masked"] = mask_text(text, known_patterns, entity_names)

        for key in ("claim_number", "dependent", "depends_on"):
            if key in claim:
                new_claim[key] = claim[key]

        result.append(new_claim)
    return result


def build_cross_pair_indexes(
    mappings: list[dict],
    patents: dict,
    products: dict,
) -> tuple[
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, list[re.Pattern]],
    dict[str, list[re.Pattern]],
    dict[str, list[str]],
]:
    patent_to_products: dict[str, set[str]] = defaultdict(set)
    product_to_patents: dict[str, set[str]] = defaultdict(set)
    for m in mappings:
        product_id = m.get("product_id")
        pnum = m.get("patent_number")
        if product_id and pnum:
            patent_to_products[pnum].add(product_id)
            product_to_patents[product_id].add(pnum)

    # Per-patent known-number patterns (own only)
    patent_known_patterns: dict[str, list[re.Pattern]] = {}
    for pnum, patent_data in patents.items():
        biblio = patent_data.get("bibliographic", {})
        patent_known_patterns[pnum] = build_patent_number_patterns(biblio)

    # Per-patent entity list: own + paired products
    patent_entities: dict[str, list[str]] = {}
    for pnum, patent_data in patents.items():
        biblio = patent_data.get("bibliographic", {})
        raw: list[str] = list(get_patent_entity_names(biblio))
        for product_id in patent_to_products.get(pnum, []):
            raw.extend(get_product_entity_names(products.get(product_id, {})))
        patent_entities[pnum] = collect_entity_names(raw)

    # Per-product entity list, known-number patterns, and paired patent titles
    product_entities: dict[str, list[str]] = {}
    product_known_patterns: dict[str, list[re.Pattern]] = {}
    product_patent_titles: dict[str, list[str]] = {}
    for product_id, product_data in products.items():
        raw: list[str] = list(get_product_entity_names(product_data))
        patterns: list[re.Pattern] = []
        titles: list[str] = []
        for pnum in product_to_patents.get(product_id, []):
            raw.extend(
                get_patent_entity_names(patents.get(pnum, {}).get("bibliographic", {}))
            )
            patterns.extend(patent_known_patterns.get(pnum, []))
            title = _normalize_title(patents.get(pnum, {}).get("title", ""))
            if title:
                titles.append(title)
        product_entities[product_id] = collect_entity_names(raw)
        product_known_patterns[product_id] = patterns
        product_patent_titles[product_id] = titles

    return (
        patent_entities,
        product_entities,
        patent_known_patterns,
        product_known_patterns,
        product_patent_titles,
    )


def mask_patent_entry(
    patent_data: dict,
    known_patterns: list[re.Pattern],
    entity_names: list[str],
) -> dict:
    # Patent title masking is intentionally not applied here; the patent title
    # is legitimate content within the patent itself and must not be removed.
    new_entry = {}
    if "bibliographic" in patent_data:
        new_entry["bibliographic"] = patent_data["bibliographic"]

    for field in ("title", "abstract", "description"):
        original = patent_data.get(field)

        if original is not None:
            new_entry[field] = original
            new_entry[f"{field}_masked"] = mask_field(
                original, known_patterns, entity_names
            )

    claims = patent_data.get("claims")
    if isinstance(claims, list):
        new_entry["claims"] = mask_claims(claims, known_patterns, entity_names)
    return new_entry


def mask_product_entry(
    product_data: dict,
    known_patterns: list[re.Pattern],
    entity_names: list[str],
    patent_titles: list[str],
) -> dict:
    new_entry = {}
    for key in ("details", "store"):
        if key in product_data:
            new_entry[key] = product_data[key]

    for field in ("title", "features", "description"):
        original = product_data.get(field)

        if original is not None:
            new_entry[field] = original
            masked = mask_field(original, known_patterns, entity_names)
            # Patent title masking applied to product fields only
            if patent_titles:
                masked = _apply_title_masking_to_field(masked, patent_titles)
            new_entry[f"{field}_masked"] = masked
    return new_entry


def main(input_path: str, output_path: str) -> None:
    print(f"Loading {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    patents: dict = data["patents"]
    products: dict = data["products"]
    mappings: list = data["mappings"]

    print(f"Patents: {len(patents)}")
    print(f"Products: {len(products)}")
    print(f"Mappings: {len(mappings)}")

    print("\nBuilding corpus token statistics")
    all_texts = []
    for p in patents.values():
        parts = [p.get("title", ""), p.get("abstract", ""), p.get("description", "")]
        all_texts.append(
            " ".join(" ".join(v) if isinstance(v, list) else str(v) for v in parts if v)
        )
    for p in products.values():
        parts = [p.get("title", ""), p.get("features", ""), p.get("description", "")]
        all_texts.append(
            " ".join(" ".join(v) if isinstance(v, list) else str(v) for v in parts if v)
        )
    _build_token_stats(all_texts)
    print(f"Common tokens (doc_freq > {_DOC_FREQ_THRESHOLD}): {len(_COMMON_TOKENS)}")

    print("\nBuilding cross-pair indexes")
    (
        patent_entities,
        product_entities,
        patent_known_patterns,
        product_known_patterns,
        product_patent_titles,
    ) = build_cross_pair_indexes(mappings, patents, products)

    print("Masking patents")
    masked_patents: dict = {}
    for i, (pnum, patent_data) in enumerate(patents.items(), 1):
        masked_patents[pnum] = mask_patent_entry(
            patent_data,
            patent_known_patterns.get(pnum, []),
            patent_entities.get(pnum, []),
        )
        if i % 500 == 0:
            print(f"{i}/{len(patents)}")

    print("Masking products")
    masked_products: dict = {}
    for i, (product_id, product_data) in enumerate(products.items(), 1):
        masked_products[product_id] = mask_product_entry(
            product_data,
            product_known_patterns.get(product_id, []),
            product_entities.get(product_id, []),
            product_patent_titles.get(product_id, []),
        )
        if i % 1000 == 0:
            print(f"{i}/{len(products)}")

    output_data = {
        "patents": masked_patents,
        "products": masked_products,
        "mappings": mappings,
        "metadata": data.get("metadata", {}),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {output_path}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask patent numbers and entity names in preprocessed data."
    )
    parser.add_argument(
        "--dataset",
        choices=["amazon", "esci", "both"],
        required=True,
        help="Dataset to mask: 'amazon','esci', or both.",
    )
    parser.add_argument(
        "--setting",
        choices=["text", "multimodal", "both"],
        default="text",
        help=(
            "'text' masks the text preprocessed data (default); "
            "'multimodal' masks the multimodal preprocessed data, "
            "'both' masks datasets for both settings."
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Override the input path. By default, resolved automatically from "
            "--dataset and --setting as "
            "experiments/data/<dataset>/preprocessed_data.json (text) or "
            "experiments/data/<dataset>/preprocessed_multimodal_data.json (multimodal)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Override the output path. By default, resolved automatically from "
            "--dataset and --setting as "
            "experiments/data/<dataset>/masked_data.json (text) or "
            "experiments/data/<dataset>/masked_multimodal_data.json (multimodal)."
        ),
    )
    args = parser.parse_args()

    datasets = ["amazon", "esci"] if args.dataset == "both" else [args.dataset]
    settings = ["text", "multimodal"] if args.setting == "both" else [args.setting]

    for dataset in datasets:
        for setting in settings:
            if args.input:
                input_path = args.input
            elif setting == "multimodal":
                input_path = (
                    f"experiments/data/{dataset}/preprocessed_multimodal_data.json"
                )
            else:
                input_path = f"experiments/data/{dataset}/preprocessed_data.json"

            # Skip if input doesn't exist
            if not Path(input_path).exists():
                print(
                    f"Skipping {dataset}/{setting}: input file not found at {input_path}"
                )
                continue

            if args.output:
                output_path = args.output
            elif setting == "multimodal":
                output_path = f"experiments/data/{dataset}/masked_multimodal_data.json"
            else:
                output_path = f"experiments/data/{dataset}/masked_data.json"

            main(input_path, output_path)
