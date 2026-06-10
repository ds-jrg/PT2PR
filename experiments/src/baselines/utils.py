"""
Text extraction and image path utilities for baseline experiments.
"""
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional


def _extract_claims_text(claims_data: list, text_key: str = "text") -> List[str]:
    """Return one string per claim from the claims list."""
    if not claims_data or not isinstance(claims_data, list):
        return []
    result = []
    for claim in claims_data:
        if isinstance(claim, dict):
            claim_text = (claim.get(text_key) or claim.get("text", "")).strip()
        elif isinstance(claim, str):
            claim_text = claim.strip()
        else:
            claim_text = ""
        if claim_text:
            result.append(claim_text)
    return result


def get_patent_text(patent_data: Dict, masked: bool = False) -> str:
    """
    Concatenate all patent text fields (title, abstract, description, claims).
    When masked=True, reads *_masked fields, falling back to plain if absent.
    """
    suffix = "_masked" if masked else ""
    claims_text_key = f"text{suffix}" if suffix else "text"

    title = (
        patent_data.get(f"title{suffix}") or patent_data.get("title", "") or ""
    ).strip()
    abstract = (
        patent_data.get(f"abstract{suffix}") or patent_data.get("abstract", "") or ""
    ).strip()
    description = (
        patent_data.get(f"description{suffix}")
        or patent_data.get("description", "")
        or ""
    ).strip()

    claims_list = _extract_claims_text(
        patent_data.get("claims", []), text_key=claims_text_key
    )
    claims_str = " ".join(claims_list) if claims_list else ""

    return " ".join(p for p in [title, abstract, description, claims_str] if p)


def get_product_text(product_data: Dict, masked: bool = False) -> str:
    """
    Concatenate all product text fields (title, description, features).
    When masked=True, reads *_masked fields, falling back to plain if absent.
    """
    suffix = "_masked" if masked else ""
    parts: List[str] = []

    title = product_data.get(f"title{suffix}") or product_data.get("title", "")
    if title:
        parts.append(str(title).strip())

    description = product_data.get(f"description{suffix}") or product_data.get(
        "description", ""
    )
    if isinstance(description, list):
        parts.extend(str(d).strip() for d in description if str(d).strip())
    elif description:
        parts.append(str(description).strip())

    features = product_data.get(f"features{suffix}") or product_data.get("features", "")
    if isinstance(features, list):
        parts.extend(str(f).strip() for f in features if str(f).strip())
    elif features:
        parts.append(str(features).strip())

    return " ".join(p for p in parts if p)


def get_patent_image_path(pub_number: str, visual_assets_dir: Path) -> Optional[Path]:
    """Return the first available hi-res patent drawing (sorted by name), or None."""
    hi_res = visual_assets_dir / "patents" / pub_number / "hi_res"
    candidates = sorted(hi_res.iterdir()) if hi_res.exists() else []
    return candidates[0] if candidates else None


def get_product_image_path(id: str, visual_assets_dir: Path) -> Optional[Path]:
    """
    Return the path to the primary product image, or None if not found.
    Prefers a file whose lowercased name starts with 'main' (hi_res first,
    large as fallback). Falls back to the first available file if no MAIN
    image is found, matching the same behaviour as get_patent_image_path.
    """
    base = visual_assets_dir / "products" / id / "images"
    for sub in ("hi_res", "large"):
        folder = base / sub
        if not folder.exists():
            continue
        candidates = sorted(folder.iterdir())
        if not candidates:
            continue
        main = next((f for f in candidates if f.name.lower().startswith("main")), None)
        return main if main else candidates[0]
    return None


def load_patent_image(pub_number: str, visual_assets_dir: Path, size: int = 512):
    """
    Load the primary hi-res patent drawing as a PIL Image resized to size x size.
    Returns None if no image is found.
    """

    path = get_patent_image_path(pub_number, visual_assets_dir)
    if path is None:
        return None
    return Image.open(path).convert("RGB").resize((size, size))


def load_product_image(id: str, visual_assets_dir: Path, size: int = 512):
    """
    Load the primary product image as a PIL Image resized to size x size.
    Returns None if no image is found.
    """

    path = get_product_image_path(id, visual_assets_dir)
    if path is None:
        return None
    return Image.open(path).convert("RGB").resize((size, size))
