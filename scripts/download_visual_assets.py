"""
Script to download visual assets from the PT2PR-Amazon dataset.

NOTE: Visual assets (product images from Amazon catalogs and patent drawings
from Google Patents) were downloaded on June 1, 2026.
Re-running this script may yield different results if
URLs have since been taken down or changed.
"""

import argparse
import gzip
import hashlib
import json
import mimetypes
import subprocess
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm


URI_SAFE = re.compile(r"[^A-Za-z0-9._-]")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
TIMEOUT = 20
RATE_LIMIT_DELAY = 0.5  # Delay between requests to avoid blocking


def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        print(
            "[ERROR] ffmpeg not found in PATH. Please install ffmpeg.", file=sys.stderr
        )
        sys.exit(1)


def sanitize_filename(s: str, max_len: int = 200) -> str:
    s = s.strip().replace(" ", "_")
    s = URI_SAFE.sub("_", s)
    return s[:max_len]


def sha256_of_file(path: Path) -> Optional[str]:
    """Calculate SHA256 hash of a file with error handling."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        print(f"[ERROR] Failed to hash {path}: {e}")
        return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def scan_existing_entities(output_dir: Path):
    """Scan output directory for already-downloaded entities."""

    existing_products = set()
    existing_patents = set()

    products_dir = output_dir / "products"
    patents_dir = output_dir / "patents"

    if products_dir.exists():
        for p in products_dir.iterdir():
            if p.is_dir():
                existing_products.add(p.name)

    if patents_dir.exists():
        for p in patents_dir.iterdir():
            if p.is_dir():
                existing_patents.add(p.name)

    return existing_products, existing_patents


def requests_session_with_retries(
    total_retries: int = 3, backoff_factor: float = 0.3
) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def guess_ext_from_content_type(ct: Optional[str], url: str) -> str:
    """Guess file extension from content type or URL."""
    if ct:
        ct_main = ct.split(";", 1)[0].strip()
        ext = mimetypes.guess_extension(ct_main)
        if ext:
            return ext
    path_ext = Path(url.split("?", 1)[0]).suffix
    if path_ext:
        return path_ext
    return ""


def validate_image(path: Path) -> bool:
    """Validate that file is a valid image (requires Pillow)."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()
        return True
    except ImportError:
        # Pillow not installed, skip validation
        return True
    except Exception as e:
        print(f"[WARN] Image validation failed for {path}: {e}")
        return False


def validate_video(path: Path) -> bool:
    """Basic video validation; check file exists and has reasonable size."""
    try:
        if not path.exists():
            return False
        size = path.stat().st_size
        # MP4 videos should be at least 10KB
        return size > 10000
    except Exception as e:
        print(f"[WARN] Video validation failed for {path}: {e}")
        return False


def extract_mp4_url_from_html(html: str) -> Optional[str]:
    """Extended MP4 extractor, searches meta tags, <source>, JSON-LD, inline JS."""
    # Meta og:video tags
    for tag in ("og:video:secure_url", "og:video", "og:video:url"):
        m = re.search(
            rf'<meta[^>]+property=["\']{tag}["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            flags=re.I,
        )
        if m:
            u = m.group(1).strip()
            if u.endswith(".mp4"):
                return u

    # <video> or <source> tags
    for m in re.finditer(
        r'<(?:video|source)[^>]+src=["\']([^"\']+)["\']', html, flags=re.I
    ):
        u = m.group(1).strip()
        if u.endswith(".mp4"):
            return u

    # JSON-LD VideoObject
    for m in re.finditer(
        r"<script[^>]+application/ld\+json[^>]*>(.*?)</script>", html, flags=re.S | re.I
    ):
        try:
            data = json.loads(m.group(1))

            def walk(obj):
                if isinstance(obj, dict):
                    if "Video" in str(obj.get("@type")):
                        for k in ("contentUrl", "url", "embedUrl"):
                            if isinstance(obj.get(k), str) and obj[k].endswith(".mp4"):
                                return obj[k]
                    for v in obj.values():
                        r = walk(v)
                        if r:
                            return r
                elif isinstance(obj, list):
                    for v in obj:
                        r = walk(v)
                        if r:
                            return r
                return None

            r = walk(data)
            if r:
                return r
        except Exception:
            continue

    # Inline JS direct links
    m = re.search(
        r'["\'](https?://m\.media-amazon\.com/[^"\']+\.mp4[^"\']*)["\']',
        html,
        flags=re.I,
    )
    if m:
        return m.group(1)

    # Fallback generic regex
    mp4s = re.findall(
        r'https?://[^\s"\'<>]+?\.mp4(?:\?[^"\s"\'<>]*)?', html, flags=re.I
    )
    if mp4s:
        return mp4s[0]

    # Fallback to m3u8 (rare)
    m3u8s = re.findall(
        r'https?://[^\s"\'<>]+?\.m3u8(?:\?[^"\s"\'<>]*)?', html, flags=re.I
    )
    if m3u8s:
        return m3u8s[0]

    return None


def fetch_highres_patent_images(
    session: requests.Session, patent_page_url: str, thumb_urls: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Scrapes Google Patents page and returns:
    - thumb_matches: URLs matching thumbnail filenames (for verification/hi-res versions)
    - highres_only: URLs that don't match any thumbnail (additional hi-res images)
    """
    try:
        time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
        resp = session.get(
            patent_page_url,
            timeout=TIMEOUT,
            headers={
                "User-Agent": USER_AGENT,
                "Referer": "https://patents.google.com/",
            },
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to fetch patent page {patent_page_url}: {e}")
        return [], []

    html = resp.text

    # Collect all PNG URLs from HTML
    matches = re.findall(
        r"https://patentimages\.storage\.googleapis\.com/[A-Za-z0-9/_\-]+\.png", html
    )
    matches = list(dict.fromkeys(matches))  # Remove duplicates, preserve order

    # Also check JSON-LD
    for m in re.finditer(
        r"<script[^>]+application/ld\+json[^>]*>(.*?)</script>", html, flags=re.S | re.I
    ):
        try:
            obj = json.loads(m.group(1))

            def walk(x):
                found = []
                if (
                    isinstance(x, str)
                    and x.startswith("https://patentimages.storage.googleapis.com/")
                    and x.endswith(".png")
                ):
                    found.append(x)
                elif isinstance(x, dict):
                    for v in x.values():
                        found.extend(walk(v))
                elif isinstance(x, list):
                    for v in x:
                        found.extend(walk(v))
                return found

            matches.extend(walk(obj))
        except Exception:
            continue

    # Remove duplicates again
    matches = list(dict.fromkeys(matches))

    # Extract filenames from thumb URLs (these are the images we already have)
    thumb_filenames = set()
    for url in thumb_urls:
        fname = Path(url.split("?", 1)[0]).name
        thumb_filenames.add(fname)

    # Categorize scraped URLs
    thumb_matches = []
    highres_only = []

    for url in matches:
        fname = Path(url.split("?", 1)[0]).name

        if fname in thumb_filenames:
            # Skip if the scraped URL is exactly the same as the thumbnail URL.
            # This prevents thumbnails from being treated as highres.
            if url in thumb_urls:
                continue

            # Otherwise keep it as highres_matched; only if URL differs.
            thumb_matches.append(url)
        else:
            highres_only.append(url)

    print(
        f"[INFO] Patent {patent_page_url}: Found {len(thumb_matches)} hi-res matches for {len(thumb_urls)} thumbs, {len(highres_only)} additional hi-res images"
    )

    return thumb_matches, highres_only


def download_hls_playlist(
    m3u8_url: str, output_path: Path
) -> Tuple[bool, Optional[str], Optional[Path]]:
    """
    Download an HLS .m3u8 playlist and assemble into a single MP4 using ffmpeg.
    Returns: (success, reason, final_path)
    """
    try:
        ensure_dir(output_path.parent)
        cmd = [
            "ffmpeg",
            "-y",  # overwrite if exists
            "-i",
            m3u8_url,
            "-c",
            "copy",  # copy streams without re-encoding
            str(output_path),
        ]
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if output_path.exists() and output_path.stat().st_size > 0:
            return True, None, output_path
        else:
            return False, "empty_file_after_ffmpeg", None
    except subprocess.CalledProcessError as e:
        return False, f"ffmpeg_failed: {e}", None
    except Exception as e:
        return False, f"exception: {e}", None


def download_url(
    session: requests.Session,
    url: str,
    dest_path: Path,
    timeout: int = 20,
    overwrite: bool = False,
    retries: int = 3,
) -> Tuple[bool, Optional[str], Optional[Path]]:
    """
    Download with retries and referer support.
    Returns: (success, error_reason, final_path)
    """
    try:
        if dest_path.exists() and not overwrite:
            # Check if file is valid
            if dest_path.stat().st_size > 0:
                return True, "exists", dest_path
            else:
                # Empty file, re-download
                try:
                    dest_path.unlink()
                except Exception:
                    pass

        ensure_dir(dest_path.parent)
        headers = {"User-Agent": USER_AGENT}
        if "patentimages.storage.googleapis.com" in url:
            headers["Referer"] = "https://patents.google.com/"

        time.sleep(RATE_LIMIT_DELAY)  # Rate limiting

        attempt = 0
        last_error = None

        while attempt < retries:
            try:
                r = session.get(url, stream=True, timeout=timeout, headers=headers)
                if r.status_code >= 400:
                    last_error = f"HTTP_{r.status_code}"
                    attempt += 1
                    time.sleep(1 + attempt)
                    continue

                # Determine final path with extension
                ext = guess_ext_from_content_type(r.headers.get("Content-Type"), url)
                final_path = dest_path
                if not dest_path.suffix and ext:
                    final_path = dest_path.with_suffix(ext)

                # Download file
                with open(final_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # Validate download
                if final_path.exists() and final_path.stat().st_size > 0:
                    return True, None, final_path
                else:
                    last_error = "empty_file"
                    attempt += 1
                    time.sleep(1 + attempt)

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                attempt += 1
                time.sleep(1 + attempt)

        return False, f"failed_after_{retries}_retries: {last_error}", None

    except Exception as e:
        return False, f"exception: {str(e)}", None


def extract_assets_from_record(
    r: dict, session: requests.Session, fetch_highres: bool
) -> List[dict]:
    """Extract all assets from a record with proper categorization."""
    assets = []
    product_id = r.get("product_id") or (r.get("product") or {}).get("product_id")
    prod = r.get("product") or {}

    # Product images
    for img in (
        prod.get("images", []) if isinstance(prod.get("images", []), list) else []
    ):
        for t in ("thumb", "large", "hi_res"):
            if t in img and img[t]:
                assets.append(
                    {
                        "url": img[t],
                        "type": "product_image",
                        "subtype": t,
                        "variant": img.get("variant"),
                        "product_id": product_id,
                        "source": "product.images",
                    }
                )

    # Product videos
    for vid in (
        prod.get("videos", []) if isinstance(prod.get("videos", []), list) else []
    ):
        if "url" in vid and vid["url"]:
            assets.append(
                {
                    "url": vid["url"],
                    "type": "product_video",
                    "title": vid.get("title"),
                    "product_id": product_id,
                    "source": "product.videos",
                }
            )

    # Patent images
    patent = r.get("patent") or {}
    pd = patent.get("patent_data") or {}
    patent_num = pd.get("bibliographic", {}).get("publicationNumber")
    thumb_urls = pd.get("images", []) if isinstance(pd.get("images", []), list) else []

    # Always add thumbnail URLs
    for idx, img_url in enumerate(thumb_urls):
        assets.append(
            {
                "url": img_url,
                "type": "patent_image",
                "patent_number": patent_num,
                "product_id": product_id,
                "source": "patent.images",
                "resolution": "thumbnail",
                "thumb_index": idx,
            }
        )

    # Defer high-res fetching to the worker phase to avoid blocking the scan loop
    if fetch_highres and thumb_urls:
        patent_page = pd.get("diagnostics", {}).get("google_patents", {}).get("url")
        if patent_page:
            assets.append(
                {
                    "url": patent_page,
                    "type": "patent_highres_fetch",
                    "patent_number": patent_num,
                    "product_id": product_id,
                    "thumb_urls": list(thumb_urls),
                }
            )

    return assets


def build_target_path(output_dir: Path, asset: dict, url_hash: str = "") -> Path:
    """
    Build target path for asset with collision prevention.
    url_hash can be used to prevent filename collisions.
    """
    product_id = asset.get("product_id") or "UNKNOWN_PRODUCT"

    if asset["type"] == "product_image":
        subtype = asset.get("subtype") or "image"
        variant = asset.get("variant") or ""
        url_basename = Path(asset["url"].split("?", 1)[0]).name

        # Add hash suffix if provided to prevent collisions
        if url_hash:
            name_parts = url_basename.rsplit(".", 1)
            if len(name_parts) == 2:
                name = sanitize_filename(
                    f"{variant}_{name_parts[0]}_{url_hash[:8]}.{name_parts[1]}"
                )
            else:
                name = sanitize_filename(f"{variant}_{url_basename}_{url_hash[:8]}")
        else:
            name = (
                sanitize_filename(f"{variant}_{url_basename}")
                if variant
                else sanitize_filename(url_basename)
            )

        return output_dir / "products" / product_id / "images" / subtype / name

    elif asset["type"] == "product_video":
        title = asset.get("title") or ""
        name_part = sanitize_filename(title or Path(asset["url"]).stem)
        if url_hash:
            name_part = f"{name_part}_{url_hash[:8]}"
        return output_dir / "products" / product_id / "videos" / name_part

    elif asset["type"] == "patent_image":
        patent_number = asset.get("patent_number") or "UNKNOWN_PATENT"
        res = asset.get("resolution", "unknown")
        url_basename = Path(asset["url"].split("?", 1)[0]).name

        # Add hash suffix if provided
        if url_hash:
            name_parts = url_basename.rsplit(".", 1)
            if len(name_parts) == 2:
                name = sanitize_filename(
                    f"{name_parts[0]}_{url_hash[:8]}.{name_parts[1]}"
                )
            else:
                name = sanitize_filename(f"{url_basename}_{url_hash[:8]}")
        else:
            name = sanitize_filename(url_basename)

        # Map resolution to folder name
        if res == "thumbnail":
            folder = "thumb"
        elif res in ("highres_matched", "highres_additional"):
            folder = "hi_res"
        else:
            folder = f"images_{res}"

        return output_dir / "patents" / patent_number / folder / name

    else:
        name = sanitize_filename(Path(asset["url"]).name or "file")
        if url_hash:
            name = f"{name}_{url_hash[:8]}"
        return output_dir / "other" / name


def load_existing_manifest(manifest_path: Path) -> Dict[str, dict]:
    """Load existing manifest for resume capability (keyed by URL)."""
    existing = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("url"):
                            existing[entry["url"]] = entry
        except Exception as e:
            print(f"[WARN] Failed to load existing manifest: {e}")
    return existing


def dedup_key_for_asset(asset: dict):
    """Return dedup key tuple for an asset (works for both manifest entries and extracted assets)."""
    t = asset.get("type")
    url = asset.get("url")
    if t == "product_image":
        return ("product_image", asset.get("product_id"), asset.get("subtype"), url)
    if t == "product_video":
        return ("product_video", asset.get("product_id"), url)
    if t == "patent_image":
        # resolution may be None for some entries; that's fine
        return (
            "patent_image",
            asset.get("patent_number"),
            asset.get("resolution"),
            url,
        )
    return ("other", url)


def download_asset(
    asset: dict,
    output_dir: Path,
    overwrite: bool,
    timeout: int,
    extract_videos: bool,
    fetch_highres: bool,
) -> Tuple[str, object]:
    """Worker-safe downloader for one asset (used by multiprocessing)."""
    # Ensure we always return identifying fields so parent can reconstruct dedup key
    asset = dict(asset)  # shallow copy to avoid modifying caller's object
    try:
        session = requests_session_with_retries()
        url = asset.get("url")
        asset_type = asset.get("type", "unknown")
        url_hash = hashlib.sha256((url or "").encode()).hexdigest()
        target = build_target_path(output_dir, asset, url_hash)

        # Fill identifying fields if missing (helps parent dedup)
        if asset_type == "product_image":
            asset.setdefault("product_id", asset.get("product_id"))
            asset.setdefault("subtype", asset.get("subtype"))
        if asset_type == "product_video":
            asset.setdefault("product_id", asset.get("product_id"))
        if asset_type == "patent_image":
            asset.setdefault("patent_number", asset.get("patent_number"))
            asset.setdefault("resolution", asset.get("resolution"))

        # Handle deferred high-res patent page fetches
        if asset_type == "patent_highres_fetch":
            thumb_urls = asset.get("thumb_urls", [])
            patent_number = asset.get("patent_number")
            thumb_matches, highres_only = fetch_highres_patent_images(
                session, url, thumb_urls
            )

            results = []
            for idx, img_url in enumerate(thumb_matches):
                img_asset = {
                    "url": img_url,
                    "type": "patent_image",
                    "patent_number": patent_number,
                    "product_id": asset.get("product_id"),
                    "source": "patent.google_patents_scrape",
                    "resolution": "highres_matched",
                    "match_index": idx,
                }
                img_hash = hashlib.sha256(img_url.encode()).hexdigest()
                img_target = build_target_path(output_dir, img_asset, img_hash)
                success, reason, final_path = download_url(
                    session, img_url, img_target, timeout=timeout, overwrite=overwrite
                )
                if success and final_path and final_path.exists():
                    if not validate_image(final_path):
                        try:
                            final_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        results.append(
                            {
                                **img_asset,
                                "downloaded": False,
                                "reason": "image_validation_failed",
                            }
                        )
                    else:
                        h = sha256_of_file(final_path)
                        results.append(
                            {
                                **img_asset,
                                "downloaded": True,
                                "local_path": str(final_path),
                                "sha256": h,
                                "file_size": final_path.stat().st_size,
                            }
                        )
                else:
                    results.append(
                        {
                            **img_asset,
                            "downloaded": False,
                            "reason": reason or "download_failed",
                        }
                    )

            for idx, img_url in enumerate(highres_only):
                img_asset = {
                    "url": img_url,
                    "type": "patent_image",
                    "patent_number": patent_number,
                    "product_id": asset.get("product_id"),
                    "source": "patent.google_patents_scrape",
                    "resolution": "highres_additional",
                    "additional_index": idx,
                }
                img_hash = hashlib.sha256(img_url.encode()).hexdigest()
                img_target = build_target_path(output_dir, img_asset, img_hash)
                success, reason, final_path = download_url(
                    session, img_url, img_target, timeout=timeout, overwrite=overwrite
                )
                if success and final_path and final_path.exists():
                    if not validate_image(final_path):
                        try:
                            final_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        results.append(
                            {
                                **img_asset,
                                "downloaded": False,
                                "reason": "image_validation_failed",
                            }
                        )
                    else:
                        h = sha256_of_file(final_path)
                        results.append(
                            {
                                **img_asset,
                                "downloaded": True,
                                "local_path": str(final_path),
                                "sha256": h,
                                "file_size": final_path.stat().st_size,
                            }
                        )
                else:
                    results.append(
                        {
                            **img_asset,
                            "downloaded": False,
                            "reason": reason or "download_failed",
                        }
                    )

            return url, results

        # Handle videos
        if extract_videos and asset_type == "product_video":
            try:
                html_resp = session.get(url, timeout=timeout)
            except Exception as e:
                return url, {
                    **asset,
                    "downloaded": False,
                    "reason": f"exception_fetching_vdp: {e}",
                }

            if html_resp.status_code != 200:
                return url, {
                    **asset,
                    "downloaded": False,
                    "reason": f"HTTP_{html_resp.status_code}",
                }

            mp4_url = extract_mp4_url_from_html(html_resp.text)
            if not mp4_url:
                return url, {**asset, "downloaded": False, "reason": "no_mp4_found"}

            video_path = target.with_suffix(".mp4")
            if mp4_url.endswith(".m3u8"):
                success, reason, final_path = download_hls_playlist(mp4_url, video_path)
            else:
                success, reason, final_path = download_url(
                    session, mp4_url, video_path, timeout=timeout, overwrite=overwrite
                )

            if (
                success
                and final_path
                and final_path.exists()
                and validate_video(final_path)
            ):
                h = sha256_of_file(final_path)
                return url, {
                    **asset,
                    "downloaded": True,
                    "local_path": str(final_path),
                    "sha256": h,
                    "file_size": final_path.stat().st_size,
                    "source_mp4": mp4_url,
                }
            else:
                return url, {
                    **asset,
                    "downloaded": False,
                    "reason": reason or "video_download_failed",
                }

        # Handle images / other files
        success, reason, final_path = download_url(
            session, url, target, timeout=timeout, overwrite=overwrite
        )
        if success and final_path and final_path.exists():
            if asset_type in ("product_image", "patent_image") and not validate_image(
                final_path
            ):
                try:
                    final_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return url, {
                    **asset,
                    "downloaded": False,
                    "reason": "image_validation_failed",
                }
            h = sha256_of_file(final_path)
            return url, {
                **asset,
                "downloaded": True,
                "local_path": str(final_path),
                "sha256": h,
                "file_size": final_path.stat().st_size,
            }

        return url, {
            **asset,
            "downloaded": False,
            "reason": reason or "download_failed",
        }

    except Exception as e:
        # Ensure we return enough identifying info even on catastrophic error
        return asset.get("url", "unknown"), {
            **asset,
            "downloaded": False,
            "reason": f"exception: {e}",
        }


def process_file(
    input_path: Path,
    output_dir: Path,
    overwrite: bool,
    timeout: int,
    extract_videos: bool,
    fetch_highres: bool,
    resume: bool = True,
    workers: int = 32,
) -> None:
    """Main processing function with threaded downloading and resume support."""

    opener = gzip.open if input_path.suffix in (".gz", ".gzip") else open
    session = requests_session_with_retries()
    ensure_dir(output_dir)

    existing_products, existing_patents = scan_existing_entities(output_dir)

    print(
        f"[INFO] Found "
        f"{len(existing_products)} existing products and "
        f"{len(existing_patents)} existing patents on disk"
    )

    manifest_path = output_dir / "manifest.jsonl"
    report_path = output_dir / "report.json"

    # Load existing manifest for resume
    existing_manifest = (
        load_existing_manifest(manifest_path)
        if resume and manifest_path.exists()
        else {}
    )

    print(f"[INFO] Loaded {len(existing_manifest)} existing entries from manifest")

    # Open manifest file for appending
    manifest_file = open(manifest_path, "a", encoding="utf-8")

    stats = {
        "total_urls": 0,
        "success_count": 0,
        "skipped_existing": 0,
        "skipped_duplicate": 0,
        "failures": [],
        "by_type": defaultdict(
            lambda: {
                "total": 0,
                "success": 0,
                "failed": 0,
            }
        ),
    }

    # Build seen_ids from successful manifest entries only
    seen_ids = set()

    for _, entry in existing_manifest.items():
        if entry.get("downloaded"):
            seen_ids.add(dedup_key_for_asset(entry))

    all_pending_assets = []

    # Dataset-level deduplication
    seen_products = set()
    seen_patents = set()

    print("[INFO] Scanning records...")

    # Read records lazily instead of loading entire file into memory
    with opener(input_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()

            if not line:
                continue

            try:
                rec = json.loads(line)

            except Exception as e:
                print(f"[WARN] Failed to parse JSON line: {e}")
                continue

            product_id = rec.get("product_id") or (rec.get("product") or {}).get(
                "product_id"
            )

            patent = rec.get("patent") or {}
            pd = patent.get("patent_data") or {}

            patent_number = pd.get("bibliographic", {}).get("publicationNumber")

            skip_product = False
            skip_patent = False

            duplicate_record = False

            if product_id:
                if product_id in seen_products:
                    skip_product = True
                    duplicate_record = True

                else:
                    seen_products.add(product_id)

                    if not overwrite and product_id in existing_products:
                        skip_product = True
                        stats["skipped_existing"] += 1

            if patent_number:
                if patent_number in seen_patents:
                    skip_patent = True
                    duplicate_record = True

                else:
                    seen_patents.add(patent_number)

                    if not overwrite and patent_number in existing_patents:
                        skip_patent = True
                        stats["skipped_existing"] += 1

            if duplicate_record:
                stats["skipped_duplicate"] += 1

            try:
                assets = extract_assets_from_record(
                    rec,
                    session,
                    fetch_highres,
                )

                # Remove skipped entity asset types
                filtered_assets = []

                for asset in assets:
                    if skip_product and asset.get("type") in (
                        "product_image",
                        "product_video",
                    ):
                        continue

                    if skip_patent and asset.get("type") in (
                        "patent_image",
                        "patent_highres_fetch",
                    ):
                        continue

                    filtered_assets.append(asset)

                assets = filtered_assets

            except Exception as e:
                print(f"[WARN] Failed extracting assets: {e}")
                continue

            # Filter only valid http(s) assets
            for asset in assets:
                url = asset.get("url") or ""

                if not (
                    isinstance(url, str)
                    and url.lower().startswith(("http://", "https://"))
                ):
                    invalid_entry = dict(asset)
                    invalid_entry["downloaded"] = False
                    invalid_entry["reason"] = "invalid_url"

                    try:
                        manifest_file.write(json.dumps(invalid_entry) + "\n")
                        manifest_file.flush()

                        # Prevent repeated invalid-url processing
                        seen_ids.add(dedup_key_for_asset(invalid_entry))

                    except Exception as e:
                        print(
                            f"[WARN] Failed to write invalid-url "
                            f"manifest entry for {url}: {e}"
                        )

                    continue

                # Deduplicate against completed + current-run assets
                k = dedup_key_for_asset(asset)

                if k in seen_ids:
                    stats["skipped_existing"] += 1
                    continue

                seen_ids.add(k)
                all_pending_assets.append(asset)

    print(
        f"[INFO] Processing {len(all_pending_assets)} assets "
        f"with {workers} workers..."
    )

    def worker(asset):
        """Safe wrapper around download_asset()."""

        try:
            return download_asset(
                asset,
                output_dir,
                overwrite,
                timeout,
                extract_videos,
                fetch_highres,
            )

        except Exception as e:
            url = asset.get("url")

            return url, {
                **asset,
                "downloaded": False,
                "reason": (f"worker_exception: " f"{type(e).__name__}: {e}"),
                "type": asset.get("type", "unknown"),
            }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(worker, asset): asset for asset in all_pending_assets
        }

        try:
            with tqdm(
                total=len(futures),
                desc="Downloading",
                unit="asset",
            ) as pbar:
                for fut in as_completed(futures):
                    asset = futures[fut]

                    try:
                        url, result = fut.result()

                    except Exception as e:
                        # Should rarely happen because worker catches errors
                        url = asset.get("url")

                        result = {
                            **asset,
                            "downloaded": False,
                            "reason": (
                                f"future_exception: " f"{type(e).__name__}: {e}"
                            ),
                            "type": asset.get("type", "unknown"),
                        }

                    # patent_highres_fetch workers return a list of results;
                    # normalise to a list so the loop below handles both cases.
                    results = result if isinstance(result, list) else [result]

                    for result in results:
                        asset_type = result.get("type", "unknown")

                        stats["total_urls"] += 1
                        stats["by_type"][asset_type]["total"] += 1

                        # Write manifest entry
                        try:
                            manifest_file.write(
                                json.dumps(result, ensure_ascii=False) + "\n"
                            )
                            manifest_file.flush()

                        except Exception as e:
                            print(
                                f"[ERROR] Failed to write manifest "
                                f"entry for {url}: {e}"
                            )

                        if result.get("downloaded"):
                            stats["success_count"] += 1
                            stats["by_type"][asset_type]["success"] += 1

                        else:
                            stats["failures"].append(
                                {
                                    "url": url,
                                    "reason": result.get("reason"),
                                    "type": asset_type,
                                }
                            )

                            stats["by_type"][asset_type]["failed"] += 1

                    pbar.update(1)

                    pbar.set_postfix(
                        {
                            "success": stats["success_count"],
                            "failed": len(stats["failures"]),
                            "skipped": stats["skipped_existing"],
                        }
                    )

        finally:
            manifest_file.close()

    # Summary report
    report = {
        "input_file": str(input_path),
        "output_directory": str(output_dir),
        "total_urls_found": stats["total_urls"],
        "successful_downloads": stats["success_count"],
        "skipped_existing": stats["skipped_existing"],
        "total_failures": len(stats["failures"]),
        "by_type": dict(stats["by_type"]),
        "failure_samples": stats["failures"][:100],
        "settings": {
            "extract_videos": extract_videos,
            "fetch_highres_patents": fetch_highres,
            "overwrite": overwrite,
            "timeout": timeout,
            "workers": workers,
        },
    }

    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2)

    print(f"Total URLs found: {stats['total_urls']}")
    print(f"Successful downloads: {stats['success_count']}")
    print(f"Skipped (existing): {stats['skipped_existing']}")
    print(f"Failures: {len(stats['failures'])}")

    print(f"\nDetailed report: {report_path}")
    print(f"Manifest: {manifest_path}")


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point with argument parsing."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=""
    )
    p.add_argument("--input", "-i", required=True, help="Input JSONL or JSONL.GZ file")
    p.add_argument("--output", "-o", required=True, help="Output directory")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p.add_argument(
        "--timeout", type=int, default=20, help="HTTP timeout in seconds (default: 20)"
    )
    p.add_argument(
        "--extract-videos",
        action="store_true",
        help="Extract and download actual MP4s from VDP pages",
    )
    p.add_argument(
        "--fetch-highres-patents",
        action="store_true",
        help="Fetch and download high-resolution patent images from Google Patents",
    )
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from existing manifest",
    )

    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start fresh, ignore existing manifest",
    )

    p.set_defaults(resume=True)
    p.add_argument(
        "--workers", type=int, default=8, help="Number of parallel workers (default: 8)"
    )

    args = p.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Validation
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 1

    if not input_path.is_file():
        print(f"[ERROR] Input path is not a file: {input_path}", file=sys.stderr)
        return 1

    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Extract videos: {args.extract_videos}")
    print(f"Fetch high-res patents: {args.fetch_highres_patents}")
    print(f"Overwrite existing: {args.overwrite}")
    print(f"Resume mode: {args.resume}")
    print(f"Timeout: {args.timeout}s")

    try:
        process_file(
            input_path=input_path,
            output_dir=output_dir,
            overwrite=args.overwrite,
            timeout=args.timeout,
            extract_videos=args.extract_videos,
            fetch_highres=args.fetch_highres_patents,
            resume=args.resume,
            workers=args.workers,
        )
        return 0
    except KeyboardInterrupt:
        print("\n[INFO] Download interrupted by user. Progress saved in manifest.")
        print("[INFO] Run again with --resume to continue.")
        return 130
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    check_ffmpeg()
    raise SystemExit(main())
