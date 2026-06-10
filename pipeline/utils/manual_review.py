"""
Checkpointing for manual review steps.
Allows reproducing the PT2PR-Amazon and PT2PR-ESCI datasets
with already completed manual review steps.

For new datasets (no manually checked file yet), the pipeline pauses at each
review step, writes the intermediate output, and prints instructions.
Re-run after creating the manually reviewed file to continue.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List


logger = logging.getLogger(__name__)

REVIEW_STOP = "REVIEW_STOP"


def review_exists(review_path: str) -> bool:
    """Return True if a manually reviewed file exists at the given path."""
    return Path(review_path).exists()


def apply_reviewed_changes(
    records: List[Dict],
    reviewed_file_path: str,
    step_label: str,
) -> List[Dict]:
    """
    Replace the automatic output with manually reviewed records.
    """
    if not review_exists(reviewed_file_path):
        return REVIEW_STOP

    reviewed_records: List[Dict] = []
    skipped = 0
    with open(reviewed_file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                reviewed_records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(
                    f"[{step_label}] Skipping malformed JSON at "
                    f"{reviewed_file_path}:{line_num} - {e}"
                )
                skipped += 1

    logger.info(
        f"[{step_label}] Reviewed changes applied: replaced {len(records)} automatic "
        f"record(s) with {len(reviewed_records)} manually checked record(s) "
        f"from '{reviewed_file_path}'."
        + (f" ({skipped} malformed lines skipped)" if skipped else "")
    )

    return reviewed_records
