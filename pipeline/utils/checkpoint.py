"""
Checkpointing for manual review steps.
Allows reproducing the PT2PR-Amazon and PT2PR-ESCI datasets
with already completed manual review steps.

For new datasets (no manually checked file yet), the pipeline pauses at each
checkpoint step, writes the intermediate output, and prints instructions.
Re-run after creating the checkpoint file to continue.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List


logger = logging.getLogger(__name__)

CHECKPOINT_STOP = "CHECKPOINT_STOP"


def checkpoint_exists(checkpoint_path: str) -> bool:
    """Return True if a checkpoint file exists at the given path."""
    return Path(checkpoint_path).exists()


def apply_checkpoint(
    records: List[Dict],
    checkpoint_path: str,
    step_label: str,
) -> List[Dict]:
    """
    Apply a checkpoint by replacing the automatic output with the manually
    checked records stored in the checkpoint file.
    """
    if not checkpoint_exists(checkpoint_path):
        return CHECKPOINT_STOP

    checkpoint_records: List[Dict] = []
    skipped = 0
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                checkpoint_records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(
                    f"[{step_label}] Skipping malformed JSON at "
                    f"{checkpoint_path}:{line_num} - {e}"
                )
                skipped += 1

    logger.info(
        f"[{step_label}] Checkpoint applied: replaced {len(records)} automatic "
        f"record(s) with {len(checkpoint_records)} manually checked record(s) "
        f"from '{checkpoint_path}'."
        + (f" ({skipped} malformed lines skipped)" if skipped else "")
    )

    return checkpoint_records
