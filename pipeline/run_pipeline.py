"""
PT2PR dataset construction pipeline

Runs all preprocessing and merging steps in sequence for a given dataset config.

Steps:
    1) extract_raw_data: scan source catalog, filter patent mentions
    2) extract_interim_pairs: parse patent numbers from product text [manual review step]
    3) extract_kind_codes: fetch kind codes from Google Patents [manual review step]
    4) extract_patent_info: fetch full patent metadata
    5) clean_patent_info: remove partial duplicate claims
    6) build_full_dataset: merge patents + products, compute stats
    7) create_example_set: sample 20 examples for manual verification

Manual review steps:
    Steps 2 and 3 apply a manual patch file after automatic processing.
    - For the provided Amazon and ESCI datasets, patch files are committed to
      the repo under pipeline/review_steps/ and are applied automatically.
    - For new datasets, the pipeline pauses at each manual review step, writes the
      intermediate output, and prints instructions for creating the patch file.
      Re-run after creating the file to continue.
"""

import argparse
import logging
from typing import List, Optional
import yaml
from pipeline.preprocessing.extract_raw_data import extract_patented_products
from pipeline.preprocessing.extract_interim_pairs import extract_interim_pairs
from pipeline.preprocessing.extract_kind_codes import extract_kind_codes
from pipeline.preprocessing.extract_patent_info import extract_patent_info
from pipeline.preprocessing.clean_patent_info import clean_patent_info
from pipeline.merging.build_full_dataset import build_full_dataset
from pipeline.merging.create_example_set import create_example_set

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _require(cfg: dict, *keys: str):
    """Retrieve a nested key from config, raising a clear error if absent."""
    current = cfg
    path = []
    for key in keys:
        path.append(key)
        if not isinstance(current, dict) or key not in current:
            raise KeyError(
                f"Required config key missing: {'.'.join(path)}\n"
                f"Check your config file."
            )
        current = current[key]
    return current


def _run_step_1(cfg: dict) -> None:
    logger.info("PREPROCESSING STEPS")
    logger.info("STEP 1: extracting patented products from source catalog")
    extract_patented_products(
        input_dir=_require(cfg, "source", "input_dir"),
        output_path=_require(cfg, "paths", "raw_products"),
        product_id_field=cfg.get("source", {}).get("product_id_field"),
    )


def _run_step_2(cfg: dict) -> None:
    logger.info("STEP 2: extracting interim patent-product pairs [manual review step]")
    extract_interim_pairs(
        input_path=_require(cfg, "paths", "raw_products"),
        output_path=_require(cfg, "paths", "interim_pairs"),
        reviewed_file_path=cfg.get("review_steps", {}).get("step_02"),
    )


def _run_step_3(cfg: dict) -> None:
    logger.info("STEP 3: fetching kind codes from Google Patents [manual review step]")
    scraping = cfg.get("scraping", {})
    extract_kind_codes(
        input_path=_require(cfg, "paths", "interim_pairs"),
        output_path=_require(cfg, "paths", "pairs_with_kinds"),
        reviewed_file_path=cfg.get("review_steps", {}).get("step_03"),
        batch_size=scraping.get("kind_codes_batch_size", 10),
        max_workers=scraping.get("kind_codes_max_workers", 5),
    )


def _run_step_4(cfg: dict) -> None:
    logger.info("STEP 4: fetching full patent metadata from Google Patents")
    scraping = cfg.get("scraping", {})
    extract_patent_info(
        input_path=_require(cfg, "paths", "pairs_with_kinds"),
        output_path=_require(cfg, "paths", "patent_info"),
        cache_path=_require(cfg, "paths", "patent_cache"),
        batch_size=scraping.get("patent_info_batch_size", 300),
        max_workers=scraping.get("patent_info_max_workers", 5),
        request_delay=scraping.get("request_delay", 2.0),
    )


def _run_step_5(cfg: dict) -> None:
    logger.info("STEP 5: cleaning partial duplicate patent claims")
    clean_patent_info(
        input_path=_require(cfg, "paths", "patent_info"),
        output_path=_require(cfg, "paths", "patent_info_cleaned"),
    )


def _run_step_6(cfg: dict) -> None:
    logger.info("MERGING STEPS")
    logger.info("STEP 6: building full merged dataset")
    build_full_dataset(
        patents_input=_require(cfg, "paths", "patent_info_cleaned"),
        products_input=_require(cfg, "paths", "raw_products"),
        output_path=_require(cfg, "paths", "merged_dataset"),
        stats_output_path=_require(cfg, "paths", "merged_stats"),
    )


def _run_step_7(cfg: dict) -> None:
    logger.info("STEP 7: creating example set")
    examples_cfg = cfg.get("examples", {})
    create_example_set(
        input_path=_require(cfg, "paths", "merged_dataset"),
        output_path=_require(cfg, "paths", "examples"),
        n=examples_cfg.get("n", 20),
        seed=examples_cfg.get("seed", 42),
    )


_STEP_RUNNERS = {
    1: _run_step_1,
    2: _run_step_2,
    3: _run_step_3,
    4: _run_step_4,
    5: _run_step_5,
    6: _run_step_6,
    7: _run_step_7,
}


def run_pipeline(
    config_path: str,
    steps: Optional[List[int]] = None,
    no_examples: bool = False,
) -> None:
    """Run the full pipeline (or a subset of steps) for the given config."""
    cfg = _load_config(config_path)
    dataset = cfg.get("dataset", "unknown")
    logger.info(f"Pipeline starting for dataset: '{dataset}'")
    logger.info(f"Config: {config_path}")

    all_steps = list(range(1, 8))
    if no_examples:
        all_steps = [s for s in all_steps if s != 7]
    if steps:
        invalid = [s for s in steps if s not in _STEP_RUNNERS]
        if invalid:
            raise ValueError(f"Unknown step number(s): {invalid}. Valid: 1-7.")
        all_steps = [s for s in all_steps if s in steps]

    logger.info(f"Steps to run: {all_steps}")

    for step_num in all_steps:
        runner = _STEP_RUNNERS[step_num]
        try:
            runner(cfg)
        except SystemExit:
            # A manual review stop
            # Reraise to propagate the exit code.
            raise
        except Exception as e:
            logger.error(f"Step {step_num} failed: {e}", exc_info=True)
            raise

    logger.info(f"Pipeline complete for dataset '{dataset}'.")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the patent-product dataset construction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a dataset .yaml config file (e.g. pipeline/configs/amazon.yaml).",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help="Run only these step numbers (e.g. --steps 1 2 3). Default: all steps.",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Skip step 7 (example set creation).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run_pipeline(
        config_path=args.config,
        steps=args.steps,
        no_examples=args.no_examples,
    )
