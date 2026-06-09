# PT2PR: A Benchmark for Multimodal Patent-to-Product Retrieval

A benchmark dataset for semantic and multimodal patent-to-product retrieval, constructed from the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) and [ESCI](https://github.com/amazon-science/esci-data) product catalogs.

This repository provides the complete pipeline to reproduce both PT2PR-Amazon and PT2PR-ESCI from scratch and run text and multimodal retrieval baselines. It can also be applied to any product catalog where products reference patent numbers.

## Prerequisites

### Environment Setup (Linux)

Python **3.12** is recommended.

*Create and activate a virtual environment:*

```bash
python3 -m venv pt2pr_venv
source pt2pr_venv/bin/activate
pip install -r requirements.txt
```

### Data Acquisition
 
The source catalogs must be obtained and accepted under their respective licenses before running the pipeline.

#### Amazon Reviews 2023

1. Visit [Amazon Reviews 2023 website](https://amazon-reviews-2023.github.io/)
2. Download the **metadata** (`meta` column) files for all categories into `data/external/amazon/`. Each file is named `meta_<Category>.jsonl.gz`.

The direct download base URL is:
```
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_<Category>.jsonl.gz
```
Replace `<Category>` with the real category names from the website, e.g. `meta_All_Beauty.jsonl.gz`.

#### ESCI

1. Clone the [ESCI repository](https://github.com/amazon-science/esci-data)
2. Locate `shopping_queries_dataset/shopping_queries_dataset_products.parquet`
3. Convert it to the PT2PR pipeline's input format, using: 

```bash
python scripts/convert_esci.py \
    --products path/to/shopping_queries_dataset_products.parquet \
    --output   data/external/esci/products.jsonl.gz
```

This keeps only English (`product_locale == "us"`) products, drops duplicates, and maps ESCI fields to the pipeline input schema so the pipeline can process both the ESCI and Amazon Reviews 2023 catalogs identically.


### Reproducing PT2PR-Amazon and PT2PR-ESCI

Run the full pipeline for a single catalog with one command:
 
```bash
# Amazon Reviews 2023
python pipeline/run_pipeline.py --config pipeline/configs/amazon.yaml
 
# ESCI
python pipeline/run_pipeline.py --config pipeline/configs/esci.yaml
```
 
Individual steps can also be run standalone.


#### Manual review steps
 
Steps 02 and 03 include a manual review checkpoint. For PT2PR-Amazon and PT2PR-ESCI, the checkpoint files are already committed to the repo (`pipeline/checkpoints/`) and are applied automatically. The pipeline replaces the automatic extraction output with the manually verified file, ensuring reproducibility.
 
For a **new dataset**, the pipeline will pause at steps 02 and 03, write the automatic extraction output to the interim path, and print instructions. Review and correct the output, then save it as the checkpoint:
 
```
pipeline/checkpoints/<new_dataset>/step_02_manual_changes.jsonl
pipeline/checkpoints/<new_dataset>/step_03_manual_changes.jsonl
```
 
Re-run the pipeline to continue from where it stopped, skipping the already-completed steps, e.g.:
 
```bash
python pipeline/run_pipeline.py --config pipeline/configs/<new_dataset>.yaml --steps 3 4 5 6 7
```
 
---

## Dataset Construction Pipeline Overview
The pipeline constructs a patent-product pair dataset through five preprocessing steps and two merging steps. Manual validation at steps 02 and 03 ensures annotation quality.

### Preprocessing
| Step | Script | Function |
| --- | --- | --- |
| 01 | `extract_raw_data.py` | Scans `.jsonl.gz` catalog files. Extracts products that mention a patent number in their text. |
| 02 | `extract_interim_pairs.py` | Parses patent numbers from text spans to form `(product, patent)` pairs. Extracts country code, kind code, and surrounding text. **Manual checkpoint**: correct extraction errors, remove spurious pairs. |
| 03 | `extract_kind_codes.py` | Fetches patent kind codes (A1, B2, S1, …) from Google Patents for each unique `(country, patent_number)` pair. **Manual checkpoint**: correct if possible or remove mismatched pairs. |
| 04 | `extract_patent_info.py` | Fetches full patent metadata from Google Patents. Results are cached locally. |
| 05 | `clean_patent_info.py` | Postprocesses and deduplicates extracted patent content. |

### Merging
| Step | Script | Function |
| --- | --- | --- |
| 01 | `build_full_dataset.py` | Joins cleaned patent-product pairs with product metadata from step 01. Deduplicates on `(product_id, patent_number)` and removes pairs where the patent has no title (these lack all other patent metadata too). Produces a global dataset and a US-patents-only subset. |
| 02 | `create_example_set.py` | Samples 20 unique patents for manual inspection. |


## Baseline Experiments
 
See [`experiments/README.md`](experiments/README.md) for instructions on running the text and multimodal retrieval baselines.