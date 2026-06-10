# PT2PR: A Benchmark for Multimodal Patent-to-Product Retrieval
A benchmark dataset for semantic and multimodal patent-to-product retrieval, constructed from the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) and [ESCI](https://github.com/amazon-science/esci-data) product catalogs.

This repository provides the complete pipeline to reproduce both PT2PR-Amazon and PT2PR-ESCI from scratch and run text and multimodal retrieval baselines. It can also be applied to any product catalog where products reference patent numbers.

## Prerequisites

### Environment Setup (Linux)
Python **3.12** is recommended.

*Clone the repository, then create and activate a virtual environment:*

```bash
git clone https://github.com/ds-jrg/PT2PR.git
cd PT2PR
python3 -m venv pt2pr_venv
source pt2pr_venv/bin/activate
pip install -r requirements.txt
```

### Data Acquisition
The source catalogs must be obtained and accepted under their respective licenses before running the pipeline.

#### Amazon Reviews 2023
To download the Amazon Reviews dataset, do: 

```bash
bash scripts/download_amazon.sh
```

#### ESCI
To download the ESCI dataset and convert it to the PT2PR pipeline format, do:

```bash
bash scripts/download_esci.sh
```


### Reproducing PT2PR-Amazon and PT2PR-ESCI
Run the full pipeline for a single catalog with one command:
 
```bash
# Amazon Reviews 2023
python -m pipeline.run_pipeline --config pipeline/configs/amazon.yaml
 
# ESCI
python -m pipeline.run_pipeline --config pipeline/configs/esci.yaml
```
 
Individual steps can also be run standalone.


#### Manual review steps
Steps 02 and 03 include a manual step. For PT2PR-Amazon and PT2PR-ESCI, the reviewed files are already committed to the repo (`pipeline/review_steps/`) and are applied automatically. The pipeline replaces the automatic extraction output with the manually verified file, ensuring reproducibility.
 
For a **new dataset**, the pipeline will pause at steps 02 and 03, write the automatic extraction output to the interim path, and print instructions. Review and correct the output, then save it as:
 
```
pipeline/review_steps/<new_dataset>/step_02_manual_changes.jsonl
pipeline/review_steps/<new_dataset>/step_03_manual_changes.jsonl
```
 
Re-run the pipeline to continue from where it stopped, skipping the already-completed steps, e.g.:
 
```bash
python -m pipeline.run_pipeline --config pipeline/configs/<new_dataset>.yaml --steps 3 4 5 6 7
```
 
---

## Dataset Construction Pipeline Overview
The pipeline constructs a patent-product pair dataset through five preprocessing steps and two merging steps. Manual validation at steps 02 and 03 ensures annotation quality.

### Preprocessing
| Step | Script | Function |
| --- | --- | --- |
| 01 | `extract_raw_data.py` | Scans `.jsonl.gz` catalog files. Extracts products that mention a patent number in their text. |
| 02 | `extract_interim_pairs.py` | Parses patent numbers from text spans to form `(product, patent)` pairs. Extracts country code, kind code, and surrounding text. **Manual review**: correct extraction errors, remove spurious pairs. |
| 03 | `extract_kind_codes.py` | Fetches patent kind codes (A1, B2, S1, …) from Google Patents for each unique `(country, patent_number)` pair. **Manual review**: correct if possible or remove mismatched pairs. |
| 04 | `extract_patent_info.py` | Fetches full patent metadata from Google Patents. Results are cached locally. |
| 05 | `clean_patent_info.py` | Postprocesses and deduplicates extracted patent content. |

### Merging
| Step | Script | Function |
| --- | --- | --- |
| 01 | `build_full_dataset.py` | Joins cleaned patent-product pairs with product metadata from step 01. Deduplicates on `(product_id, patent_number)` and removes pairs where the patent has no title (these lack all other patent metadata too). Produces a global dataset and a US-patents-only subset. |
| 02 | `create_example_set.py` | Samples 20 unique patents for manual inspection. |


## Baseline Experiments
See [`experiments/README.md`](experiments/README.md) for instructions on running the text and multimodal retrieval baselines.