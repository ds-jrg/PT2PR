# PT2PR Baseline Experiments
Text and multimodal retrieval baselines for patent-to-product matching, evaluated on PT2PR-Amazon and PT2PR-ESCI.

## Prerequisites
Activate the virtual environment set up in the main PT2PR repo README before running any of the commands below:

```bash
cd PT2PR
source pt2pr_venv/bin/activate
```

## Data Preparation
Both settings (text and multimodal) require the datasets to be preprocessed and masked. These steps must be run before any experiments.

The multimodal setting for the PT2PR-Amazon experiments additionally requires visual assets (product images and patent drawings) to be present on disk. To download, do:

```bash
python -m scripts.download_visual_assets \
    --input data/processed/amazon/full_patent_product_dataset_US_only.jsonl \
    --output <your_visual_assets_dir>/amazon \
    --fetch-highres-patents
```

Then preprocess and mask for both **text** and **multimodal** settings:

```bash
python -m experiments.src.dataset.preprocess_data --dataset both --setting both \
    --visual-assets-dir <your_visual_assets_dir>
python -m experiments.src.dataset.mask_data --dataset both --setting both
```


## Running Experiments

Results are written to `experiments/results/<dataset>/<setting>_setting/`.

### Text setting

```bash
python -m experiments.src.baselines.run --setting text
```

### Multimodal setting

```bash
python -m experiments.src.baselines.run --dataset amazon --setting multimodal \
    --visual-assets-dir <your_visual_assets_dir>
```

### Both datasets and settings at once

```bash
python -m experiments.src.baselines.run --setting both \
    --visual-assets-dir <your_visual_assets_dir>
```

To run a single dataset instead of both, add `--dataset amazon` or `--dataset esci` to any of the commands above.

### Options

| Flag | Description |
| --- | --- |
| `--dataset` | `amazon`, `esci`, or `both` (default: `both`) |
| `--setting` | `text`, `multimodal`, or `both` (default: `both`) |
| `--visual-assets-dir` | Root directory of visual assets. Expected structure: `<dir>/<dataset>/patents/` and `<dir>/<dataset>/products/`. Required for multimodal setting. No default; path depends on your local storage. |
| `--no-mask` | Use plain (unmasked) text fields instead of `*_masked` variants. |
