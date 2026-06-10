#!/bin/bash
# Download and convert the ESCI dataset to PT2PR pipeline format.
# Requires: git, git-lfs, python, pandas, pyarrow

set -e

OUTPUT_DIR="${1:-data/external/esci}"
REPO_DIR=$(mktemp -d)

echo "Checking git-lfs..."
if ! command -v git-lfs &> /dev/null; then
    echo "Error: git-lfs is not installed."
    echo "Install it with: sudo apt install git-lfs"
    exit 1
fi

echo "Cloning ESCI repository (with LFS)..."
git clone --depth 1 https://github.com/amazon-science/esci-data.git "$REPO_DIR"
git -C "$REPO_DIR" lfs pull

echo "Converting to pipeline format..."
python -m scripts.convert_esci \
    --products "$REPO_DIR/shopping_queries_dataset/shopping_queries_dataset_products.parquet" \
    --output   "$OUTPUT_DIR/products.jsonl.gz"

echo "Cleaning up..."
rm -rf "$REPO_DIR"

echo "Done. Output saved to $OUTPUT_DIR"
