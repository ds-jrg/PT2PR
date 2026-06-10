#!/bin/bash
# Download the Amazon Reviews 2023 dataset.
set -e

OUTPUT_DIR="${1:-data/external/amazon}"
mkdir -p "$OUTPUT_DIR"

BASE_URL="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories"

CATEGORIES=(
    "All_Beauty" "Amazon_Fashion" "Appliances" "Arts_Crafts_and_Sewing"
    "Automotive" "Baby_Products" "Beauty_and_Personal_Care" "Books"
    "CDs_and_Vinyl" "Cell_Phones_and_Accessories" "Clothing_Shoes_and_Jewelry"
    "Digital_Music" "Electronics" "Gift_Cards" "Grocery_and_Gourmet_Food"
    "Handmade_Products" "Health_and_Household" "Health_and_Personal_Care"
    "Home_and_Kitchen" "Industrial_and_Scientific" "Kindle_Store"
    "Magazine_Subscriptions" "Movies_and_TV" "Musical_Instruments"
    "Office_Products" "Patio_Lawn_and_Garden" "Pet_Supplies" "Software"
    "Sports_and_Outdoors" "Subscription_Boxes" "Tools_and_Home_Improvement"
    "Toys_and_Games" "Video_Games" "Unknown"
)

for CATEGORY in "${CATEGORIES[@]}"; do
    FILE="meta_${CATEGORY}.jsonl.gz"
    DEST="$OUTPUT_DIR/$FILE"
    if [ -f "$DEST" ]; then
        echo "Skipping $FILE (already exists)"
        continue
    fi
    echo "Downloading $FILE..."
    wget -q --show-progress -O "$DEST" "$BASE_URL/$FILE"
done

echo "Done. Files saved to $OUTPUT_DIR"
