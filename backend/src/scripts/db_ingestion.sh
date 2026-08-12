#!/bin/bash

# Configuration
CONFIG_FILE="./src/configs/ingestion.yaml"
MODELS=("hf_clip_L" "hf_clip_H" "hf_siglip")

echo "Starting DB Ingestion Pipeline..."
echo "Config: $CONFIG_FILE"

for MODEL_NAME in "${MODELS[@]}"; do
    echo "Ingesting Model: $MODEL_NAME"
    
    python -m src.services.ingestion.pipeline \
        --config "$CONFIG_FILE" \
        --model_name "$MODEL_NAME"
done

echo "All Ingestion Pipelines Completed!"
