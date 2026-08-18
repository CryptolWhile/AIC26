#!/bin/bash

echo "Starting OCR DB Ingestion Pipeline..."
echo "Config: ./src/configs/ocr_ingestion.yaml"

python -m src.services.ingestion.ocr_pipeline \
    --config ./src/configs/ocr_ingestion.yaml

echo "OCR Ingestion Pipeline Completed!"
