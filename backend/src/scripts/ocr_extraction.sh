#!/bin/bash

# Default values
V_START=1
V_END=9999
K_START=1
K_END=26
NUM_GPUS=1
CONFIG_PATH="./src/configs/ocr.yaml"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --V_start) V_START="$2"; shift ;;
        --V_end) V_END="$2"; shift ;;
        --K_start) K_START="$2"; shift ;;
        --K_end) K_END="$2"; shift ;;
        --num_gpus) NUM_GPUS="$2"; shift ;;
        --config) CONFIG_PATH="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "Starting OCR Extraction Pipeline..."
echo "Config: $CONFIG_PATH"
echo "K_start: $K_START, K_end: $K_END"
echo "V_start: $V_START, V_end: $V_END"
echo "Num GPUs: $NUM_GPUS"

python -m src.services.vision.ocr.pipeline \
    --config "$CONFIG_PATH" \
    --K_start "$K_START" \
    --K_end "$K_END" \
    --V_start "$V_START" \
    --V_end "$V_END" \
    --num_gpus "$NUM_GPUS"
