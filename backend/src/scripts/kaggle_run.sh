#!/bin/bash

# Ensure we are in the backend directory
# cd /kaggle/working/backend-code

# Shot extraction using 2 GPUs on L27 (Folder 27)
python -m src.services.processing.pipeline shot_extraction --config ./src/configs/processing.yaml --K_start 27 --K_end 27 --num_gpus 2

# Keyframe extraction using 2 GPUs
# python -m src.services.processing.pipeline keyframe_extraction --config ./src/configs/processing.yaml --K_start 27 --K_end 27 --num_gpus 2
