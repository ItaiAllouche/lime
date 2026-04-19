#!/bin/sh

MODEL=llava
IMAGE_PATH=/home/itaiallouche/datasets/coco/val2014/COCO_val2014_000000100000.jpg
MAX_NEW_TOKENS=50

uv run playground_vlm.py \
  --model $MODEL \
  --image_path $IMAGE_PATH \
  --prompt "Describe the image in detail." \
  --max_new_tokens $MAX_NEW_TOKENS \
  --device_num 0