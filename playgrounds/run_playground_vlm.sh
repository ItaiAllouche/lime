#!/bin/sh

MODEL=llava
IMAGE_PATH=../examples/images/COCO_val2014_000000052462.jpg
MAX_NEW_TOKENS=150
DEVICE_NUM=3
PROMPT="Describe the image in detail."

uv run playground_vlm.py \
  --model $MODEL \
  --image_path $IMAGE_PATH \
  --prompt $PROMPT \
  --max_new_tokens $MAX_NEW_TOKENS \
  --device_num $DEVICE_NUM