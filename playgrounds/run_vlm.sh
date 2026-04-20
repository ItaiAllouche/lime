#!/bin/sh

uv run vlm.py \
  --model llava \
  --image_path ../examples/images/COCO_val2014_000000052462.jpg \
  --prompt "Describe the image in detail." \
  --max_new_tokens 150 \
  --device_num 0