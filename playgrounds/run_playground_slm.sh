#!/bin/sh

MODEL=qwen2audio
AUDIO_PATH=/home/itaiallouche/datasets/coco/val2014/COCO_val2014_000000100000.jpg
MAX_NEW_TOKENS=3

uv run playground_slm.py \
  --model $MODEL \
  --audio_path $AUDIO_PATH \
  --prompt "Transcribe the audio." \
  --max_new_tokens $MAX_NEW_TOKENS \
  --device_num 0