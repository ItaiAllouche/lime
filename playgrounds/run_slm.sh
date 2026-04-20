#!/bin/sh

uv run slm.py \
  --model qwen2audio \
  --audio_path ../examples/audios/672-122797-0000.wav \
  --prompt "Transcribe the audio." \
  --max_new_tokens 50 \
  --device_num 0