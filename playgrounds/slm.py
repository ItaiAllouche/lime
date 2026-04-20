import torch
import argparse

import sys
sys.path.append('../')
from utils import clean_gpu_cache
from models.qwen2_audio import Qwen2AudioLIME


MODELS_DICT = {
    'qwen2audio': Qwen2AudioLIME,
}

def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_num)
        device = f"cuda:{args.device_num}"
    else:
        device = "cpu"

    print((f"Using device: {device}"))
    clean_gpu_cache(device_num=args.device_num, print_stats=False)
    print(f"Cleaned GPU cache before loading model on GPU {args.device_num}")    

    print(f'Loading {args.model}...')
    model = MODELS_DICT[args.model]().to(device, dtype=torch.bfloat16)
    inputs = model.get_inputs_for_forward(
        instruction=args.prompt,
        wav_path=args.audio_path,
        device_num=args.device_num,
    )

    print(f'\nGenerating output...')
    output = model.generate(
        inputs=inputs,
        max_new_tokens=args.max_new_tokens,
        plot=False,
    )

    print(f"Model response's: {output['response'].strip()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        help='models: qwen2audio, salmonn',
        choices=['qwen2audio', 'salmonn']
    )  
    parser.add_argument(
        "--audio_path",
        type=str,
        default='/app/datasets/coco/val2014',
        help="Path for audio to process",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Transcribe the audio.",
        help="Prompt used for caption generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--device_num",
        type=int,
        default=0,
        help="Starting GPU device index",
    )

    args = parser.parse_args()

    main(args)