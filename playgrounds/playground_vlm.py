import torch
import argparse
from utils import clean_gpu_cache

import sys
sys.path.append('/app/dev/')
from models.qwenvl2_5 import QwenVL2_5KVOpt
from models.qwenvl import QwenVLKVOpt
from models.llava import LlavaKVOpt   

MODELS_DICT = {
    'qwenvl2_5': QwenVL2_5KVOpt,
    'qwenvl': QwenVLKVOpt,
    'llava': LlavaKVOpt,
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
        image_path=args.image_path,
        device_num=args.device_num,
    )
    output = model.generate(
        inputs=inputs,
        approach='opt',
        opt_steps=7,
        opt_lr=0.0003,
        lambda_kl=0.1,
        max_new_tokens=args.max_new_tokens,
        plot=False,
    )

    print(f"Model response's: {output['response'].strip()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        help='models: qwenvl, qwenvl2_5, llava',
        choices=['qwenvl', 'llava', 'qwenvl2_5']
    )  
    parser.add_argument(
        "--image_path",
        type=str,
        default='/app/datasets/coco/val2014',
        help="Path for image to process",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image in detail.",
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