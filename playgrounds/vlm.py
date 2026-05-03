import torch
import argparse

import sys
sys.path.append('../')
from utils import clean_gpu_cache

# a workaround for loading qwenvl
# download Qwen-VL-Chat model cache if needed (only for qwenvl model)
if len(sys.argv) > 1 and '--model' in sys.argv:
    model_idx = sys.argv.index('--model') + 1
    if model_idx < len(sys.argv) and sys.argv[model_idx] == 'qwenvl':
        print("Caching Qwen-VL-Chat model...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True)
            del model, tokenizer
        except Exception as e:
            pass

from models.qwenvl2_5 import QwenVL2_5LIME
from models.qwenvl import QwenVLLIME
from models.llava import LlavaLIME  

MODELS_DICT = {
    'qwenvl2_5': QwenVL2_5LIME,
    'qwenvl': QwenVLLIME,
    'llava': LlavaLIME,
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
    model = MODELS_DICT[args.model](verbose=args.verbose).to(device, dtype=torch.bfloat16)
    inputs = model.get_inputs_for_forward(
        instruction=args.prompt,
        image_path=args.image_path,
        device_num=args.device_num,
    )

    print(f'Generating output...')
    output = model.generate(
        inputs=inputs,
        max_new_tokens=args.max_new_tokens,
        verbose=False,
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
        default='../examples/images/COCO_val2014_000000052462.jpg',
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
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Verbose for model initialization",
    )    

    args = parser.parse_args()

    main(args)