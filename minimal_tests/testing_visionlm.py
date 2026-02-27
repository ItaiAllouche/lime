import argparse
import torch

import sys
sys.path.append('/app/dev/')
from models.visionlms.llava.llava import LlavaKVOpt
from utils import clean_gpu_cache

MODELS = {
    'llava': LlavaKVOpt,
}

parser = argparse.ArgumentParser(description='Testing VisionLMs KV Optimazation in Infernece')
parser.add_argument('--model', help='models: llava', choices=['llava'])
parser.add_argument('--device', help='device (gpu) number', type=int, default=0)

args = parser.parse_args()

device_num = args.device
clean_gpu_cache(device_num=device_num)

device = f'cuda:{device_num}' if torch.torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

model = MODELS[args.model](verbose=True).to(device, dtype=torch.bfloat16)

image_instruct_dict = {
    'fruit': {
        'image':'/app/datasets/llava-bench/images/002.jpg',
        'instruction': '"What type of fruit is this??'
    },
    'animals': {
        'image': '/app/datasets/llava-bench/images/018.jpg',
        'instruction': 'What are the animals in the painting and what are they doing?'
    },
    'bag': {
        'image': '/app/datasets/llava-bench/images/bag.png',
        'instruction': 'Describe the image and state the number of bags'
    },
}

inputs_fruit = model.get_inputs_for_forward(
    instruction=image_instruct_dict['fruit']['instruction'],
    image_path=image_instruct_dict['fruit']['image'],
    device_num=device_num
)
inputs_animals = model.get_inputs_for_forward(
    instruction=image_instruct_dict['animals']['instruction'],
    image_path=image_instruct_dict['animals']['image'],
    device_num=device_num
)
inputs_bag = model.get_inputs_for_forward(
    instruction=image_instruct_dict['bag']['instruction'],
    image_path=image_instruct_dict['bag']['image'],
    device_num=device_num
)

approach='opt'
opt_steps = [5, 7]
opt_lrs = [7e-4, 1e-3, 5e-3, 1e-2]
lambads_kl = [5e-2,1e-1, 7e-1, 1]
lambdas_relevance_multimodal = [5e-2,1e-1, 7e-1, 1]
lambdas_relevance_text = [5e-2,1e-1, 7e-1, 1]
max_new_tokens = 170

for opt_step in opt_steps:
    for opt_lr in opt_lrs:
        for lambda_kl in lambads_kl:
            for lambda_relevance_multimodal in lambdas_relevance_multimodal:
                for lambda_relevance_text in lambdas_relevance_text:
                    clean_gpu_cache(device_num=device_num)
                    print('\n')
                    print(image_instruct_dict['fruit']['instruction'])
                    fruit_output = model.generate(
                        inputs=inputs_fruit,
                        approach=approach,
                        opt_steps=opt_step,
                        opt_lr=opt_lr,
                        lambda_kl=lambda_kl,
                        lambda_relevance_multimodal=lambda_relevance_multimodal,
                        lambda_relevance_text=lambda_relevance_text,
                        max_new_tokens=max_new_tokens,
                        plot=False
                    )
                    print(f'------------------------------------------------------------------------------------------')
                    print(image_instruct_dict['animals']['instruction'])
                    animals_output = model.generate(
                        inputs=inputs_animals,
                        approach=approach,
                        opt_steps=opt_step,
                        opt_lr=opt_lr,
                        lambda_kl=lambda_kl,
                        lambda_relevance_multimodal=lambda_relevance_multimodal,
                        lambda_relevance_text=lambda_relevance_text,
                        max_new_tokens=max_new_tokens,
                        plot=False
                    )
                    print(f'------------------------------------------------------------------------------------------')
                    print(image_instruct_dict['bag']['instruction'])
                    bag_output = model.generate(
                        inputs=inputs_bag,
                        approach=approach,
                        opt_steps=opt_step,
                        opt_lr=opt_lr,
                        lambda_kl=lambda_kl,
                        lambda_relevance_multimodal=lambda_relevance_multimodal,
                        lambda_relevance_text=lambda_relevance_text,
                        max_new_tokens=max_new_tokens,
                        plot=False
                    )
                    print(f'Outputs:\n{fruit_output}\n{animals_output}\n{bag_output}')
                    print(f'\n------------------------------------------------------------------------------------------')
                    print(f'opt_step={opt_step} | opt_lr={opt_lr} | lambda_kl={lambda_kl} | lambda_relevance={lambda_relevance_multimodal}')
                    print(f'------------------------------------------------------------------------------------------\n')                
