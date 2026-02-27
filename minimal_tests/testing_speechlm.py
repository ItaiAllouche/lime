import argparse
import torch

import sys
sys.path.append('/app/dev/')
from models.speechlms.qwen2_audio.qwen2_audio import Qwen2AudioKVOpt
from models.speechlms.salmonn7b.salmonn7b import Salmonn7B
from utils import clean_gpu_cache

MODELS = {
    'qwen2audio': Qwen2AudioKVOpt,
    'salmonn7b': Salmonn7B,
}

parser = argparse.ArgumentParser(description='Testing SpeechLMs KV Optimazation in Infernece')
parser.add_argument('--model', help='models: qwen2audio, salmonn7b', choices=['qwen2audio', 'salmonn7b'])
parser.add_argument('--device', help='device (gpu) number', type=int, default=0)

args = parser.parse_args()

device_num = args.device
clean_gpu_cache(device_num=device_num)

device = f'cuda:{device_num}' if torch.torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

if args.model == 'qwen2audio':
    model = MODELS[args.model](verbose=True).to(device, dtype=torch.bfloat16)
else:
     model = MODELS[args.model](device_num=device_num)

wav_path = '/app/datasets/librispeech/test-clean/237/126133/237-126133-0003.wav'
instruction = 'Transcribe the audio. Output only the raw transcription, no commentary.'

wav_instruct_dict = {
    'water': {
        'wav':'/app/datasets/air_bench/Foundation/Sound_AQA_clothoaqa/river_mouth3.wav',
        'instruction': 'How many times exactly does the water splash?'
    },
    'gunshot': {
        'wav': '/app/datasets/air_bench/Foundation/Sound_AQA_avqa/3049.flac',
        'instruction': 'how many gun shots there are in the audio?'
    },
    'cat_meowing': {
        'wav': '/app/datasets/air_bench/Foundation/Audio_Grounding_AudioGrounding/Y_GI7meqlYZk.flac',
        'instruction': 'How many dog barks there are in the audio?'
    },
    'police': {
        'wav': '/app/datasets/air_bench/Foundation/Sound_AQA_avqa/3593.flac',
        'instruction': 'What are the police officers saying?'
    },
}

inputs_cat = model.get_inputs_for_forward(
    instruction=wav_instruct_dict['cat_meowing']['instruction'],
    wav_path=wav_instruct_dict['cat_meowing']['wav'],
    device_num=device_num
)
inputs_cat_describe = model.get_inputs_for_forward(
    instruction='Describe the audio in details.',
    wav_path=wav_instruct_dict['cat_meowing']['wav'],
    device_num=device_num
)
inputs_gunshot = model.get_inputs_for_forward(
    instruction=wav_instruct_dict['gunshot']['instruction'],
    wav_path=wav_instruct_dict['gunshot']['wav'],
    device_num=device_num
)
inputs_police = model.get_inputs_for_forward(
    instruction=wav_instruct_dict['police']['instruction'],
    wav_path=wav_instruct_dict['police']['wav'],
    device_num=device_num
)

approach='opt'
opt_steps = [5, 7]
opt_lrs = [7e-5, 2e-4, 7e-4]
lambads_kl = [7e-3, 1e-2, 5e-2, 1e-1, 7e-1, 1]
lambdas_relevance_multimodal = [1]#[1e-1, 7e-1, 1]
lambdas_relevance_text = [1]#[5e-2, 1e-1, 7e-1, 1]

for opt_step in opt_steps:
    for opt_lr in opt_lrs:
        for lambda_kl in lambads_kl:
            for lambda_relevance_multimodal in lambdas_relevance_multimodal:
                for lambda_relevance_text in lambdas_relevance_text:
                    clean_gpu_cache(device_num=device_num)
                    print('\n')
                    print(wav_instruct_dict['cat_meowing']['instruction'])
                    cat_output, _, _ = model.generate(
                        inputs=inputs_cat,
                        approach=approach,
                        opt_steps=opt_step,
                        opt_lr=opt_lr,
                        lambda_kl=lambda_kl,
                        lambda_relevance_multimodal=lambda_relevance_multimodal,
                        lambda_relevance_text=lambda_relevance_text,
                        max_new_tokens=30,
                        plot=False
                    )
                    print(f'------------------------------------------------------------------------------------------')
                    print('Describe the audio in details.')
                    cat_descrbie_output, _, _ = model.generate(
                        inputs=inputs_cat_describe,
                        approach=approach,
                        opt_steps=opt_step,
                        opt_lr=opt_lr,
                        lambda_kl=lambda_kl,
                        lambda_relevance_multimodal=lambda_relevance_multimodal,
                        lambda_relevance_text=lambda_relevance_text,
                        max_new_tokens=50,
                        plot=False
                    )
                    print(f'------------------------------------------------------------------------------------------')
                    print(wav_instruct_dict['gunshot']['instruction'])
                    gunshot_output, _, _ = model.generate(
                        inputs=inputs_gunshot,
                        approach=approach,
                        opt_steps=opt_step,
                        opt_lr=opt_lr,
                        lambda_kl=lambda_kl,
                        lambda_relevance_multimodal=lambda_relevance_multimodal,
                        lambda_relevance_text=lambda_relevance_text,
                        max_new_tokens=30,
                        plot=False
                    )
                    print(f'------------------------------------------------------------------------------------------')
                    print(wav_instruct_dict['police']['instruction'])
                    police_output, _, _ = model.generate(
                        inputs=inputs_police,
                        approach=approach,
                        opt_steps=opt_step,
                        opt_lr=opt_lr,
                        lambda_kl=lambda_kl,
                        lambda_relevance_multimodal=lambda_relevance_multimodal,
                        lambda_relevance_text=lambda_relevance_text,
                        max_new_tokens=30,
                        plot=False
                    )
                    print(f'Outputs:\n{cat_output}\n{cat_descrbie_output}\n{gunshot_output}\n{police_output}')
                    print(f'\n------------------------------------------------------------------------------------------')
                    print(f'opt_step={opt_step} | opt_lr={opt_lr} | lambda_kl={lambda_kl} | lambda_relevance_multimodal={lambda_relevance_multimodal} | lambda_relevance_text={lambda_relevance_text}')
                    print(f'------------------------------------------------------------------------------------------\n')                
