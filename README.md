# Official PyTorch Implementation Of The Paper "Mitigating Multimodal LLMs Hallucinations via Relevance Propagation at Inference Time"

<h1 align="center">
  <img src="assets/fig1.jpg" height="370">
  <img src="assets/fig2.jpg" height="370">
</h1>

<p align="center">

> <strong>Abstract.</strong> *Multimodal large language models (MLLMs) have revolutionized the landscape of AI, demonstrating impressive capabilities in tackling complex vision and audio-language tasks.
>However, a critical challenge remains: these models often suffer from hallucinations, generating outputs that diverge from the provided perceptual inputs.
>This tendency stems from an inherent imbalance in modality utilization during inference, where the dominance of textual tokens undermines the potential of perceptual inputs.
>As a result, the model frequently resorts to textual language priors at the expense of grounded evidence.
>To tackle this issue, we propose Learning Inference-time Modality Enhancement (LIME), a training-free framework designed to bolster multimodal grounding by explicitly enhancing modality usage during decoding.
>LIME leverages Layer-wise Relevance Propagation (LRP) to quantify token-level contributions and defines a relevance-based objective that promotes increased reliance on perceptual inputs.
>This objective is enforced through inference-time updates to the model's key-value representations, without modifying model parameters or requiring additional training data. 
>We evaluate LIME across multiple multimodal benchmarks in both vision and audio domains, demonstrating consistent reductions in hallucinations and enhanced grounding while preserving generation quality.
>Further analysis shows that LIME increases modality contribution and produces more localized and semantically aligned relevance patterns.*
</p>

## Setup

### Prerequisites
- Python 3.11+
- PyTorch 2.1.2
- CUDA 12.0 (optional, CPU support available)

### Installation

Install uv on your machine, see intrucitons [here](https://docs.astral.sh/uv/getting-started/installation/).

Clone and setup the repository:

```bash
# clone cmd
cd lime
uv sync
```

## Inference

We provide both CLI and Python interfaces for running inference with LIME.

### CLI Usage (example for LLaVA)

```bash
cd playgrounds
uv vlm.py \
    --model llava \
    --prompt "What is in this image?" \
    --image_path path/to/image \
    --device_num 0 \
    --max_new_tokens 50
```

### Python Usage (example for Qwen2Audio)
```python
import torch
from models.qwen2_audio import Qwen2AudioLIME

# initialize model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = Qwen2AudioLIME(verbose=True).to(device, dtype=torch.bfloat16)

# prepare inputs
inputs = model.get_inputs_for_forward(
    instruction="What do you hear in this audio?",
    wav_path="path/to/audio/file",
    device_num=0
)

# generate with LIME
output = model.generate(
    inputs=inputs,
    max_new_tokens=50,
    verbose=False
)

print(f"Response: {output['response']}")
```
## TODO

- [ ] Add SALMONN implementation  
  *(Not yet included due to environment constraints — SALMONN depends on a specialized setup that is currently incompatible with the rest of the repository.)*
- [ ] Add citation
- [ ] Add ArXiv patch  

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
