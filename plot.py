import matplotlib.pyplot as plt
import numpy as np
from typing import Literal, Optional

def plot_relevance(
        relevances,
        modality_start: int,
        modality_end: int,
        example: str,
        target_step: Optional[int] = None
    ):
    fig, ax = plt.subplots(figsize=(8, 4))

    if target_step is None:
        idx_list = range(len(relevances))
    else:
        idx_list = [target_step]

    for step in idx_list:
        x = np.arange(len(relevances[step]))
        ax.plot(x,relevances[step], label=f'step {step}')

    ax.axvline(modality_start, linestyle="--", color="orange", linewidth=1)
    ax.axvline(modality_end, linestyle="--", color="orange", linewidth=1)
    ymin, ymax = ax.get_ylim()
    ax.text(modality_start, int(ymax*0.95), '<SOA>', color='black', ha='center', fontsize=6.5)
    ax.text(modality_end, int(ymax*0.95), '<EOA>', color='black', ha='center', fontsize=6.5)
    ax.set_xlabel("Token idx")
    ax.set_ylabel("Relevance")
    ax.set_title(f"Relevance Score Per Token")
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.0), borderaxespad=0.0, frameon=True)
    fig.subplots_adjust(right=0.72)

    # fig.tight_layout()
    plt.show()
    fig.savefig(f'/app/dev/figs/relevance/{example}.pdf', format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_attn_maps(
        attentions, 
        target_step:int,
        layers: list,
        modality_start: int,
        modality_end: int,
        example: str
    ):

    num_layers = len(layers)

    n_rows, n_cols = num_layers // 4, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4))
    axes = axes.flatten()

    for i, layer_idx in enumerate(layers):
        ax = axes[i]
        attn = attentions[target_step][layer_idx].squeeze()

        im = ax.imshow(attn.detach().cpu().float().numpy(), origin="upper", aspect="auto")

        # Mark audio region (vertical & horizontal boundaries)
        ax.axvline(modality_start - 0.5, linestyle="--", color="orange", linewidth=0.5)
        ax.axvline(modality_end   - 0.5, linestyle="--", color="orange", linewidth=0.5)
        ax.axhline(modality_start - 0.5, linestyle="--", color="orange", linewidth=0.5)
        ax.axhline(modality_end   - 0.5, linestyle="--", color="orange", linewidth=0.5)

        ax.set_title(f"Layer {layer_idx}", fontsize=9)
        L = attn.shape[0]
        tick_spacing = 20
        ticks = np.arange(0, L, tick_spacing)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks, fontsize=6, rotation=90)
        ax.set_yticklabels(ticks, fontsize=6)

    for k in range(num_layers, n_rows * n_cols):
        fig.delaxes(axes[k])

    
    # colorbar in its own axis on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Attention weight")
    fig.suptitle(f"Qwen2-Audio — Self-attention maps - Head: mean")

    fig.text(0.5, 0.01, "Key ", ha="center", fontsize=10)
    fig.text(0.01, 0.5, "Query ", va="center", rotation="vertical", fontsize=10) 

    plt.show()
    fig.savefig(f'/app/dev/figs/attention/{example}.pdf', format="pdf", dpi=300, bbox_inches="tight")

def plot_heatmaps(attentions, audio_start: int, audio_end: int, text_part: Literal['all', 'history', 'prompt'] = 'all'):
    layers = len(attentions)
    heads = attentions[0].shape[1]
    
    audio_mean = np.zeros((layers, heads), dtype=np.float32)
    text_mean = np.zeros((layers, heads), dtype=np.float32)

    for layer in range(layers):
        attn_per_layer = attentions[layer].squeeze()
        for head in range(heads):
            audio_region = attn_per_layer[head, -1, audio_start:audio_end+1]
            prompt_region = attn_per_layer[head, -1, :audio_start]
            history_region = attn_per_layer[head, -1, audio_end+1:]
            audio_mean[layer, head] = audio_region.mean()
            
            if text_part == 'all':
                text_mean[layer, head] = (prompt_region.sum() + history_region.sum()) / (prompt_region.shape[0] + history_region.shape[0])
            elif text_part == 'history':
                text_mean[layer, head] = history_region.mean()
            else:
                text_mean[layer, head] = prompt_region.mean()

    vmin = min(audio_mean.min(), text_mean.min())
    vmax = max(audio_mean.max(), text_mean.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # audio region heatmap
    im1 = axes[0].imshow(audio_mean, aspect='auto', origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title("Mean Attention Weights over Audio Tokens")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    plt.colorbar(im1, ax=axes[0])

    # text region heatmap
    im2 = axes[1].imshow(text_mean, aspect='auto', origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title("Mean Attention Weights over Text Tokens")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    plt.colorbar(im2, ax=axes[1])

    # --- optional: better tick labeling ---
    axes[0].set_xticks(range(0, heads, max(1, heads // 8)))
    axes[1].set_xticks(range(0, heads, max(1, heads // 8)))
    axes[0].set_yticks(range(0, layers, max(1, layers // 8)))
    axes[1].set_yticks(range(0, layers, max(1, layers // 8)))

    plt.tight_layout()
    plt.show()
    fig.savefig('/app/dev/figs/layers_heads_heatmaps.pdf', format="pdf", dpi=300, bbox_inches="tight")

def plot_attention_weights(attentions, audio_start: int, audio_end: int, layer: int = 31):
    attn = attentions[layer].squeeze()
    attn_segment = attn.mean(dim=0)
    
    audio_segment = attn_segment[-1, audio_start:audio_end+1]
    history_segment = attn_segment[-1, audio_end+1:]
    prompt_segment = attn_segment[-1, :audio_start]

    tensors = {audio_segment: '<AUDIO>', history_segment: '<HISTORY>', prompt_segment: '<PROMPT>'}

    plt.figure(figsize=(8, 4))

    for t, label in tensors.items():
        x = torch.arange(t.shape[0])
        plt.plot(x.cpu().numpy(),
                t.cpu().numpy(),
                label=f'{label}')

    plt.xlabel("Token idx")
    plt.ylabel("Attention Weight")
    plt.title(f"Attention Weights Of Layer: {layer} For All Tokens (avg over all heads)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('/app/dev/figs/attention_weights.pdf', format="pdf", dpi=300, bbox_inches="tight")