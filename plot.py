import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import math
from PIL import Image
import torch
import torch.nn.functional as F
import os
import librosa

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
    fig.savefig(f'./figs/relevance/{example}.pdf', format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_relevance_vlm(
        image_path: str,
        relevance: np.ndarray,
        token: str,
        modality_bos_idx: int,
        modality_eos_idx: int,
        step: int,
        out_dir: str,
        alpha: float = 0.7
):
    def relevance_to_grid(relevance_1d: np.ndarray):
        n = relevance_1d.shape[0]
        side = int(math.sqrt(n))
        if side * side != n:
            side = int(math.ceil(math.sqrt(n)))
            pad = side * side - n
            relevance_1d = np.pad(relevance_1d, (0, pad), constant_values=0.0)
        return relevance_1d.reshape(side, side)

    rel = np.asarray(relevance, dtype=np.float32)
    image_rel = rel[modality_bos_idx:modality_eos_idx + 1]
    grid = relevance_to_grid(image_rel)
    
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    hm = torch.tensor(grid, dtype=torch.float32)[None, None, :, :]
    hm = F.interpolate(hm, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-12)
    
    figsize = (4,4)
    # save original image once, with same canvas settings as heatmap output
    original_out_path = f'{out_dir}/original.pdf'
    if not os.path.exists(original_out_path):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image_np)
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(original_out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    out_path = f'{out_dir}/step_{step}_{token}.pdf'

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_np)
    ax.imshow(hm, cmap="jet", alpha=alpha)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved step {step} in : {out_path}")

def plot_relevance_slm(
    audio_path: str,
    relevance: np.ndarray,
    token: str,
    modality_bos_idx: int,
    modality_eos_idx: int,
    step: int,
    out_dir: str,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    spec_cmap: str = "viridis",
    relevance_cmap: str = "Greens",
    waveform_alpha: float = 0.9,
    relevance_alpha_max: float = 0.7,
):
    rel = np.asarray(relevance, dtype=np.float32)

    # Audio-only relevance (exclude BOS/EOS markers)
    start = modality_bos_idx + 1
    end = modality_eos_idx
    audio_rel = rel[start:end]
    if audio_rel.size == 0:
        raise ValueError("Audio relevance slice is empty. Check modality_bos_idx/modality_eos_idx.")

    # Normalize relevance to [0, 1]
    rmin = float(audio_rel.min())
    rmax = float(audio_rel.max())
    if rmax - rmin < 1e-12:
        audio_rel = np.zeros_like(audio_rel, dtype=np.float32)
    else:
        audio_rel = (audio_rel - rmin) / (rmax - rmin)

    # Load waveform
    wav, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
    t = np.arange(wav.shape[0], dtype=np.float32) / float(sr_loaded)

    # 1) Original spectrogram only
    stft = librosa.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
    )
    mag = np.abs(stft)
    spec_db = librosa.amplitude_to_db(mag + 1e-12, ref=np.max)

    duration = len(wav) / float(sr_loaded)
    extent = [0.0, duration, 0.0, sr_loaded / 2.0]
    vmin = float(np.percentile(spec_db, 8.0))
    vmax = float(np.percentile(spec_db, 99.7))

    os.makedirs(out_dir, exist_ok=True)
    original_out_path = os.path.join(out_dir, "original_spectrogram.pdf")
    if not os.path.exists(original_out_path):
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.imshow(
            spec_db,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=spec_cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        plt.tight_layout(pad=0.15)
        fig.savefig(original_out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    # 2) Waveform + relevance map (no right y-axis, no amplitude axis)
    rel_x = np.linspace(0.0, 1.0, num=audio_rel.shape[0], dtype=np.float32)
    wav_x = np.linspace(0.0, 1.0, num=wav.shape[0], dtype=np.float32)
    rel_upsampled = np.interp(wav_x, rel_x, audio_rel)
    rel_upsampled = np.clip(rel_upsampled, 0.0, 1.0)

    token_safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in token)
    if token_safe == "":
        token_safe = "tok"
    out_path = os.path.join(out_dir, f"step_{step}_{token_safe}.pdf")

    fig, ax = plt.subplots(figsize=(9, 3.0))

    y_abs = float(np.max(np.abs(wav)) + 1e-8)
    y_min, y_max = -y_abs, y_abs

    # Relevance map spread over waveform vertical extent (time-only relevance)
    rel_map = rel_upsampled[None, :]
    im_rel = ax.imshow(
        rel_map,
        origin="lower",
        aspect="auto",
        extent=[t[0], t[-1], y_min, y_max],
        cmap=relevance_cmap,
        vmin=0.0,
        vmax=1.0,
        alpha=np.clip(rel_map * relevance_alpha_max, 0.0, 1.0),
    )

    # Waveform on top
    ax.plot(t, wav, color="black", linewidth=0.85, alpha=waveform_alpha)

    # Keep only time axis
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.2)

    # Relevance colorbar on the right
    cbar = fig.colorbar(im_rel, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Relevance")
    cbar.set_ticks([0.0, 0.5, 1.0])

    plt.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved original spectrogram in: {original_out_path}")
    print(f"Saved step {step} in: {out_path}")

def plot_ablation_accuracy(model: str):
    plt.style.use('default')
    
    if model == 'llava':
        lambdas = [0.001, 0.01, 0.1, 1.0]
        acc_delta_k = [87.8, 88.66, 89.1, 87.1]
        acc_delta_v = [88.2, 89.1, 89.7, 88]
        acc_delta_kv = [89.6, 90, 90.27, 89.75]
        baseline_acc = 83.49
    
    else:
        lambdas = [0.0007, 0.007, 0.07, 0.7]
        acc_delta_k = [61.85, 62.75, 62.9, 62.23]
        acc_delta_v = [62.75, 62.81, 62.85, 62.2]
        acc_delta_kv = [62.6, 63.63, 63.45, 62.31]
        baseline_acc = 56.19        


    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    # Background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Colors closer to the second image
    color_kv = '#5A6FF0'    # lighter blue
    color_k = '#F15A3A'    # orange-red
    color_v = '#16C79A'   # cyan-green
    color_base = '#A463F2' # purple-ish if needed elsewhere

    linewidth = 1
    markersize = 5
    axis_fontsize = 11
    labels_fontsize = 12

    # Plot lines
    ax.plot(
        lambdas, acc_delta_k,
        marker='o', markersize=markersize,
        linewidth=linewidth, color=color_k,
        label=r'$\Delta K$'
    )

    ax.plot(
        lambdas, acc_delta_v,
        marker='s', markersize=markersize,
        linewidth=linewidth, color=color_v,
        label=r'$\Delta V$'
    )

    ax.plot(
        lambdas, acc_delta_kv,
        marker='d', markersize=markersize,
        linewidth=linewidth, color=color_kv,
        label=r'$\Delta KV$'
    )

    ax.axhline(
        y=baseline_acc,
        linestyle='--',
        linewidth=linewidth,
        color=color_base,
        label='Baseline'
    )

    # Axes
    ax.set_xscale('log')
    ax.set_xlabel(r'$\lambda$', fontsize=labels_fontsize)
    
    if model == 'llava':
        ax.set_ylabel('Accuracy [%]', fontsize=labels_fontsize)

    if model == 'llava':
        ax.set_title(f'LLaVA-1.5-7B', fontsize=13)
    else:
        ax.set_title(f'Qwen2-Audio-7B-Instruct', fontsize=13)

    # Ticks
    ax.set_xticks(lambdas)

    if model == 'llava':
        ax.set_xticklabels(
            [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'],
            fontsize=axis_fontsize
        )
    else:
        ax.set_xticklabels(
            [r"$7 \times 10^{-4}$", r"$7 \times 10^{-3}$", r"$7 \times 10^{-2}$", r"$7 \times 10^{-1}$"],
            fontsize=axis_fontsize
        )
    ax.tick_params(axis='y', labelsize=axis_fontsize)

    # Grid similar to second image
    ax.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.5, color='gray')

    # Thin spines like the second plot
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('gray')

    # Legend: simple, no fancy rounded gray box
    ax.legend(
        loc='lower right',
        fontsize=axis_fontsize,
        frameon=True,
        ncol=1,
        columnspacing=1.0,
        handletextpad=0.5,
        bbox_to_anchor=(1.01, 0.05)
    )

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    plt.tight_layout()
    plt.savefig(f"/home/itaiallouche/speechLM-explainability/figs/ablation/{model}.pdf", format='pdf', bbox_inches='tight')
    plt.show()
