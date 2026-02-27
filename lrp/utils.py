import torch
import gc
import librosa
import numpy as np
import matplotlib.pyplot as plt

def clean_gpu_cache():
    torch.cuda.set_device(0)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("alloc:", torch.cuda.memory_allocated()/1e9, "GB")
    print("reserved:", torch.cuda.memory_reserved()/1e9, "GB")

def compute_logemel_db(
    wav_path: str,
    sr: int = 16_000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float | None = None,
    top_db: float = 80.0,
    pad: bool = False,
    pad_value: float | None = None,
):
    y, _ = librosa.load(wav_path, sr=sr)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
        center=True,
    )

    # dB scale with limited dynamic range
    logmel_db = librosa.power_to_db(S, ref=np.max, top_db=top_db)

    if pad:
        # fix length to 3000 frames (time axis)
        T = logmel_db.shape[1]
        target = 3000
        if pad_value is None:
            pad_value = -float(top_db)

        if T == target:
            out = logmel_db
        elif T > target:
            out = logmel_db[:, :target]
        else:
            pad_T = target - T
            pad = np.full((logmel_db.shape[0], pad_T), pad_value, dtype=logmel_db.dtype)
            logmel_db = np.concatenate([logmel_db, pad], axis=1)

    return logmel_db.astype(np.float32)

def plot_logmel(spect:np.ndarray, sr: int = 16_000, hop_length: int = 160):
    n_frames = spect.shape[1]
    dur_sec = (n_frames - 1) * hop_length / sr
    f_max = sr / 2.0

    plt.figure(figsize=(10, 4), dpi=120)
    extent = [0.0, dur_sec, 0.0, f_max]
    im = plt.imshow(spect, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("STFT (log|X|+1)")
    plt.colorbar(im, label="Amplitude (log1p)")

    plt.tight_layout()
    plt.show()

def plot_logmel_and_relevance(spect:np.ndarray, relevance:np.ndarray, sr: int = 16_000, hop_length: int = 160):
    n_frames = spect.shape[1]
    dur_sec = (n_frames - 1) * hop_length / sr
    f_max = sr / 2.0

    plt.figure(figsize=(10, 4), dpi=120)
    extent = [0.0, dur_sec, 0.0, f_max]
    
    im = plt.imshow(relevance[:, :spect.shape[1]], origin="lower", aspect="auto", cmap='bwr', extent=extent, vmin=-1, vmax=1)
    plt.imshow(spect, origin="lower", aspect="auto", cmap='gray', extent=extent, alpha=0.2)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Audio Relevance")
    plt.colorbar(im, label="Relevance values")

    plt.tight_layout()
    plt.show()



