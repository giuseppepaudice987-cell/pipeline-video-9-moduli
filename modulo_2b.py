# modulo_2b.py
"""
Step 2B — Estensione temporale del video grezzo (2A) su Kaggle T4 16GB.

Strategia: SVD-XT (img2vid) incatenato sull'ultimo frame + crossfade
sull'overlap per nascondere lo stacco tra chunk.

Dipendenze (cella Kaggle una tantum, NON dentro il modulo):
    !pip install -q -U diffusers transformers accelerate imageio imageio-ffmpeg
"""

from __future__ import annotations

import gc
import os
import time
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline

from config import PATHS, pulisci_memoria

# ----------------------------- Manopole -----------------------------
MODEL_ID          = "stabilityai/stable-video-diffusion-img2vid-xt"
TARGET_SECONDS    = 6.0     # durata finale desiderata
FPS_FALLBACK      = 8.0     # se i metadati del video non reportano fps
FRAMES_PER_CHUNK  = 25      # nativo SVD-XT: non alzare
OVERLAP           = 5       # frame di crossfade coda/testa
MAX_CHUNKS        = 6       # guardrail anti-loop e anti-drift d'identità
SEED              = 42
MOTION_BUCKET_ID  = 110     # < 127 default: moto più calmo, meno ghosting
NOISE_AUG         = 0.02    # basso = fedeltà al frame di conditioning
STEPS             = 25
DECODE_CHUNK_SIZE = 2       # 2 = safe su T4; 4 se hai headroom
FP32_FALLBACK     = False   # True solo se compaiono frame neri/NaN (fp16 su T4)
# ---------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[2B] {msg}", flush=True)


def _leggi_video(path: str) -> Tuple[List[np.ndarray], float]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input 2A mancante: {path}. Esegui prima modulo_2a.")
    reader = imageio.get_reader(path, "ffmpeg")
    try:
        fps = float(reader.get_meta_data().get("fps") or FPS_FALLBACK)
        frames = [np.asarray(f, dtype=np.uint8) for f in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"Video 2A vuoto: {path}")
    if not np.isfinite(fps) or fps <= 0 or fps > 120:
        fps = FPS_FALLBACK
    return frames, fps


def _dims_target(w: int, h: int) -> Tuple[int, int]:
    # Canonici SVD, multipli di 64: niente resize impliciti in export.
    return (1024, 576) if w >= h else (576, 1024)


def _adatta(frame: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    """Cover + center-crop al canvas SVD, preservando l'aspect ratio."""
    tw, th = size
    img = Image.fromarray(frame)
    w, h = img.size
    if (w, h) == (tw, th):
        return img
    s = max(tw / w, th / h)
    img = img.resize((max(tw, round(w * s)), max(th, round(h * s))), Image.LANCZOS)
    w, h = img.size
    left, top = (w - tw) // 2, (h - th) // 2
    return img.crop((left, top, left + tw, top + th))


def _carica_pipeline() -> StableVideoDiffusionPipeline:
    dtype = torch.float32 if FP32_FALLBACK else torch.float16
    kwargs = {"torch_dtype": dtype}
    if not FP32_FALLBACK:
        kwargs["variant"] = "fp16"
    pipe = StableVideoDiffusionPipeline.from_pretrained(MODEL_ID, **kwargs)
    pipe.set_progress_bar_config(leave=False)
    if FP32_FALLBACK:
        pipe.enable_sequential_cpu_offload()   # lento ma stabile
    else:
        pipe.enable_model_cpu_offload()        # UNet/VAE/encoder avvicendati in VRAM
    if hasattr(pipe.unet, "enable_forward_chunking"):
        pipe.unet.enable_forward_chunking()    # spezza la FF attention temporale
    return pipe


def _genera_chunk(pipe, cond: Image.Image, fps: float,
                  size: Tuple[int, int], idx: int,
                  noise_aug: float = NOISE_AUG) -> List[np.ndarray]:
    gen = torch.Generator(device="cpu").manual_seed(SEED + idx)
    out = pipe(
        cond,
        height=size[1], width=size[0],
        num_frames=FRAMES_PER_CHUNK,
        num_inference_steps=STEPS,
        fps=int(round(fps)),            # conditioning temporale coerente con l'export
        motion_bucket_id=MOTION_BUCKET_ID,
        noise_aug_strength=noise_aug,
        decode_chunk_size=DECODE_CHUNK_SIZE,
        generator=gen,
    ).frames[0]
    return [np.asarray(f, dtype=np.uint8) for f in out]


def _degenere(frames: List[np.ndarray]) -> bool:
    """Rileva chunk neri/NaN (overflow fp16 sul VAE temporal decoder)."""
    return all(float(f.mean()) < 1.0 and float(f.std()) < 1.0 for f in frames[:3])


def _fonde(seq: List[np.ndarray], nuovi: List[np.ndarray], overlap: int) -> List[np.ndarray]:
    """Crossfade lineare: nuovi[0] ≈ condizionamento, quindi si sfuma sulla coda."""
    ov = min(overlap, len(seq), len(nuovi) - 1)
    if ov <= 0:
        return seq + nuovi
    coda = seq[-ov:]
    fusi = []
    for i in range(ov):
        a = (i + 1) / (ov + 1)
        mix = (1.0 - a) * coda[i].astype(np.float32) + a * nuovi[i].astype(np.float32)
        fusi.append(mix.round().clip(0, 255).astype(np.uint8))
    return seq[:-ov] + fusi + nuovi[ov:]


def _scrivi_video(frames: List[np.ndarray], fps: float, path: str) -> None:
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=16,  # compatibile con lipsync/ffmpeg a valle
    )
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()


def main() -> str:
    t0 = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("Step 2B richiede GPU CUDA (Kaggle T4).")
    _log(f"GPU: {torch.cuda.get_device_name(0)}")

    frames, fps = _leggi_video(PATHS["2A_VIDEO"])
    h, w = frames[0].shape[:2]
    size = _dims_target(w, h)
    # Normalizzo TUTTI i frame al canvas target: crossfade e concat richiedono shape uniformi.
    frames = [np.asarray(_adatta(f, size)) for f in frames]
    target_frames = max(len(frames), int(round(TARGET_SECONDS * fps)))
    _log(f"Input {len(frames)} frame @ {fps:.2f}fps → target {target_frames} "
         f"({TARGET_SECONDS:.1f}s), canvas {size[0]}x{size[1]}")

    # Pipeline caricata solo se serve davvero estendere.
    pipe = None if len(frames) >= target_frames else _carica_pipeline()

    chunks = 0
    while len(frames) < target_frames and chunks < MAX_CHUNKS:
        cond = Image.fromarray(frames[-1])
        nuovi = _genera_chunk(pipe, cond, fps, size, idx=chunks)
        if _degenere(nuovi):
            _log("Chunk degenere: retry con noise_aug=0 e seed diverso…")
            nuovi = _genera_chunk(pipe, cond, fps, size, idx=chunks + 1000, noise_aug=0.0)
            if _degenere(nuovi):
                raise RuntimeError("SVD produce frame neri: imposta FP32_FALLBACK=True.")
        frames = _fonde(frames, nuovi, OVERLAP)
        chunks += 1
        peak = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()
        _log(f"Chunk {chunks}: tot {len(frames)} frame | picco VRAM {peak:.1f} GB")
        gc.collect()
        torch.cuda.empty_cache()

    frames = frames[:target_frames]
    if len(frames) < target_frames:
        _log(f"MAX_CHUNKS raggiunto: output {len(frames)/fps:.1f}s "
             f"invece di {TARGET_SECONDS:.1f}s")

    _scrivi_video(frames, fps, PATHS["2B_EXTENDED"])
    _log(f"Salvato {PATHS['2B_EXTENDED']} — {len(frames)} frame, "
         f"{len(frames)/fps:.1f}s, {time.time()-t0:.0f}s totali")

    if pipe is not None:
        del pipe  # il `del model` dentro pulisci_memoria rimuove solo il ref locale:
                  # libero qui il riferimento vero, poi gc + empty_cache.
    pulisci_memoria()
    return PATHS["2B_EXTENDED"]


if __name__ == "__main__":
    main()
