"""
modulo_1a.py — STEP 1A · Generazione dello sfondo (background plate)

Pipeline : AI video su Kaggle · GPU T4 16 GB (sm_75 → fp16, no bf16)
Input    : prompt testuale (+ negative prompt, seed)
Output   : config.PATHS["1A_SFONDO"]  → PNG 16:9, allineato allo Step 2A (I2V)
Dipendenze: diffusers>=0.31, transformers, accelerate, safetensors, Pillow

Uso:
    from modulo_1a import genera_sfondo
    path = genera_sfondo(prompt="...", seed=42)
"""
from __future__ import annotations

import os

# --- Ambiente: PRIMA di importare torch/diffusers ---------------------------
# La cache HF (~7 GB) non deve erodere i 20 GB di /kaggle/working (quota output).
if os.path.isdir("/kaggle"):
    os.environ.setdefault("HF_HOME", "/kaggle/temp/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import logging
import math
import time
from datetime import datetime, timezone

import torch
from PIL import Image

import config

__all__ = ["genera_sfondo"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | 1A | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("step_1a")

# --- Parametri di default ---------------------------------------------------
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SIZE_TARGET = (1024, 576)   # 16:9 esatto · SVD nativo · multiplo di 32
STEPS = 30
GUIDANCE = 6.0
SEED = 42

NEGATIVE_DEFAULT = (
    "person, people, human, man, woman, child, face, silhouette, crowd, "
    "text, watermark, signature, logo, caption, frame, border, "
    "blurry, out of focus, lowres, jpeg artifacts, deformed, distorted, oversaturated"
)

# Bucket nativi SDXL: si genera qui, poi si adatta al target (no stretching).
_BUCKETS_SDXL = (
    (1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216),
    (1344, 768), (768, 1344), (1536, 640), (640, 1536),
)


# --- Utility ----------------------------------------------------------------
def _bucket_piu_vicino(size: tuple[int, int]) -> tuple[int, int]:
    """Bucket SDXL con aspect ratio più vicino al target (distanza logaritmica)."""
    r = size[0] / size[1]
    return min(_BUCKETS_SDXL, key=lambda b: abs(math.log((b[0] / b[1]) / r)))


def _adatta_risoluzione(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Cover-resize + crop centrato: preserva l'aspect ratio, nessuno stretching."""
    tw, th = size
    if img.size == (tw, th):
        return img
    w, h = img.size
    scala = max(tw / w, th / h)
    nw, nh = max(tw, round(w * scala)), max(th, round(h * scala))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - tw) // 2, (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def _carica_pipeline(offload: bool):
    """Carica SDXL in fp16. offload=True = piano B a bassa VRAM (~6 GB)."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,   # T4: fp16 obbligato
        variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        solver_order=2,
        use_karras_sigmas=True,
    )
    pipe.set_progress_bar_config(leave=False)

    # VAE SDXL in fp16 → overflow/NaN (frame neri). Upcast fp32 in decode.
    pipe.upcast_vae()
    pipe.vae.enable_slicing()

    if offload:
        pipe.vae.enable_tiling()
        pipe.enable_attention_slicing("max")
        pipe.enable_model_cpu_offload()      # NON chiamare .to("cuda") con offload
    else:
        pipe.to("cuda")                      # torch>=2: SDPA memory-efficient di default
    return pipe


def _render(prompt, negative_prompt, seed, steps, guidance, bucket, size, offload) -> Image.Image:
    pipe = None
    try:
        pipe = _carica_pipeline(offload)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.inference_mode():
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=bucket[0],
                height=bucket[1],
                num_inference_steps=steps,
                guidance_scale=guidance,
                num_images_per_prompt=1,
                generator=generator,
                output_type="pil",
            )
        return _adatta_risoluzione(out.images[0], size)
    finally:
        del pipe
        config.pulisci_memoria()


def _salva_metadata(png_path: str, dati: dict) -> None:
    """Sidecar JSON: riproducibilità e debug della pipeline (costo ~0)."""
    import diffusers
    meta = {
        "step": "1A",
        "model": MODEL_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        **dati,
    }
    with open(os.path.splitext(png_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# --- API pubblica -----------------------------------------------------------
def genera_sfondo(
    prompt: str,
    negative_prompt: str = NEGATIVE_DEFAULT,
    seed: int = SEED,
    steps: int = STEPS,
    guidance_scale: float = GUIDANCE,
    size: tuple[int, int] = SIZE_TARGET,
    output_path: str | None = None,
    forza: bool = False,
) -> str:
    """
    Genera il background plate e lo salva in config.PATHS["1A_SFONDO"].
    Ritorna il path del PNG (concatenabile con lo Step 1B).

    forza=False → se il file esiste già, lo step viene saltato (resume economico).
    In caso di OOM riprova automaticamente con CPU-offload + VAE tiling.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt vuoto: lo Step 1A richiede una descrizione della scena.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Nessuna GPU visibile. Su Kaggle: Settings → Accelerator → GPU T4 x2 "
            "(il modulo usa una sola GPU: 16 GB, non 32)."
        )

    out_path = output_path or config.PATHS["1A_SFONDO"]
    if os.path.exists(out_path) and not forza:
        log.info("Checkpoint già presente, salto la generazione: %s", out_path)
        return out_path

    if size[0] % 32 or size[1] % 32:
        log.warning("Risoluzione %s non multipla di 32: molti modelli I2V la richiedono.", size)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    bucket = _bucket_piu_vicino(size)
    log.info("GPU: %s | bucket %s → target %s | seed %d", torch.cuda.get_device_name(0), bucket, size, seed)

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    args = (prompt, negative_prompt, seed, steps, guidance_scale, bucket, size)
    try:
        img = _render(*args, offload=False)
    except torch.cuda.OutOfMemoryError:
        log.warning("OOM a piena VRAM: retry con model CPU-offload + VAE tiling.")
        config.pulisci_memoria()
        img = _render(*args, offload=True)

    img.save(out_path, "PNG")
    _salva_metadata(out_path, {
        "prompt": prompt, "negative_prompt": negative_prompt, "seed": seed,
        "steps": steps, "guidance_scale": guidance_scale,
        "bucket": bucket, "size": list(size), "path": out_path,
    })

    picco = torch.cuda.max_memory_allocated() / 1024**3
    log.info("Step 1A completato in %.1fs | VRAM picco %.2f GB | %s", time.time() - t0, picco, out_path)
    return out_path


# --- Smoke test -------------------------------------------------------------
if __name__ == "__main__":
    percorso = genera_sfondo(
        prompt=(
            "empty minimalist loft living room at golden hour, large window with city skyline, "
            "warm cinematic lighting, volumetric haze, 35mm photograph, shallow depth of field, "
            "no people, highly detailed, subtle film grain"
        ),
        seed=SEED,
        forza=True,
    )
    print("Sfondo pronto:", percorso)
