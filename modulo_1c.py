"""
modulo_1c.py — Step 1C: Fix & Detail del keyframe.

Input : PATHS["1B_KEYFRAME"]  (step_1b_keyframe.png)
Output: PATHS["1C_FIXED"]     (step_1c_fixed.png, stesse dimensioni di 1B)

Strategia (T4 16GB, fp16):
  1) Pass img2img globale a bassa strength: armonizza il composito 1A+1B
     (cuciture, luci, texture) preservando composizione e identità.
  2) Face-detailer: Haar cascade -> crop volto -> img2img mirato ->
     reincolla con maschera sfumata. Stesso pipeline, zero VRAM extra.
"""

import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline

from config import PATHS, pulisci_memoria

# ----------------------------- Parametri -----------------------------
MODEL_ID = "Lykon/dreamshaper-8"   # SD1.5 photoreal, safetensors
SEED = 42
MAX_LATO = 1024                    # cap lato lungo in lavorazione
STRENGTH_GLOBAL = 0.28             # <0.2 inefficace, >0.4 drift di composizione
STRENGTH_FACE = 0.35
GUIDANCE = 5.0                     # CFG basso: a bassa strength evita oversaturazione
STEPS_GLOBAL = 30
STEPS_FACE = 24
FACE_DETAIL = True
MAX_VOLTI = 2                      # bound sul tempo di calcolo
FACE_EXPAND = 1.8                  # il crop include capelli/mento/collo
FACE_RES = 512                     # risoluzione del crop per il pass volto

# NB: per massima coerenza, passare a esegui_1c() lo stesso prompt usato in 1B.
PROMPT_DEFAULT = (
    "RAW photo, photorealistic, detailed natural skin texture, detailed eyes, "
    "soft cinematic lighting, sharp focus, high quality"
)
FACE_SUFFIX = ", detailed symmetric face, detailed iris, natural skin pores"
NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, 3d render, cgi, "
    "deformed face, distorted face, asymmetric eyes, cross-eyed, bad anatomy, "
    "extra fingers, fused fingers, blurry, low quality, jpeg artifacts, "
    "watermark, text, logo, oversaturated, plastic skin, airbrushed"
)


# ----------------------------- Helper -----------------------------
def _snap8(v: float) -> int:
    """SD richiede dimensioni multiple di 8."""
    return max(8, int(round(v / 8)) * 8)


def _carica_keyframe(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[1C] Input mancante: {path}. Esegui prima modulo_1b."
        )
    img = Image.open(path).convert("RGB")
    orig_size = img.size
    w, h = img.size
    scala = min(1.0, MAX_LATO / max(w, h))
    if scala < 1.0:
        img = img.resize((int(w * scala), int(h * scala)), Image.LANCZOS)
    w, h = img.size
    img = img.resize((_snap8(w), _snap8(h)), Image.LANCZOS)
    return img, orig_size


def _carica_pipeline(device: str, dtype: torch.dtype):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,          # risparmia VRAM, evita falsi positivi
        requires_safety_checker=False,
        use_safetensors=True,
    )
    # DPM++ 2M Karras: qualità migliore a parità di step rispetto al default
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    pipe.set_progress_bar_config(leave=False)
    return pipe.to(device)


def _passaggio_globale(pipe, img, prompt, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=img,
        strength=STRENGTH_GLOBAL,
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS_GLOBAL,
        generator=gen,
    ).images[0]


def _rileva_volti(img: Image.Image):
    """Haar cascade: niente dipendenze extra, cv2 è già su Kaggle."""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    ms = max(48, int(min(img.size) * 0.05))
    boxes = ()
    for nome in ("haarcascade_frontalface_default.xml",
                 "haarcascade_frontalface_alt2.xml"):
        clf = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, nome))
        boxes = clf.detectMultiScale(gray, 1.1, 5, minSize=(ms, ms))
        if len(boxes):
            break
    # I volti più grandi prima (sono i più importanti per il lipsync downstream)
    return sorted(boxes, key=lambda b: int(b[2]) * int(b[3]),
                  reverse=True)[:MAX_VOLTI]


def _dettaglio_volto(pipe, img, box, prompt, seed, device):
    w, h = img.size
    x, y, bw, bh = (int(v) for v in box)
    cx, cy = x + bw / 2, y + bh / 2
    lato = min(int(max(bw, bh) * FACE_EXPAND), w, h)
    if lato < 96:                      # troppo piccolo per un pass sensato
        return img
    x0 = int(min(max(cx - lato / 2, 0), w - lato))
    y0 = int(min(max(cy - lato / 2, 0), h - lato))

    crop = img.crop((x0, y0, x0 + lato, y0 + lato))
    crop_sd = crop.resize((FACE_RES, FACE_RES), Image.LANCZOS)

    gen = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=prompt + FACE_SUFFIX,
        negative_prompt=NEGATIVE_PROMPT,
        image=crop_sd,
        strength=STRENGTH_FACE,
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS_FACE,
        generator=gen,
    ).images[0].resize((lato, lato), Image.LANCZOS)

    # Maschera sfumata: il bordo del crop non deve lasciare cuciture
    mask = Image.new("L", (lato, lato), 0)
    m = int(lato * 0.14)
    ImageDraw.Draw(mask).rectangle([m, m, lato - m, lato - m], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(4, lato // 16)))

    fuso = Image.composite(out, crop, mask)
    img = img.copy()
    img.paste(fuso, (x0, y0))
    return img


# ----------------------------- Entry point -----------------------------
def esegui_1c(prompt: str | None = None, seed: int = SEED,
              face_detail: bool = FACE_DETAIL) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cpu":
        print("[1C] ATTENZIONE: CUDA non disponibile, fallback CPU (molto lento).")
    prompt = prompt or PROMPT_DEFAULT

    img, orig_size = _carica_keyframe(PATHS["1B_KEYFRAME"])
    pipe = None
    try:
        pipe = _carica_pipeline(device, dtype)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        print(f"[1C] Pass globale | {img.size} | strength={STRENGTH_GLOBAL}")
        try:
            img = _passaggio_globale(pipe, img, prompt, seed, device)
        except torch.cuda.OutOfMemoryError:
            print("[1C] OOM: retry con CPU offload + attention slicing.")
            pulisci_memoria()
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            img = _passaggio_globale(pipe, img, prompt, seed, device)

        if face_detail:
            boxes = _rileva_volti(img)
            print(f"[1C] Volti rilevati: {len(boxes)}")
            for i, box in enumerate(boxes):
                img = _dettaglio_volto(pipe, img, box, prompt,
                                       seed + 100 + i, device)

        # Contratto con 2A: stesse dimensioni del keyframe 1B
        if img.size != orig_size:
            img = img.resize(orig_size, Image.LANCZOS)
        img.save(PATHS["1C_FIXED"])
        print(f"[1C] Salvato: {PATHS['1C_FIXED']}")
        if device == "cuda":
            print(f"[1C] Picco VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        return PATHS["1C_FIXED"]
    finally:
        # Il `del` in config.pulisci_memoria rimuove solo il nome locale:
        # va azzerato anche il reference del chiamante prima del gc.
        pipe = None
        pulisci_memoria()


if __name__ == "__main__":
    esegui_1c()
