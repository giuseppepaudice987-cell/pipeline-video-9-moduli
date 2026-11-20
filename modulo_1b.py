"""
modulo_1b.py — Step 1B: KEYFRAME (soggetto inserito nello sfondo di 1A)

Input : PATHS["1A_SFONDO"]   -> output_pipeline/step_1a_sfondo.png
Output: PATHS["1B_KEYFRAME"] -> output_pipeline/step_1b_keyframe.png

Strategia: SDXL Inpainting fp16. Fuori dalla maschera lo sfondo resta
pixel-perfect; il soggetto viene generato solo nell'area mascherata.
Ottimizzato per Kaggle T4 16GB: fp16 + VAE-fix + slicing/tiling + fallback OOM.
"""

import os
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL

from config import PATHS, pulisci_memoria

# ------------------------- PARAMETRI EDITABILI -------------------------
MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"  # evita NaN / immagini nere del VAE SDXL in fp16

PROMPT_SOGGETTO = (
    "a woman standing in the scene, facing the camera, medium shot, "
    "photorealistic, detailed natural face, soft cinematic lighting"
)
NEGATIVE_PROMPT = (
    "bad anatomy, deformed hands, extra fingers, blurry, low quality, "
    "watermark, text, logo, cartoon, anime"
)

SEED = 42
STEPS = 35
GUIDANCE = 7.5
STRENGTH = 0.99                      # ~1.0 = rigenera completamente l'area mascherata
LATO_MAX = 1024                      # lato lungo max; dimensioni arrotondate a 64
MASK_BOX = (0.30, 0.18, 0.70, 1.00)  # (x0, y0, x1, y1) in frazioni: area del soggetto
MASK_FEATHER = 24                    # px di sfumatura sul bordo maschera
USA_CPU_OFFLOAD = False              # True = ~6GB VRAM ma più lento
# -----------------------------------------------------------------------


def _arrotonda64(x: float) -> int:
    return max(64, int(round(x / 64)) * 64)


def carica_sfondo() -> Image.Image:
    path = PATHS["1A_SFONDO"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"[1B] Sfondo non trovato: {path}. Esegui prima modulo_1a.")
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scala = min(1.0, LATO_MAX / max(w, h))
    size = (_arrotonda64(w * scala), _arrotonda64(h * scala))
    if size != (w, h):
        img = img.resize(size, Image.LANCZOS)
    print(f"[1B] Sfondo caricato: {img.size[0]}x{img.size[1]}")
    return img


def crea_maschera(size, box=MASK_BOX, feather=MASK_FEATHER) -> Image.Image:
    """Bianco = area da rigenerare (soggetto), nero = sfondo protetto."""
    w, h = size
    x0, y0, x1, y1 = box
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse(
        [int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)], fill=255
    )
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def _carica_pipe(offload: bool) -> StableDiffusionXLInpaintPipeline:
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_ID, vae=vae, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True,
    )
    pipe.set_progress_bar_config(desc="[1B] Inpainting")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass  # senza xformers, torch 2.x usa già SDPA memory-efficient
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def _inferenza(pipe, sfondo, maschera) -> Image.Image:
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    with torch.inference_mode():
        return pipe(
            prompt=PROMPT_SOGGETTO,
            negative_prompt=NEGATIVE_PROMPT,
            image=sfondo,
            mask_image=maschera,
            strength=STRENGTH,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=gen,
            add_watermarker=False,
        ).images[0]


def genera_keyframe() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("[1B] Nessuna GPU CUDA: su Kaggle imposta Accelerator = GPU T4.")

    pulisci_memoria()  # eventuali residui VRAM dello step 1A
    sfondo = carica_sfondo()
    maschera = crea_maschera(sfondo.size)

    print(f"[1B] Carico {MODEL_ID} in fp16...")
    pipe = _carica_pipe(offload=USA_CPU_OFFLOAD)

    try:
        keyframe = _inferenza(pipe, sfondo, maschera)
    except torch.cuda.OutOfMemoryError:
        print("[1B] OOM: riprovo con model CPU offload (più lento, ~6GB VRAM).")
        del pipe
        pulisci_memoria()
        pipe = _carica_pipe(offload=True)
        keyframe = _inferenza(pipe, sfondo, maschera)

    out = PATHS["1B_KEYFRAME"]
    keyframe.save(out)
    print(f"[1B] Keyframe salvato -> {out}")

    # NB: pulisci_memoria(model) fa `del` solo sul riferimento locale:
    # il `del pipe` va fatto qui nel chiamante per liberare davvero la VRAM.
    del pipe
    pulisci_memoria()
    return out


if __name__ == "__main__":
    genera_keyframe()
