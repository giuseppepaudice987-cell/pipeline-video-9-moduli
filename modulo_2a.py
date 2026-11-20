"""
modulo_2a.py — STEP 2A: Image-to-Video (keyframe 1C -> video raw 2A)
Target: Kaggle GPU T4 16GB VRAM, fp16 + model CPU offload.

Setup (una tantum, in una cella del notebook):
    !pip install -q -U diffusers transformers accelerate imageio-ffmpeg

Prerequisiti Kaggle: Internet ON, Accelerator GPU T4,
Secret "HF_TOKEN" abilitato (repo SVD gated: licenza da accettare su HF).
"""

import os

import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

from config import PATHS, pulisci_memoria

# ---------------- Parametri (tarati per T4) ----------------
MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
WIDTH, HEIGHT = 1024, 576     # risoluzione nativa SVD (W x H), entrambi /64
NUM_FRAMES = 25               # SVD-XT genera 25 frame
FPS_EXPORT = 8                # ~3.1 s di clip, estesa nello step 2B
NUM_STEPS = 25
MOTION_BUCKET_ID = 80         # basso = movimento contenuto (adatto al parlato)
NOISE_AUG_STRENGTH = 0.02     # basso = massima fedeltà al keyframe
DECODE_CHUNK_SIZE = 2         # decode VAE a blocchi: principale difesa anti-OOM
SEED = 42


def _hf_token():
    """Token HF: prima variabile d'ambiente, poi Kaggle Secrets."""
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:
            from kaggle_secrets import UserSecretsClient
            tok = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            tok = None
    return tok


def _prepara_keyframe(path: str) -> Image.Image:
    """Carica il keyframe e lo porta a WxH con cover-crop centrale (no stretch)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Keyframe mancante: {path}. Esegui prima lo step 1C.")
    img = Image.open(path).convert("RGB")
    scala = max(WIDTH / img.width, HEIGHT / img.height)
    img = img.resize((round(img.width * scala), round(img.height * scala)), Image.LANCZOS)
    left, top = (img.width - WIDTH) // 2, (img.height - HEIGHT) // 2
    return img.crop((left, top, left + WIDTH, top + HEIGHT))


def genera_video() -> str:
    """Keyframe 1C -> clip raw 2A. Ritorna il path del video generato."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU non visibile: Kaggle -> Settings -> Accelerator -> GPU T4.")

    immagine = _prepara_keyframe(PATHS["1C_FIXED"])
    torch.cuda.reset_peak_memory_stats()
    pipe = None

    try:
        try:
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16, variant="fp16", token=_hf_token()
            )
        except OSError as e:
            if any(k in str(e).lower() for k in ("gated", "401", "403", "access")):
                raise RuntimeError(
                    "Accesso negato a SVD-XT (repo gated). "
                    "1) Accetta la licenza su huggingface.co/"
                    "stabilityai/stable-video-diffusion-img2vid-xt "
                    "2) Kaggle -> Add-ons -> Secrets: etichetta 'HF_TOKEN' (token read), "
                    "abilitato nel notebook."
                ) from e
            raise

        # Offload: UNet/CLIP/VAE staffettati CPU<->GPU. NON chiamare pipe.to("cuda").
        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        pipe.unet.enable_forward_chunking()   # temporal attention a blocchi
        pipe.set_progress_bar_config(desc="2A | I2V")

        generatore = torch.Generator(device="cpu").manual_seed(SEED)
        with torch.inference_mode():
            frames = pipe(
                immagine,
                width=WIDTH, height=HEIGHT,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_STEPS,
                motion_bucket_id=MOTION_BUCKET_ID,
                noise_aug_strength=NOISE_AUG_STRENGTH,
                decode_chunk_size=DECODE_CHUNK_SIZE,
                generator=generatore,
            ).frames[0]

        export_to_video(frames, PATHS["2A_VIDEO"], fps=FPS_EXPORT)

        if not (os.path.isfile(PATHS["2A_VIDEO"]) and os.path.getsize(PATHS["2A_VIDEO"]) > 0):
            raise RuntimeError("Export fallito: file video assente o vuoto.")

        print(f"Step 2A completato: {len(frames)} frame, "
              f"{len(frames) / FPS_EXPORT:.2f}s -> {PATHS['2A_VIDEO']}")
        print(f"Picco VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        return PATHS["2A_VIDEO"]

    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            "OOM sul T4: abbassa DECODE_CHUNK_SIZE a 1 (prima manopola), "
            "poi eventualmente NUM_STEPS."
        ) from e
    finally:
        # del locale: pulisci_memoria(model) da sola non libera il riferimento chiamante
        if pipe is not None:
            del pipe
        pulisci_memoria()


if __name__ == "__main__":
    genera_video()
