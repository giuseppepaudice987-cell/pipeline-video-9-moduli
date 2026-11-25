"""
modulo_4a.py

Step 4A della pipeline:
- Input:  PATHS["3B_LIPSYNC"]
- Output: PATHS["4A_SMOOTH"]

Obiettivo:
- ridurre micro-rumore/flickering temporale
- mantenere l'audio originale se possibile
- non appesantire la GPU T4
- funzionare anche se NVENC o alcuni filtri ffmpeg non sono disponibili

Note:
- usa ffmpeg esterno
- se disponibile, prova encoding NVENC
- fallback automatici a libx264
- scritto per essere robusto in ambiente Kaggle
"""

import os
import shutil
import subprocess
import sys

from config import PATHS, OUTPUT_DIR, pulisci_memoria

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


DEBUG = os.getenv("DEBUG_MODULO_4A", "0").strip().lower() in {"1", "true", "yes"}
FORCE = (
    os.getenv("FORCE_STEP_4A", "0").strip().lower() in {"1", "true", "yes"}
    or os.getenv("FORCE_PIPELINE", "0").strip().lower() in {"1", "true", "yes"}
)

# Filtro base: garantisce yuv420p e dimensioni pari, utili per H.264.
BASE_VF = "format=yuv420p,scale=trunc(iw/2)*2:trunc(ih/2)*2"

# Smoothing temporale/spaziale leggero.
# Valori volutamente bassi per non impastare il labiale.
SMOOTH_VF = BASE_VF + ",hqdn3d=1.0:1.0:2.0:2.0"


def _tail(text: str, max_chars: int = 2500) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def _run(cmd: list, desc: str = "") -> tuple:
    """
    Esegue un comando ffmpeg.
    Ritorna (ok, stderr_tail).
    """
    if DEBUG:
        print("[4A] CMD:", " ".join(cmd))
    elif desc:
        print(f"[4A] {desc}")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="ignore",
            check=False,
        )
    except Exception as exc:
        return False, str(exc)

    if proc.returncode != 0:
        return False, _tail(proc.stderr)

    return True, _tail(proc.stderr)


def trova_ffmpeg() -> str:
    """
    Cerca ffmpeg prima nel PATH, poi eventualmente nel pacchetto imageio-ffmpeg.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg and os.path.exists(system_ffmpeg) and os.access(system_ffmpeg, os.X_OK):
        return system_ffmpeg

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe) and os.access(exe, os.X_OK):
            return exe
    except Exception:
        pass

    raise RuntimeError(
        "ffmpeg non trovato. Installa ffmpeg di sistema oppure imageio-ffmpeg."
    )


def _ffmpeg_list(ffmpeg: str, flag: str) -> str:
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", flag],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            errors="ignore",
            timeout=30,
            check=False,
        )
        return proc.stdout or ""
    except Exception:
        return ""


def encoder_disponibile(ffmpeg: str, name: str) -> bool:
    out = _ffmpeg_list(ffmpeg, "-encoders")
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == name:
            return True
    return False


def filtro_disponibile(ffmpeg: str, name: str) -> bool:
    out = _ffmpeg_list(ffmpeg, "-filters")
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == name:
            return True
    return False


def _video_encoder_options(ffmpeg: str) -> list:
    """
    Ritorna una lista di opzioni video, in ordine di preferenza.
    """
    opts = []

    # NVENC se disponibile: veloce e non carica modelli PyTorch.
    if CUDA_AVAILABLE and encoder_disponibile(ffmpeg, "h264_nvenc"):
        opts.append(
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "medium",
                "-rc",
                "vbr",
                "-cq",
                "19",
                "-b:v",
                "0",
            ]
        )

    # CPU fallback principale.
    if encoder_disponibile(ffmpeg, "libx264"):
        opts.append(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])
        opts.append(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20"])

    # Fallback estremo.
    if not opts:
        opts.append(["-c:v", "mpeg4", "-q:v", "5"])

    return opts


def _describe_attempt(vf_args: list, video_args: list, audio_mode: str) -> str:
    enc = video_args[1] if len(video_args) >= 2 else "sconosciuto"

    if any("hqdn3d" in str(x) for x in vf_args):
        filt = "smooth(hqdn3d)"
    elif vf_args:
        filt = "base(format/scale)"
    else:
        filt = "none"

    return f"video={enc}, filtro={filt}, audio={audio_mode}"


def _build_cmd(
    ffmpeg: str,
    inp: str,
    out: str,
    vf_args: list,
    video_args: list,
    audio_mode: str,
) -> list:
    cmd = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        inp,
        "-map",
        "0:v:0",
    ]

    if audio_mode == "aac":
        cmd += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k"]
    elif audio_mode == "copy":
        cmd += ["-map", "0:a:0?", "-c:a", "copy"]
    else:
        cmd += ["-an"]

    if vf_args:
        cmd += vf_args

    cmd += video_args
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", out]

    return cmd


def main(force: bool = False) -> str:
    inp = PATHS["3B_LIPSYNC"]
    out = PATHS["4A_SMOOTH"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(inp):
        raise FileNotFoundError(f"Input 3B non trovato: {inp}")

    if os.path.getsize(inp) == 0:
        raise ValueError(f"Input 3B vuoto: {inp}")

    tmp = out + ".tmp.mp4"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except Exception:
            pass

    if (
        not force
        and os.path.isfile(out)
        and os.path.getsize(out) > 0
        and os.path.getmtime(out) >= os.path.getmtime(inp)
    ):
        print("[4A] Output già presente e aggiornato. Nessun lavoro da fare.")
        return out

    ffmpeg = trova_ffmpeg()

    use_smooth = filtro_disponibile(ffmpeg, "hqdn3d")
    if not use_smooth:
        print(
            "[4A] Filtro hqdn3d non disponibile: "
            "uso solo normalizzazione formato/dimensioni."
        )

    video_options = _video_encoder_options(ffmpeg)

    smooth_vf_args = ["-vf", SMOOTH_VF]
    base_vf_args = ["-vf", BASE_VF]
    no_vf_args = []

    attempts = []

    # 1) Tentativi preferiti con smoothing vero e proprio.
    if use_smooth:
        for vid in video_options[:2]:
            attempts.append((smooth_vf_args, vid, "aac"))
            attempts.append((smooth_vf_args, vid, "copy"))

    # 2) Fallback con filtro base e senza filtro.
    for vid in video_options[:2]:
        attempts.append((base_vf_args, vid, "aac"))
        attempts.append((base_vf_args, vid, "copy"))
        attempts.append((no_vf_args, vid, "aac"))
        attempts.append((no_vf_args, vid, "copy"))

    # 3) Fallback finale più aggressivo.
    if len(video_options) > 2:
        attempts.append((no_vf_args, video_options[2], "aac"))
        attempts.append((no_vf_args, video_options[2], "copy"))
        attempts.append((no_vf_args, video_options[2], "none"))
    else:
        attempts.append((no_vf_args, video_options[-1], "none"))

    # Deduplicazione.
    seen = set()
    unique_attempts = []
    for vf_args, vid_args, audio_mode in attempts:
        key = (tuple(vf_args), tuple(vid_args), audio_mode)
        if key not in seen:
            seen.add(key)
            unique_attempts.append((vf_args, vid_args, audio_mode))

    last_error = ""

    try:
        for i, (vf_args, vid_args, audio_mode) in enumerate(unique_attempts, 1):
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

            cmd = _build_cmd(ffmpeg, inp, tmp, vf_args, vid_args, audio_mode)
            desc = _describe_attempt(vf_args, vid_args, audio_mode)
            print(f"[4A] Tentativo {i}/{len(unique_attempts)} -> {desc}")

            ok, err = _run(cmd, desc if DEBUG else "")

            if ok and os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
                os.replace(tmp, out)

                if audio_mode == "none":
                    print("[4A] WARNING: output scritto senza audio.")
                elif not any("hqdn3d" in str(x) for x in vf_args):
                    print("[4A] WARNING: smoothing temporale completo non applicato.")

                print(f"[4A] Scritto: {out}")
                return out

            last_error = err
            if err:
                print(f"[4A] Tentativo fallito: {_tail(err, 400)}")

        raise RuntimeError(
            f"Generazione 4A fallita dopo {len(unique_attempts)} tentativi. "
            f"Ultimo errore: {_tail(last_error, 1200)}"
        )

    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


if __name__ == "__main__":
    try:
        force = FORCE or ("--force" in sys.argv)
        main(force=force)
    except KeyboardInterrupt:
        print("[4A][ERRORE] Interrotto dall'utente.")
        sys.exit(130)
    except Exception as exc:
        print(f"[4A][ERRORE] {exc}")
        sys.exit(1)
    finally:
        pulisci_memoria()
