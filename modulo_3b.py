# -*- coding: utf-8 -*-
"""
modulo_3b.py — Step 3B · Lip-sync (Wav2Lip) — Kaggle T4 16GB.

Input : PATHS["2B_EXTENDED"]  (video esteso, con volto visibile)
        PATHS["3A_AUDIO"]     (voce .wav)
Output: PATHS["3B_LIPSYNC"]   (mp4 h264 + aac)

Dipendenze: torch, numpy, opencv, scipy, librosa, ffmpeg  (tutti già in Kaggle).
Pesi: wav2lip_gan.pth + s3fd.pth  (Kaggle Dataset offline OR mirror automatici).
"""
from __future__ import annotations

import os
import sys
import time
import types
import shutil
import argparse
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile

from config import PATHS, pulisci_memoria

# --------------------------------------------------------------------------- #
# Layout / costanti
# --------------------------------------------------------------------------- #
WORK = Path(os.environ.get("PIPELINE_WORK", "/kaggle/working" if Path("/kaggle").exists() else "."))
REPO = WORK / "Wav2Lip"
WEIGHTS = WORK / "weights_3b"
KAGGLE_INPUT = Path("/kaggle/input")
REPO_URL = "https://github.com/Rudrabha/Wav2Lip.git"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hparams audio ufficiali Wav2Lip (NON toccare: il checkpoint è addestrato su questi)
SR, N_FFT, HOP, WIN = 16000, 800, 200, 800
N_MELS, FMIN, FMAX = 80, 55, 7600
PREEMPH, REF_DB, MIN_DB, MAX_ABS = 0.97, 20.0, -100.0, 4.0
MEL_STEP, IMG_SIZE = 16, 96

# Mirror best-effort: NON garantiti nel tempo. Il fallback serio è il Kaggle Dataset.
MIRRORS = {
    "wav2lip_gan.pth": [
        "https://huggingface.co/camenduru/Wav2Lip/resolve/main/wav2lip_gan.pth",
        "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth",
    ],
    "wav2lip.pth": [
        "https://huggingface.co/camenduru/Wav2Lip/resolve/main/wav2lip.pth",
        "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip.pth",
    ],
    "s3fd.pth": [
        "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
        "https://huggingface.co/camenduru/Wav2Lip/resolve/main/s3fd.pth",
    ],
}
ALIASES = {"s3fd.pth": ["s3fd-619a316812.pth"]}


@dataclass
class Opzioni3B:
    gan: bool = True                 # True = più nitido, False = sync leggermente migliore
    batch_size: int = 16             # frame per forward Wav2Lip (96x96: leggerissimo)
    detect_batch: int = 8            # frame per forward s3fd (questo pesa in VRAM)
    detect_max_side: int = 640       # downscale solo per la face detection
    pads: Tuple[int, int, int, int] = (0, 10, 0, 0)   # top, bottom, left, right (mento)
    smooth: int = 5                  # media mobile sul bbox (dispari) → anti-jitter
    fp16: bool = True
    loop_mode: str = "pingpong"      # "pingpong" | "loop" | "trim"
    resize_factor: int = 1           # 2 = dimezza la risoluzione del video sorgente
    crf: int = 18
    preset: str = "medium"
    max_seconds: Optional[float] = None
    static_box: Optional[Tuple[int, int, int, int]] = None  # fallback manuale (x1,y1,x2,y2)


def _log(msg: str) -> None:
    print(f"[3B] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Runtime patches (Kaggle 2025/2026: torch>=2.6, numpy>=2.x)
# --------------------------------------------------------------------------- #
def _patch_runtime() -> None:
    for alias, target in (("float", float), ("int", int), ("bool", bool), ("object", object)):
        if not hasattr(np, alias):
            setattr(np, alias, target)

    if not getattr(torch, "_patched_3b", False):
        _orig_load = torch.load

        def _load(*a, **kw):
            # Sicuro: carichiamo solo checkpoint nostri/fidati, mai pickle arbitrari.
            kw.setdefault("weights_only", False)
            return _orig_load(*a, **kw)

        torch.load = _load
        torch._patched_3b = True

    try:
        from torch.utils import model_zoo  # noqa: F401
    except Exception:
        mz = types.ModuleType("torch.utils.model_zoo")
        mz.load_url = torch.hub.load_state_dict_from_url
        sys.modules["torch.utils.model_zoo"] = mz
        torch.utils.model_zoo = mz

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()


def _ensure_repo() -> Path:
    if (REPO / "models" / "wav2lip.py").exists():
        return REPO
    _log("clone Wav2Lip (shallow)…")
    r = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Clone fallito (Internet attivo in Kaggle?).\n{r.stderr[-500:]}")
    return REPO


def _import_repo():
    """Import isolato: il repo NON resta permanentemente in sys.path."""
    repo = str(_ensure_repo())
    for name in ("models", "face_detection"):
        mod = sys.modules.get(name)
        if mod is not None and repo not in str(getattr(mod, "__file__", "") or ""):
            raise RuntimeError(f"Namespace '{name}' già occupato da un altro step: "
                               f"esegui 3B in una sessione/processo separato.")
    sys.path.insert(0, repo)
    try:
        from models.wav2lip import Wav2Lip
        from face_detection.detection.sfd.sfd_detector import SFDDetector
    finally:
        if repo in sys.path:
            sys.path.remove(repo)
    return Wav2Lip, SFDDetector


# --------------------------------------------------------------------------- #
# Pesi
# --------------------------------------------------------------------------- #
def _download(url: str, dst: Path, min_bytes: int) -> bool:
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        if tmp.stat().st_size < min_bytes:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dst)
        _log(f"scaricato {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        _log(f"mirror fallito: {url} ({type(e).__name__})")
        tmp.unlink(missing_ok=True)
        return False


def _resolve_weight(filename: str, env_var: str, min_mb: float = 40.0) -> Path:
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    min_bytes = int(min_mb * 1e6)

    env = os.environ.get(env_var)
    if env and Path(env).is_file():
        return Path(env)

    local = WEIGHTS / filename
    if local.is_file() and local.stat().st_size >= min_bytes:
        return local

    names = [filename] + ALIASES.get(filename, [])
    if KAGGLE_INPUT.exists():
        for name in names:
            for hit in sorted(KAGGLE_INPUT.rglob(name)):
                if hit.stat().st_size >= min_bytes:
                    _log(f"peso da Dataset: {hit}")
                    return hit

    for url in MIRRORS.get(filename, []):
        if _download(url, local, min_bytes):
            return local

    raise RuntimeError(
        f"Peso '{filename}' non trovato. Soluzione consigliata: carica un Kaggle Dataset "
        f"contente {', '.join(names)} e aggiungi il path con os.environ['{env_var}'], "
        f"oppure attiva Internet nelle impostazioni del notebook."
    )


# --------------------------------------------------------------------------- #
# Audio → mel (fedele agli hparams Wav2Lip, senza dipendere da repo/audio.py)
# --------------------------------------------------------------------------- #
def _audio_to_mel(audio_path: Path, tmp_dir: Path) -> np.ndarray:
    import librosa

    wav16k = tmp_dir / "audio_16k.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
                    "-vn", "-ac", "1", "-ar", str(SR), str(wav16k)], check=True)
    sr, data = wavfile.read(str(wav16k))
    if data.dtype == np.int16:
        wav = data.astype(np.float32) / 32768.0
    else:
        wav = data.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    y = signal.lfilter([1, -PREEMPH], [1], wav)
    D = librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                     center=True, pad_mode="reflect")          # reflect = librosa 0.7
    basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    S = 20.0 * np.log10(np.maximum(1e-5, np.dot(basis, np.abs(D)))) - REF_DB
    mel = np.clip((2 * MAX_ABS) * ((S - MIN_DB) / (-MIN_DB)) - MAX_ABS, -MAX_ABS, MAX_ABS)

    if np.isnan(mel).any():
        raise RuntimeError("Mel con NaN: audio corrotto o silenzioso.")
    if mel.shape[1] < MEL_STEP:
        mel = np.pad(mel, ((0, 0), (0, MEL_STEP - mel.shape[1])), mode="edge")
    return mel.astype(np.float32)


def _mel_chunks(mel: np.ndarray, fps: float, max_frames: Optional[int]) -> list:
    mult, T, out, i = 80.0 / fps, mel.shape[1], [], 0
    while True:
        s = int(i * mult)
        if s + MEL_STEP > T:
            out.append(mel[:, T - MEL_STEP:])
            break
        out.append(mel[:, s:s + MEL_STEP])
        i += 1
        if max_frames and len(out) >= max_frames:
            break
    return out


# --------------------------------------------------------------------------- #
# Video / face detection
# --------------------------------------------------------------------------- #
def _read_frames(video: Path, resize_factor: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Video illeggibile: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps == fps and 1 < fps < 121 else 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if resize_factor > 1:
            h, w = f.shape[:2]
            f = cv2.resize(f, (max(2, w // resize_factor), max(2, h // resize_factor)),
                           interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError("Video senza frame decodificabili.")
    nbytes = sum(f.nbytes for f in frames)
    _log(f"video: {len(frames)} frame @ {fps:.3f} fps, {frames[0].shape[1]}x{frames[0].shape[0]}, "
         f"{nbytes / 1e9:.2f} GB RAM")
    if nbytes > 4e9:
        _log("ATTENZIONE: buffer video >4GB, valuta resize_factor=2.")
    return frames, fps


def _detect_boxes(det, frames, cfg: Opzioni3B) -> np.ndarray:
    n = len(frames)
    H, W = frames[0].shape[:2]
    s = min(1.0, cfg.detect_max_side / max(H, W))
    dw, dh = max(64, int(W * s) // 2 * 2), max(64, int(H * s) // 2 * 2)
    sx, sy = dw / W, dh / H
    boxes = np.full((n, 4), np.nan, np.float32)

    for start in range(0, n, cfg.detect_batch):
        idx = list(range(start, min(start + cfg.detect_batch, n)))
        batch = np.stack([cv2.resize(frames[i], (dw, dh), interpolation=cv2.INTER_AREA)[..., ::-1]
                          for i in idx])
        batch = np.ascontiguousarray(batch)  # RGB uint8, atteso da s3fd
        with torch.no_grad():
            preds = det.detect_from_batch(batch)
        for j, d in enumerate(preds):
            if d is None or len(d) == 0:
                continue
            best = max(d, key=lambda b: float(b[-1]))
            x1, y1, x2, y2 = (float(v) for v in best[:4])
            boxes[idx[j]] = [x1 / sx, y1 / sy, x2 / sx, y2 / sy]

    valid = np.isfinite(boxes[:, 0])
    hit = int(valid.sum())
    _log(f"volti rilevati: {hit}/{n} frame ({100 * hit / n:.1f}%)")

    if hit == 0:
        if cfg.static_box is None:
            raise RuntimeError("Nessun volto rilevato. Il keyframe 1B contiene un viso frontale? "
                               "Prova detect_max_side=960 oppure imposta Opzioni3B.static_box.")
        _log("fallback: static_box manuale")
        boxes[:] = np.asarray(cfg.static_box, np.float32)
    elif hit < n:
        for c in range(4):
            boxes[:, c] = np.interp(np.arange(n), np.flatnonzero(valid), boxes[valid, c])

    k = max(1, cfg.smooth | 1)
    if k > 1 and n > k:
        pad = np.pad(boxes, ((k // 2, k // 2), (0, 0)), mode="edge")
        boxes = np.stack([np.convolve(pad[:, c], np.ones(k) / k, mode="valid")
                          for c in range(4)], axis=1)

    pt, pb, pl, pr = cfg.pads
    boxes[:, 0] = np.clip(boxes[:, 0] - pl, 0, W - 2)
    boxes[:, 1] = np.clip(boxes[:, 1] - pt, 0, H - 2)
    boxes[:, 2] = np.clip(boxes[:, 2] + pr, 2, W)
    boxes[:, 3] = np.clip(boxes[:, 3] + pb, 2, H)
    return boxes.astype(np.int32)


def _frame_sequence(n_src: int, n_out: int, mode: str) -> np.ndarray:
    if mode == "trim":
        return np.arange(min(n_src, n_out))
    if mode == "loop":
        return np.arange(n_out) % n_src
    period = max(1, 2 * n_src - 2)              # ping-pong: 0,1,…,n-1,n-2,…,1
    t = np.arange(n_out) % period
    return np.where(t < n_src, t, period - t)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def esegui(video_path: Optional[str] = None,
           audio_path: Optional[str] = None,
           out_path: Optional[str] = None,
           cfg: Optional[Opzioni3B] = None) -> str:

    cfg = cfg or Opzioni3B()
    video = Path(video_path or PATHS["2B_EXTENDED"])
    audio = Path(audio_path or PATHS["3A_AUDIO"])
    out = Path(out_path or PATHS["3B_LIPSYNC"])
    for p, step in ((video, "2B"), (audio, "3A")):
        if not p.is_file():
            raise FileNotFoundError(f"Input mancante: {p} (esegui prima lo step {step}).")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg non disponibile nel PATH.")
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pulisci_memoria()
    _patch_runtime()

    ckpt_name = "wav2lip_gan.pth" if cfg.gan else "wav2lip.pth"
    ckpt = _resolve_weight(ckpt_name, "WAV2LIP_CKPT")
    s3fd_path = _resolve_weight("s3fd.pth", "S3FD_CKPT", min_mb=40.0)
    Wav2LipCls, SFDDetector = _import_repo()

    model = det = proc = None
    tmp_dir = WORK / "tmp_3b"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- modelli ---
        model = Wav2LipCls()
        sd = torch.load(str(ckpt), map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        model = model.eval().to(DEVICE).requires_grad_(False)
        del sd
        det = SFDDetector(device=DEVICE, path_to_detector=str(s3fd_path))
        _log(f"modelli su {DEVICE} (fp16={cfg.fp16 and DEVICE == 'cuda'})")

        # --- sorgenti ---
        frames, fps = _read_frames(video, cfg.resize_factor)
        mel = _audio_to_mel(audio, tmp_dir)
        max_frames = int(cfg.max_seconds * fps) if cfg.max_seconds else None
        chunks = _mel_chunks(mel, fps, max_frames)
        seq = _frame_sequence(len(frames), len(chunks), cfg.loop_mode)
        n_out = len(seq)
        if cfg.loop_mode != "trim" and n_out > 4 * len(frames):
            _log(f"ATTENZIONE: audio {n_out / fps:.1f}s vs video {len(frames) / fps:.1f}s "
                 f"→ loop x{n_out / len(frames):.1f}, possibile ripetizione visibile.")

        boxes = _detect_boxes(det, frames, cfg)
        del det
        det = None
        pulisci_memoria()

        # --- writer: pipe raw → ffmpeg (h264 + aac, single pass) ---
        H, W = frames[0].shape[:2]
        H2, W2 = H - H % 2, W - W % 2
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W2}x{H2}", "-r", f"{fps:.6f}",
               "-i", "pipe:0", "-i", str(audio),
               "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf),
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
               "-r", f"{fps:.6f}", "-shortest", "-movflags", "+faststart", str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # --- inference batched ---
        amp = cfg.fp16 and DEVICE == "cuda"
        for start in range(0, n_out, cfg.batch_size):
            idx = seq[start:start + cfg.batch_size]
            imgs, mels = [], []
            for k, fi in enumerate(idx):
                x1, y1, x2, y2 = boxes[fi]
                face = cv2.resize(frames[fi][y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
                masked = face.copy()
                masked[IMG_SIZE // 2:] = 0                       # maschera metà inferiore
                imgs.append(np.concatenate([masked, face], axis=0) / 255.0)
                mels.append(chunks[start + k])

            img_t = torch.from_numpy(np.asarray(imgs, np.float32)).to(DEVICE)
            mel_t = torch.from_numpy(np.asarray(mels, np.float32)).unsqueeze(1).to(DEVICE)
            with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=amp):
                pred = model(mel_t, img_t)
            pred = (pred.float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)

            for k, fi in enumerate(idx):
                x1, y1, x2, y2 = boxes[fi]
                frame = frames[fi].copy()
                frame[y1:y2, x1:x2] = cv2.resize(pred[k], (x2 - x1, y2 - y1))
                proc.stdin.write(np.ascontiguousarray(frame[:H2, :W2]).tobytes())

            if (start // cfg.batch_size) % 20 == 0:
                done = min(start + cfg.batch_size, n_out)
                el = time.time() - t0
                _log(f"{done}/{n_out} frame · {done / max(el, 1e-6):.1f} fps · {el:.0f}s")

        proc.stdin.close()
        err = proc.stderr.read().decode(errors="ignore")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg fallito:\n{err[-800:]}")
        if not out.is_file() or out.stat().st_size < 1024:
            raise RuntimeError("Output vuoto o corrotto.")

        peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
        _log(f"OK → {out} · {n_out} frame / {n_out / fps:.2f}s · "
             f"{time.time() - t0:.1f}s · VRAM picco {peak:.2f} GB")
        return str(out)

    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.kill()
        for f in tmp_dir.glob("*.wav"):
            f.unlink(missing_ok=True)
        del model, det
        pulisci_memoria()


run = esegui  # alias per l'orchestratore


def _main():
    ap = argparse.ArgumentParser(description="Step 3B — lip-sync Wav2Lip")
    ap.add_argument("--video"); ap.add_argument("--audio"); ap.add_argument("--out")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--detect-batch", type=int, default=8)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-gan", action="store_true")
    ap.add_argument("--loop-mode", choices=["pingpong", "loop", "trim"], default="pingpong")
    ap.add_argument("--resize-factor", type=int, default=1)
    a = ap.parse_args()
    cfg = Opzioni3B(gan=not a.no_gan, batch_size=a.batch_size, detect_batch=a.detect_batch,
                    fp16=not a.no_fp16, loop_mode=a.loop_mode, resize_factor=a.resize_factor)
    esegui(a.video, a.audio, a.out, cfg)


if __name__ == "__main__":
    _main()
