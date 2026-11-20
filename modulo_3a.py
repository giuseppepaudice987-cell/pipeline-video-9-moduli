"""
modulo_3a.py — Step 3A: sintesi vocale (TTS) della pipeline video.

Modello : Coqui XTTS-v2 (fork mantenuto `coqui-tts`, import name `TTS`)
Output  : PATHS["3A_AUDIO"] -> WAV PCM 16-bit, mono, 24 kHz

Uso (notebook):
    from modulo_3a import esegui_3a
    esegui_3a("Testo del copione.")

Uso (shell):
    python modulo_3a.py "Testo del copione."
"""
from __future__ import annotations

import importlib.util
import math
import os
import re
import subprocess
import sys

import numpy as np

from config import PATHS, pulisci_memoria

__all__ = ["genera_voce", "esegui_3a"]

# ------------------------------ Parametri -----------------------------------
TESTO_DEFAULT = (
    "Benvenuto. Questa è la voce generata al passo tre A della pipeline. "
    "Sostituisci questo testo con il copione definitivo."
)
LINGUA = "it"
SPEAKER_DEFAULT = "Claribel Dervla"   # voce built-in XTTS (accento EN sull'it)
SPEAKER_WAV: str | None = None        # es. "ref_voce.wav": 6-10 s puliti -> cloning
SR_TARGET = 24_000                    # rate nativo XTTS-v2
MAX_CHAR_CHUNK = 200                  # < limite XTTS per l'italiano (213)
PAUSA_TRA_CHUNK_S = 0.25
TARGET_PICCO = 0.891                  # -1 dBFS
FADE_MS = 10
MODELLO_XTTS = "tts_models/multilingual/multi-dataset/xtts_v2"


# ------------------------------ Dipendenze ----------------------------------
def _assicura_dipendenze() -> None:
    """Installa solo ciò che manca (Kaggle-friendly, no side-effect all'import)."""
    mancanti = []
    for modulo, pacchetto in (("TTS", "coqui-tts"),
                              ("soundfile", "soundfile"),
                              ("scipy", "scipy")):
        if importlib.util.find_spec(modulo) is None:
            mancanti.append(pacchetto)
    if mancanti:
        print(f"[3A] Installo dipendenze mancanti: {mancanti}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *mancanti])


# ------------------------------ Chunking ------------------------------------
def _chunk_testo(testo: str, max_char: int = MAX_CHAR_CHUNK) -> list[str]:
    """Spezza il copione in chunk sotto il limite caratteri di XTTS,
    rispettando i confini di frase (poi virgole, poi parole)."""
    testo = re.sub(r"\s+", " ", testo).strip()
    if not testo:
        raise ValueError("[3A] Testo vuoto: niente da sintetizzare.")

    frasi = [f for f in re.split(r"(?<=[.!?…])\s+", testo) if f]
    pezzi: list[str] = []
    for frase in frasi:
        if len(frase) <= max_char:
            pezzi.append(frase)
            continue
        # Frase monstre: split su virgole/punto e virgola, poi hard-split parole.
        for sotto in re.split(r"(?<=[,;:])\s+", frase):
            sotto = sotto.strip()
            if not sotto:
                continue
            if len(sotto) <= max_char:
                pezzi.append(sotto)
            else:
                buf = ""
                for parola in sotto.split(" "):
                    if buf and len(buf) + 1 + len(parola) > max_char:
                        pezzi.append(buf)
                        buf = parola
                    else:
                        buf = f"{buf} {parola}".strip()
                if buf:
                    pezzi.append(buf)

    # Impacchetta pezzi adiacenti per non sintetizzare micro-frasi.
    chunks: list[str] = []
    buf = ""
    for p in pezzi:
        if buf and len(buf) + 1 + len(p) > max_char:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf} {p}".strip()
    if buf:
        chunks.append(buf)
    return chunks


# ------------------------------ DSP ------------------------------------------
def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    from scipy.signal import resample_poly
    g = math.gcd(sr_in, sr_out)
    return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)


def _postprocess(audio: np.ndarray, sr: int) -> np.ndarray:
    """Mono -> trim silenzi -> normalizzazione picco -> fade anti-click."""
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.ndim != 1:
        audio = audio.mean(axis=-1)

    # Trim testa/coda (soglia ~ -60 dBFS, margine 50 ms).
    mask = np.abs(audio) > 1e-3
    if mask.any():
        i0 = int(np.argmax(mask))
        i1 = len(mask) - int(np.argmax(mask[::-1]))
        pad = int(0.05 * sr)
        audio = audio[max(0, i0 - pad):min(len(audio), i1 + pad)]

    picco = float(np.max(np.abs(audio))) if audio.size else 0.0
    if picco < 1e-6:
        raise RuntimeError("[3A] Audio silenzioso: sintesi fallita.")
    audio *= TARGET_PICCO / picco

    n_fade = min(int(FADE_MS * sr / 1000), len(audio) // 2)
    if n_fade > 0:
        ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
        audio[:n_fade] *= ramp
        audio[-n_fade:] *= ramp[::-1]
    return np.clip(audio, -1.0, 1.0)


# ------------------------------ Modello --------------------------------------
def _carica_tts(device: str):
    # Licenza CPML (uso NON commerciale): accettazione headless per Kaggle.
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    # Offline: attacca un dataset con i pesi e imposta TTS_HOME, es.:
    # os.environ["TTS_HOME"] = "/kaggle/input/xtts-v2-pesi"
    from TTS.api import TTS
    return TTS(MODELLO_XTTS, progress_bar=False).to(device)


# ------------------------------ API pubblica ---------------------------------
def genera_voce(
    testo: str,
    out_path: str = PATHS["3A_AUDIO"],
    lingua: str = LINGUA,
    speaker_wav: str | None = SPEAKER_WAV,
    speaker_name: str = SPEAKER_DEFAULT,
    forza: bool = False,
) -> str:
    """Sintetizza `testo` e salva il WAV in `out_path`. Ritorna il path."""
    if os.path.exists(out_path) and not forza:
        print(f"[3A] Skip: {out_path} già presente (forza=True per rigenerare).")
        return out_path

    _assicura_dipendenze()
    import soundfile as sf
    import torch

    chunks = _chunk_testo(testo)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[3A] ATTENZIONE: CUDA non disponibile, sintesi su CPU (molto lenta).")

    if speaker_wav is not None:
        if not os.path.isfile(speaker_wav):
            raise FileNotFoundError(f"[3A] Speaker wav non trovato: {speaker_wav}")
        kwargs_speaker = {"speaker_wav": speaker_wav}
        print(f"[3A] Voice cloning da: {speaker_wav}")
    else:
        kwargs_speaker = {"speaker": speaker_name}
        print(f"[3A] Voce built-in: {speaker_name} (per un it naturale usa speaker_wav)")

    print(f"[3A] {MODELLO_XTTS} | device={device} | chunk={len(chunks)}")
    tts = None
    try:
        tts = _carica_tts(device)
        try:
            sr_modello = int(tts.synthesizer.output_sample_rate)
        except AttributeError:
            sr_modello = SR_TARGET  # XTTS-v2 è nativamente 24 kHz

        silenzio = np.zeros(int(PAUSA_TRA_CHUNK_S * SR_TARGET), dtype=np.float32)
        segmenti: list[np.ndarray] = []
        for i, chunk in enumerate(chunks, 1):
            wav = tts.tts(text=chunk, language=lingua, **kwargs_speaker)
            wav = _resample(np.asarray(wav, dtype=np.float32), sr_modello, SR_TARGET)
            if segmenti:
                segmenti.append(silenzio)
            segmenti.append(wav)
            print(f"[3A] Chunk {i}/{len(chunks)} -> {len(wav) / SR_TARGET:.1f}s")
        audio = _postprocess(np.concatenate(segmenti), SR_TARGET)
    finally:
        # pulisci_memoria(model) fa `del` solo sul riferimento locale:
        # rilasciamo qui quello vero, poi svuotiamo la cache GPU.
        if tts is not None:
            del tts
        pulisci_memoria()

    # Scrittura atomica: il file compare in PATHS solo a sintesi completa.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp.wav"
    sf.write(tmp, audio, SR_TARGET, subtype="PCM_16")
    os.replace(tmp, out_path)

    durata = len(audio) / SR_TARGET
    if durata < 0.3:
        raise RuntimeError(f"[3A] Audio troppo corto ({durata:.2f}s): controlla il testo.")
    mb = os.path.getsize(out_path) / 1e6
    print(f"[3A] OK -> {out_path} | {durata:.2f}s | {SR_TARGET} Hz mono PCM16 | {mb:.2f} MB")
    if durata > 120:
        print("[3A] Avviso: >120s, lo step 3B sarà pesante: valuta di spezzare il copione.")
    return out_path


def esegui_3a(testo: str | None = None, forza: bool = False) -> str:
    """Entry-point per l'orchestratore della pipeline."""
    return genera_voce(testo if testo is not None else TESTO_DEFAULT, forza=forza)


if __name__ == "__main__":
    testo_cli = " ".join(sys.argv[1:]).strip()
    esegui_3a(testo_cli or None)
