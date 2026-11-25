"""
modulo_4b.py

Step 4B: final rendering 1080p.

Input:
- output_pipeline/step_4a_smooth.mp4

Output:
- output_pipeline/step_4b_final_1080p.mp4

Logica:
- Scala/pad verso 1920x1080 mantenendo aspect ratio.
- Priorità audio:
  1) step_3a_voce.wav
  2) step_3b_lipsync.mp4
  3) audio eventualmente già presente in step_4a_smooth.mp4
  4) output solo video
- Encoding H.264 + AAC per massima compatibilità MP4.
- Nessun uso diretto di GPU/VRAM: encoding CPU via ffmpeg.
"""

import os
import shutil
import subprocess

from config import PATHS, pulisci_memoria


TARGET_W = 1920
TARGET_H = 1080

# Fit dentro 1920x1080, poi pad nero centrale se l'aspect non è 16:9.
VIDEO_FILTER = (
    f"scale=w='min({TARGET_W},iw*{TARGET_H}/ih)':"
    f"h='min({TARGET_H},ih*{TARGET_W}/iw)':"
    "force_original_aspect_ratio=decrease:"
    "force_divisible_by=2:"
    "flags=lanczos,"
    f"pad=w={TARGET_W}:h={TARGET_H}:x=({TARGET_W}-iw)/2:y=({TARGET_H}-ih)/2:color=black,"
    "setsar=1"
)


class FFmpegError(RuntimeError):
    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


def _get_ffmpeg() -> str:
    """Trova l'eseguibile ffmpeg, con fallback su imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe

    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg non disponibile. Installa ffmpeg oppure `pip install imageio-ffmpeg`."
        ) from exc

    raise RuntimeError("ffmpeg non disponibile.")


def _probe_stderr(ffmpeg: str, path: str) -> str:
    """Esegue ffmpeg -i sul file e restituisce stderr per ispezione stream."""
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return ""

    cmd = [ffmpeg, "-hide_banner", "-i", path]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        errors="ignore",
    )
    return proc.stderr or ""


def _has_video(ffmpeg: str, path: str) -> bool:
    info = _probe_stderr(ffmpeg, path)
    return "Video:" in info


def _has_audio(ffmpeg: str, path: str) -> bool:
    info = _probe_stderr(ffmpeg, path)
    return "Audio:" in info


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Esegue un comando ffmpeg e solleva errore dettagliato se fallisce."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="ignore",
    )

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-4000:]
        message = (
            f"ffmpeg fallito con codice {proc.returncode}.\n"
            f"Comando: {' '.join(cmd)}\n"
            f"STDERR:\n{stderr_tail}"
        )
        raise FFmpegError(message, stderr=proc.stderr or "")

    return proc


def _is_audio_related_error(stderr: str) -> bool:
    """
    Cerca segnali di errore legati all'audio per provare fallback senza audio.
    Non usa match troppo generici per evitare falsi positivi su errori video.
    """
    s = (stderr or "").lower()
    keys = (
        "apad",
        "no such filter",
        "matches no streams",
        "stream map",
        "audio codec",
        "audio stream",
        "sample rate",
        "samplerate",
        "channel layout",
        "resampl",
        "could not open codec",
        "codec not currently supported in container",
        "invalid data found when processing input",
    )
    return any(k in s for k in keys)


def _build_command(
    ffmpeg: str,
    video_in: str,
    video_out: str,
    audio_source: str | None = None,
    embedded_audio: bool = False,
) -> list[str]:
    """
    Costruisce il comando ffmpeg finale.

    audio_source:
        Se valorizzato, viene usato come secondo input audio.
    embedded_audio:
        Se True e audio_source è None, prova a usare l'audio dentro video_in.
    """
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
    ]

    cmd.extend(["-i", video_in])

    if audio_source:
        cmd.extend(["-i", audio_source])
        cmd.extend([
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-af", "apad",
            "-shortest",
        ])
    elif embedded_audio:
        cmd.extend([
            "-map", "0:v:0",
            "-map", "0:a:0",
        ])
    else:
        cmd.extend([
            "-map", "0:v:0",
            "-an",
        ])

    cmd.extend([
        "-vf", VIDEO_FILTER,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        "-threads", "0",
    ])

    if audio_source or embedded_audio:
        cmd.extend([
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
        ])

    cmd.extend([
        "-movflags", "+faststart",
        "-max_muxing_queue_size", "1024",
        video_out,
    ])

    return cmd


def _remove_quiet(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def main() -> None:
    ffmpeg = _get_ffmpeg()

    video_in = PATHS["4A_SMOOTH"]
    video_out = PATHS["4B_FINAL"]

    if not os.path.isfile(video_in) or os.path.getsize(video_in) == 0:
        raise FileNotFoundError(f"Input 4A mancante o vuoto: {video_in}")

    video_info = _probe_stderr(ffmpeg, video_in)
    if "Video:" not in video_info:
        raise ValueError(f"Il file 4A non contiene una traccia video valida: {video_in}")

    has_embedded_audio = "Audio:" in video_info

    os.makedirs(os.path.dirname(video_out), exist_ok=True)

    # Priorità audio esterna.
    audio_candidates = [
        PATHS["3A_AUDIO"],
        PATHS["3B_LIPSYNC"],
    ]

    audio_source = None
    for candidate in audio_candidates:
        if _has_audio(ffmpeg, candidate):
            audio_source = candidate
            break

    print("[4B] Final rendering 1080p")
    print(f"[4B] Video in: {video_in}")

    if audio_source:
        print(f"[4B] Audio esterno scelto: {audio_source}")
    elif has_embedded_audio:
        print("[4B] Uso audio eventualmente già presente in 4A.")
    else:
        print("[4B] Nessuna traccia audio disponibile: output solo video.")

    # Tentativi: esterno -> embedded -> muto.
    attempts: list[tuple[str | None, bool]] = []

    if audio_source:
        attempts.append((audio_source, False))

    if has_embedded_audio:
        attempts.append((None, True))

    attempts.append((None, False))

    last_error: FFmpegError | None = None

    for idx, (current_audio_source, use_embedded_audio) in enumerate(attempts):
        if current_audio_source:
            label = f"audio esterno: {current_audio_source}"
        elif use_embedded_audio:
            label = "audio embedded in 4A"
        else:
            label = "senza audio"

        print(f"[4B] Tentativo {idx + 1}/{len(attempts)}: {label}")

        cmd = _build_command(
            ffmpeg=ffmpeg,
            video_in=video_in,
            video_out=video_out,
            audio_source=current_audio_source,
            embedded_audio=use_embedded_audio,
        )

        try:
            _run(cmd)
            break
        except FFmpegError as exc:
            last_error = exc

            is_last_attempt = idx == len(attempts) - 1

            if is_last_attempt or not _is_audio_related_error(exc.stderr):
                _remove_quiet(video_out)
                raise

            print(
                "[4B] Errore probabilmente audio. "
                "Provo fallback con configurazione audio successiva."
            )
            _remove_quiet(video_out)
    else:
        if last_error is not None:
            raise last_error

    if not os.path.isfile(video_out) or os.path.getsize(video_out) == 0:
        raise RuntimeError(f"Output 4B non creato o vuoto: {video_out}")

    size_mb = os.path.getsize(video_out) / (1024 * 1024)
    print(f"[4B] Output pronto: {video_out} ({size_mb:.2f} MB)")

    pulisci_memoria()


if __name__ == "__main__":
    main()
