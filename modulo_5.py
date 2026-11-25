import os
import json
import subprocess
from pathlib import Path

from config import OUTPUT_DIR, PATHS

_AUDIO_SAMPLE_RATE = 44100
_DEFAULT_FPS = 30


def _run(cmd):
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Eseguibile mancante: '{cmd[0]}'. Installa FFmpeg/FFprobe e assicurati che sia nel PATH."
        ) from exc

    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        tail = "\n".join(details.splitlines()[-15:])
        raise RuntimeError(f"Comando esterno fallito ({cmd[0]}).\n{tail}")

    return proc


def _probe_media(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    raw = _run(cmd).stdout

    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe ha restituito JSON non valido per: {path}") from exc

    streams = data.get("streams", []) or []

    video = None
    for s in streams:
        if s.get("codec_type") == "video":
            disposition = s.get("disposition", {}) or {}
            if not disposition.get("attached_pic", 0):
                video = s
                break
    if video is None:
        video = next((s for s in streams if s.get("codec_type") == "video"), None)

    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = 0.0
    for source in (data.get("format", {}) or {}, video or {}, audio or {}):
        try:
            value = float(source.get("duration"))
            if value > 0:
                duration = value
                break
        except (TypeError, ValueError):
            pass

    width = 0
    height = 0
    if video:
        try:
            width = int(video.get("width") or 0)
        except (TypeError, ValueError):
            width = 0
        try:
            height = int(video.get("height") or 0)
        except (TypeError, ValueError):
            height = 0

    return {
        "path": path,
        "has_video": video is not None,
        "has_audio": audio is not None,
        "duration": duration,
        "width": width,
        "height": height,
    }


def _resolve_media(candidate, label="file"):
    if candidate is None or str(candidate).strip() == "":
        raise ValueError(f"Percorso {label} non valido.")

    p = Path(str(candidate)).expanduser()

    if p.is_file():
        return p.resolve()

    if not p.is_absolute():
        alt = Path(OUTPUT_DIR) / p
        if alt.is_file():
            return alt.resolve()

    raise FileNotFoundError(
        f"{label} non trovato: '{candidate}' (cercato anche in OUTPUT_DIR)."
    )


def _even_size(value, fallback: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = fallback

    if v <= 0:
        v = fallback

    if v % 2 != 0:
        v += 1

    return v


def esegui_5(
    lista_clip: list = None,
    path_audio_ambiente: str = None,
    volume_ambiente: float = 0.15,
    nome_output: str = "scena_finita_master.mp4",
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not nome_output:
        nome_output = "scena_finita_master.mp4"

    if lista_clip is None or (isinstance(lista_clip, (list, tuple)) and len(lista_clip) == 0):
        if not isinstance(PATHS, dict) or "4B_FINAL" not in PATHS:
            raise ValueError("config.PATHS non contiene la chiave '4B_FINAL'.")

        default_clip = PATHS["4B_FINAL"]
        raw_clips = list(default_clip) if isinstance(default_clip, (list, tuple)) else [default_clip]

    elif isinstance(lista_clip, (str, Path)):
        raw_clips = [lista_clip]

    else:
        raw_clips = list(lista_clip)

    if not raw_clips:
        raise ValueError("Nessuna clip disponibile per il montaggio finale.")

    infos = []
    for clip in raw_clips:
        clip_path = _resolve_media(clip, "clip")
        info = _probe_media(clip_path)

        if not info["has_video"]:
            raise ValueError(f"La clip non contiene uno stream video: {clip_path}")

        if info["duration"] <= 0:
            raise ValueError(f"Durata non rilevabile per la clip: {clip_path}")

        infos.append(info)

    first = infos[0]
    width = _even_size(first.get("width"), 1280)
    height = _even_size(first.get("height"), 720)

    try:
        volume_ambiente = float(volume_ambiente)
    except (TypeError, ValueError):
        volume_ambiente = 0.0

    volume_ambiente = max(0.0, min(10.0, volume_ambiente))

    ambient_path = None
    if path_audio_ambiente and volume_ambiente > 0.0:
        ambient_path = _resolve_media(path_audio_ambiente, "audio ambiente")
        amb_info = _probe_media(ambient_path)

        if not amb_info["has_audio"]:
            raise ValueError(f"Il file audio ambiente non contiene audio: {ambient_path}")

        if amb_info["duration"] <= 0:
            raise ValueError(f"Durata non rilevabile per l'audio ambiente: {ambient_path}")

    total_duration = sum(float(info["duration"]) for info in infos)
    if total_duration <= 0:
        raise ValueError("Durata totale non valida per il montaggio finale.")

    fade_dur = min(1.0, total_duration / 3.0)
    if fade_dur < 0.05:
        fade_dur = 0.0

    filter_parts = []
    concat_labels = []

    for i, info in enumerate(infos):
        dur = float(info["duration"])

        filter_parts.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={_DEFAULT_FPS},format=yuv420p,"
            f"setpts=PTS-STARTPTS,"
            f"tpad=stop_mode=clone:stop_duration={dur:.6f},"
            f"trim=duration={dur:.6f},"
            f"setpts=PTS-STARTPTS[v{i}]"
        )

        if info["has_audio"]:
            filter_parts.append(
                f"[{i}:a]aresample={_AUDIO_SAMPLE_RATE},"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"asetpts=PTS-STARTPTS,"
                f"atrim=duration={dur:.6f},"
                f"asetpts=PTS-STARTPTS,"
                f"apad=whole_dur={dur:.6f}[a{i}]"
            )
        else:
            filter_parts.append(
                f"anullsrc=r={_AUDIO_SAMPLE_RATE}:cl=stereo:d={dur:.6f},"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]"
            )

        concat_labels.append(f"[v{i}][a{i}]")

    filter_parts.append(
        "".join(concat_labels) + f"concat=n={len(infos)}:v=1:a=1[vcat][acat]"
    )

    if fade_dur > 0:
        fade_out_start = max(0.0, total_duration - fade_dur)

        filter_parts.append(
            f"[vcat]fade=in:st=0:d={fade_dur:.6f},"
            f"fade=out:st={fade_out_start:.6f}:d={fade_dur:.6f},"
            f"format=yuv420p[vout]"
        )

        audio_fade = (
            f",afade=in:st=0:d={fade_dur:.6f},"
            f"afade=out:st={fade_out_start:.6f}:d={fade_dur:.6f}"
        )
    else:
        filter_parts.append("[vcat]format=yuv420p[vout]")
        audio_fade = ""

    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]

    for info in infos:
        cmd.extend(["-i", str(info["path"])])

    if ambient_path:
        amb_index = len(infos)
        cmd.extend(["-stream_loop", "-1", "-i", str(ambient_path)])

        filter_parts.append(
            f"[acat]aresample={_AUDIO_SAMPLE_RATE},"
            f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"atrim=duration={total_duration:.6f},"
            f"asetpts=PTS-STARTPTS,"
            f"apad=whole_dur={total_duration:.6f}[voice]"
        )

        filter_parts.append(
            f"[{amb_index}:a]aresample={_AUDIO_SAMPLE_RATE},"
            f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"volume={volume_ambiente:.6f},"
            f"atrim=duration={total_duration:.6f},"
            f"asetpts=PTS-STARTPTS,"
            f"apad=whole_dur={total_duration:.6f}[amb]"
        )

        filter_parts.append(
            f"[voice][amb]amix=inputs=2:duration=first:dropout_transition=0,"
            f"volume=2.0{audio_fade}[aout]"
        )

    else:
        filter_parts.append(
            f"[acat]aresample={_AUDIO_SAMPLE_RATE},"
            f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"atrim=duration={total_duration:.6f},"
            f"asetpts=PTS-STARTPTS{audio_fade}[aout]"
        )

    out_path = os.path.abspath(os.path.join(OUTPUT_DIR, nome_output))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd.extend([
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-threads", "0",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", str(_AUDIO_SAMPLE_RATE),
        "-ac", "2",
    ])

    if out_path.lower().endswith((".mp4", ".mov", ".m4a")):
        cmd.extend(["-movflags", "+faststart"])

    cmd.append(out_path)

    _run(cmd)

    return out_path
