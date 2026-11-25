# -*- coding: utf-8 -*-
"""
app.py
Interfaccia centrale Gradio per orchestrare la pipeline video AI su Kaggle (GPU T4 16GB).

Moduli richiesti nella stessa cartella:
- config.py
- modulo_1a.py ... modulo_5.py
"""

import os
import gc
import sys
import json
import time
import glob
import random
import inspect
import asyncio
import pathlib
import traceback
import importlib
from datetime import datetime
from typing import Any, Dict, List, Optional

import gradio as gr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    from config import OUTPUT_DIR, PATHS, pulisci_memoria as _config_pulisci_memoria
except Exception:
    OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), "output"))
    PATHS = {}
    _config_pulisci_memoria = None

OUTPUT_DIR = str(OUTPUT_DIR or os.path.abspath(os.path.join(os.getcwd(), "output")))
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not isinstance(PATHS, dict):
    PATHS = {}


def _detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = _detect_device()


def pulisci_memoria_safe():
    """
    Chiama pulisci_memoria() di config se disponibile, altrimenti fa fallback
    su gc + svuotamento cache CUDA.
    """
    try:
        if callable(_config_pulisci_memoria):
            _config_pulisci_memoria()
        else:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Definizione moduli
# ---------------------------------------------------------------------------
MODULE_DEFS = {
    "1a": {"module": "modulo_1a", "funcs": ["esegui_1a"], "label": "1A. Sfondo Scena"},
    "1b": {"module": "modulo_1b", "funcs": ["esegui_1b", "genera_keyframe"], "label": "1B. Inserimento Attore"},
    "1c": {"module": "modulo_1c", "funcs": ["esegui_1c"], "label": "1C. Controllo Posa/Mani"},
    "2a": {"module": "modulo_2a", "funcs": ["esegui_2a"], "label": "2A. Animazione Video"},
    "2b": {"module": "modulo_2b", "funcs": ["esegui_2b"], "label": "2B. Continuità Clip"},
    "3a": {"module": "modulo_3a", "funcs": ["esegui_3a"], "label": "3A. Voce Italiana (TTS)"},
    "3b": {"module": "modulo_3b", "funcs": ["esegui_3b"], "label": "3B. Labiale (Lip-Sync)"},
    "4a": {"module": "modulo_4a", "funcs": ["esegui_4a"], "label": "4A. Fluidità 24/30 fps"},
    "4b": {"module": "modulo_4b", "funcs": ["esegui_4b"], "label": "4B. Upscale 1080p"},
    "5": {"module": "modulo_5", "funcs": ["esegui_5"], "label": "5. Montaggio & Sound Design"},
}

STEP_ORDER = ["1a", "1b", "1c", "2a", "2b", "3a", "3b", "4a", "4b", "5"]

_MODULE_CACHE: Dict[str, Any] = {}
_FUNCTION_CACHE: Dict[str, Any] = {}
_LOAD_ERRORS: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Utils file / output
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

IMAGE_PATTERNS = ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.webp", "**/*.bmp", "**/*.tif", "**/*.tiff", "**/*.gif"]
VIDEO_PATTERNS = ["**/*.mp4", "**/*.mov", "**/*.avi", "**/*.mkv", "**/*.webm", "**/*.m4v", "**/*.wmv"]
AUDIO_PATTERNS = ["**/*.wav", "**/*.mp3", "**/*.flac", "**/*.ogg", "**/*.m4a", "**/*.aac"]


def _is_hidden(path: str) -> bool:
    try:
        return any(part.startswith(".") for part in pathlib.Path(path).parts)
    except Exception:
        return False


def _safe_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


def snapshot_files() -> Dict[str, float]:
    snap = {}
    if not os.path.isdir(OUTPUT_DIR):
        return snap
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            p = os.path.join(root, f)
            if _is_hidden(p):
                continue
            snap[p] = _safe_mtime(p)
    return snap


def new_or_modified_files(before: Dict[str, float], after: Dict[str, float]) -> List[str]:
    out = []
    for p, m in after.items():
        if p not in before or before[p] != m:
            out.append(p)
    out.sort(key=lambda x: _safe_mtime(x), reverse=True)
    return out


def latest_file(patterns: List[str], exts: Optional[set] = None) -> Optional[str]:
    candidates = []
    for pat in patterns:
        try:
            matches = glob.glob(os.path.join(OUTPUT_DIR, pat), recursive=True)
        except Exception:
            matches = []
        for p in matches:
            if not os.path.isfile(p) or _is_hidden(p):
                continue
            ext = os.path.splitext(p)[1].lower()
            if exts is None or ext in exts:
                candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda x: _safe_mtime(x))


def config_path(*names) -> Optional[str]:
    for n in names:
        if not isinstance(PATHS, dict):
            continue
        v = PATHS.get(n)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, (str, pathlib.Path)) and os.path.exists(str(item)):
                    return str(item)
        elif isinstance(v, (str, pathlib.Path)) and os.path.exists(str(v)):
            return str(v)
    return None


def latest_image() -> Optional[str]:
    return (
        config_path("sfondo", "background", "background_image", "keyframe", "image", "immagine")
        or latest_file(IMAGE_PATTERNS, IMAGE_EXTS)
    )


def latest_video() -> Optional[str]:
    return (
        config_path("video", "clip", "last_video", "last_clip")
        or latest_file(VIDEO_PATTERNS, VIDEO_EXTS)
    )


def latest_audio() -> Optional[str]:
    return (
        config_path("audio", "voice", "tts", "voce")
        or latest_file(AUDIO_PATTERNS, AUDIO_EXTS)
    )


def classify_paths(paths: Optional[List[str]]):
    unique = []
    seen = set()
    for p in paths or []:
        try:
            if not p:
                continue
            p = str(p)
            if os.path.isfile(p) and p not in seen:
                seen.add(p)
                unique.append(p)
        except Exception:
            continue

    image = next((p for p in unique if os.path.splitext(p)[1].lower() in IMAGE_EXTS), None)
    video = next((p for p in unique if os.path.splitext(p)[1].lower() in VIDEO_EXTS), None)
    audio = next((p for p in unique if os.path.splitext(p)[1].lower() in AUDIO_EXTS), None)
    return unique, image, video, audio


# ---------------------------------------------------------------------------
# Import dinamico moduli
# ---------------------------------------------------------------------------
def get_module(key: str):
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]

    mod_name = MODULE_DEFS[key]["module"]
    try:
        mod = importlib.import_module(mod_name)
        _MODULE_CACHE[key] = mod
        return mod
    except Exception as e:
        _MODULE_CACHE[key] = None
        _LOAD_ERRORS[key] = f"{type(e).__name__}: {e}"
        return None


def get_function(key: str):
    if key in _FUNCTION_CACHE:
        return _FUNCTION_CACHE[key]

    mod = get_module(key)
    if mod is None:
        _FUNCTION_CACHE[key] = None
        return None

    for fn_name in MODULE_DEFS[key]["funcs"]:
        if hasattr(mod, fn_name):
            fn = getattr(mod, fn_name)
            if callable(fn):
                _FUNCTION_CACHE[key] = fn
                return fn

    err = f"Funzione non trovata in {MODULE_DEFS[key]['module']}. Cercate: {', '.join(MODULE_DEFS[key]['funcs'])}"
    _LOAD_ERRORS[key] = err
    _FUNCTION_CACHE[key] = None
    return None


# ---------------------------------------------------------------------------
# Matching parametri / valori UI
# ---------------------------------------------------------------------------
CANONICAL_ALIASES = {
    "prompt": {"prompt", "caption", "description", "descrizione", "subject", "scene", "scene_prompt"},
    "negative_prompt": {"negative_prompt", "negative", "negativo", "neg_prompt"},
    "text": {"text", "testo", "sentence", "frase", "script", "transcript", "lyrics"},
    "image": {
        "image", "img", "immagine", "sfondo", "background", "background_image",
        "keyframe", "frame", "input_image", "source_image", "base_image"
    },
    "reference_image": {
        "reference_image", "ref_image", "reference", "riferimento",
        "reference_img", "style_image", "actor_image"
    },
    "pose_image": {
        "pose_image", "pose", "control_image", "pose_ref",
        "hands_image", "hand_image", "mani", "pose_reference"
    },
    "video": {"video", "clip", "mp4", "input_video", "source_video", "movie", "footage"},
    "reference_video": {"reference_video", "ref_video", "video_riferimento", "reference_clip"},
    "audio": {
        "audio", "voice_audio", "speech_audio", "voce", "speech",
        "wav", "mp3", "input_audio", "tts_audio"
    },
    "music": {"music", "musica", "soundtrack", "sfx", "sound_design", "background_music"},
    "voice": {"voice", "speaker", "voice_id", "voice_name", "voce_id"},
    "language": {"language", "lingua", "lang"},
    "seed": {"seed", "seme", "random_seed", "random_state"},
    "steps": {"steps", "num_inference_steps", "inference_steps", "step"},
    "guidance_scale": {"guidance_scale", "guidance", "cfg", "cfg_scale", "guidance_strength"},
    "strength": {"strength", "control_strength", "denoise", "denoising_strength", "motion_strength"},
    "fps": {"fps", "frame_rate", "framerate"},
    "target_fps": {"target_fps", "output_fps", "new_fps", "desired_fps", "fps_target"},
    "width": {"width", "larghezza", "w"},
    "height": {"height", "altezza", "h"},
    "duration": {"duration", "durata", "seconds", "length", "num_seconds"},
    "target_resolution": {"target_resolution", "resolution", "res", "risoluzione", "size"},
    "speed": {"speed", "velocita", "rate", "speech_rate", "tempo", "playback_rate"},
    "clips": {"clips", "video_clips", "clip_list", "videos", "files", "media_files"},
    "output_dir": {"output_dir", "outdir", "output_folder", "save_dir", "save_folder", "destination_dir"},
    "output_path": {"output_path", "out_path", "save_path", "destination", "output_file", "outfile", "output"},
    "device": {"device", "device_map", "torch_device", "cuda_device"},
    "model": {"model", "model_name", "checkpoint", "ckpt", "model_id"},
    "lora": {"lora", "lora_path", "adapter", "adapter_path"},
}

ALIAS_TO_CANONICAL = {}
for _canon, _aliases in CANONICAL_ALIASES.items():
    for _alias in _aliases:
        ALIAS_TO_CANONICAL[_alias.lower()] = _canon


def canonical_from_param(name: str) -> str:
    if not name:
        return ""
    n = str(name).lower().strip().replace("-", "_").replace(" ", "_")
    if n in ALIAS_TO_CANONICAL:
        return ALIAS_TO_CANONICAL[n]

    for canon, aliases in CANONICAL_ALIASES.items():
        for alias in aliases:
            if len(alias) >= 4 and alias in n:
                return canon
    return n


def clean_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v if v else None
    if isinstance(v, (list, tuple, set)):
        vals = [clean_value(x) for x in v]
        vals = [x for x in vals if x is not None]
        return vals if vals else None
    if isinstance(v, (int, float, bool)):
        return v
    return v


def prepare_values(keys: List[str], args: List[Any]) -> Dict[str, Any]:
    values = {}
    for k, v in zip(keys, args):
        values[k] = clean_value(v)

    extra_raw = values.get("extra_json")
    if extra_raw:
        try:
            extra = json.loads(extra_raw)
            if isinstance(extra, dict):
                values["_extra"] = extra
                for ek, ev in extra.items():
                    if ek not in values or clean_value(values[ek]) is None:
                        values[ek] = clean_value(ev)
        except Exception:
            values["_extra_error"] = "Extra JSON non valido"

    if "seed" in values:
        try:
            seed_val = values["seed"]
            if seed_val is None:
                values["seed"] = random.randint(0, 2**31 - 1)
            else:
                s = int(float(seed_val))
                if s < 0:
                    s = random.randint(0, 2**31 - 1)
                values["seed"] = s
        except Exception:
            values["seed"] = random.randint(0, 2**31 - 1)

    return values


def build_auto_context(key: str, values: Dict[str, Any]) -> Dict[str, Any]:
    auto = {
        "device": DEVICE,
        "output_dir": OUTPUT_DIR,
    }

    img = latest_image()
    vid = latest_video()
    aud = latest_audio()

    if key in ("1b", "1c", "2a") and not values.get("image") and img:
        auto["image"] = img

    if key == "1c" and not values.get("pose_image"):
        pose = (
            config_path("posa", "pose", "pose_image", "hands", "mani")
            or latest_file(
                ["**/pose*.png", "**/pose*.jpg", "**/pose*.jpeg", "**/hands*.png", "**/mani*.png"],
                IMAGE_EXTS,
            )
        )
        if pose:
            auto["pose_image"] = pose

    if key in ("2b", "3b", "4a", "4b", "5") and not values.get("video") and vid:
        auto["video"] = vid

    if key == "2b" and not values.get("reference_video") and vid:
        auto["reference_video"] = vid

    if key in ("3b", "5") and not values.get("audio") and aud:
        auto["audio"] = aud

    if key == "3a":
        if not values.get("text") and values.get("prompt"):
            auto["text"] = values["prompt"]

    if key == "5" and not values.get("music"):
        music = (
            config_path("music", "soundtrack", "sfx")
            or latest_file(
                [
                    "**/music*.mp3", "**/music*.wav", "**/soundtrack*.mp3",
                    "**/soundtrack*.wav", "**/sfx*.mp3", "**/sfx*.wav"
                ],
                AUDIO_EXTS,
            )
        )
        if music:
            auto["music"] = music

    return auto


def find_value_for_param(param_name: str, values: Dict[str, Any], auto: Dict[str, Any]) -> Any:
    if param_name in values:
        v = clean_value(values[param_name])
        if v is not None:
            return v

    canon = canonical_from_param(param_name)

    if canon and canon in values:
        v = clean_value(values[canon])
        if v is not None:
            return v

    if canon and canon in auto:
        v = clean_value(auto[canon])
        if v is not None:
            return v

    # Fallback mirati
    if canon == "text" and clean_value(values.get("prompt")) is not None:
        return values["prompt"]
    if canon == "prompt" and clean_value(values.get("text")) is not None:
        return values["text"]
    if canon == "target_fps" and clean_value(values.get("fps")) is not None:
        return values["fps"]
    if canon == "fps" and clean_value(values.get("target_fps")) is not None:
        return values["target_fps"]

    return None


def build_call_args(fn: callable, values: Dict[str, Any], auto: Dict[str, Any]):
    sig = inspect.signature(fn)
    args = []
    kwargs = {}
    passed_keys = set()
    has_varkw = False

    for p in sig.parameters.values():
        if p.kind == p.VAR_KEYWORD:
            has_varkw = True
            continue
        if p.kind == p.VAR_POSITIONAL:
            continue

        val = find_value_for_param(p.name, values, auto)

        if val is None:
            if p.default is inspect._empty:
                if p.annotation is bool:
                    val = False
                else:
                    continue
            else:
                continue

        if p.kind == p.POSITIONAL_ONLY:
            args.append(val)
        else:
            kwargs[p.name] = val
            passed_keys.add(p.name)
            canon = canonical_from_param(p.name)
            if canon:
                passed_keys.add(canon)

    if has_varkw:
        skip = {"extra_json", "_extra", "_extra_error", "status"}
        for k, v in values.items():
            if k in skip or k in kwargs or k.startswith("_"):
                continue
            cv = clean_value(v)
            if cv is None:
                continue
            kwargs[k] = cv

        extra = values.get("_extra")
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k not in kwargs:
                    cv = clean_value(v)
                    if cv is not None:
                        kwargs[k] = cv

    return args, kwargs


# ---------------------------------------------------------------------------
# Invocazione funzioni
# ---------------------------------------------------------------------------
def invoke_function(fn: callable, args: List[Any], kwargs: Dict[str, Any]) -> Any:
    if inspect.iscoroutinefunction(fn):
        return asyncio.run(fn(*args, **kwargs))
    return fn(*args, **kwargs)


def call_function_safe(fn: callable, args: List[Any], kwargs: Dict[str, Any]) -> Any:
    try:
        return invoke_function(fn, args, kwargs)
    except TypeError as e:
        msg = str(e).lower()
        sig = inspect.signature(fn)
        has_varkw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        allowed = {
            p.name
            for p in sig.parameters.values()
            if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        }

        if not has_varkw and kwargs and ("unexpected keyword argument" in msg or "argument" in msg):
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            if len(filtered_kwargs) != len(kwargs):
                try:
                    return invoke_function(fn, args, filtered_kwargs)
                except Exception:
                    pass
        raise


# ---------------------------------------------------------------------------
# Normalizzazione output
# ---------------------------------------------------------------------------
def maybe_save_ndarray(obj: Any) -> Optional[str]:
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            from PIL import Image
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out = os.path.join(OUTPUT_DIR, f"auto_image_{ts}.png")
            try:
                img = Image.fromarray(obj)
            except Exception:
                img = Image.fromarray(obj.astype("uint8"))
            img.save(out)
            return out
    except Exception:
        return None
    return None


def collect_paths(obj: Any, found: List[str], depth: int = 0):
    if obj is None or depth > 5:
        return

    saved = maybe_save_ndarray(obj)
    if saved:
        found.append(saved)
        return

    if isinstance(obj, (str, pathlib.Path)):
        p = str(obj)
        if os.path.exists(p):
            found.append(p)
        return

    if isinstance(obj, dict):
        priority_keys = (
            "output", "outputs", "result", "results", "image", "images",
            "video", "videos", "audio", "audios", "file", "files",
            "path", "paths", "artifact", "artifacts"
        )
        for k in priority_keys:
            if k in obj:
                collect_paths(obj[k], found, depth + 1)
        for k, v in obj.items():
            if k not in priority_keys:
                collect_paths(v, found, depth + 1)
        return

    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            collect_paths(v, found, depth + 1)
        return

    for attr in ("path", "filepath", "file_path", "filename", "file", "url", "uri", "video", "image", "audio"):
        try:
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if isinstance(val, (str, pathlib.Path)) and os.path.exists(str(val)):
                    found.append(str(val))
                    return
                collect_paths(val, found, depth + 1)
        except Exception:
            pass


def normalize_result(result: Any, new_files: List[str]):
    found = []
    log = result

    if result is not None:
        saved = maybe_save_ndarray(result)
        if saved:
            found.append(saved)
        else:
            collect_paths(result, found)

        if isinstance(result, dict):
            log = result
        elif isinstance(result, (list, tuple)):
            log = result
        elif not isinstance(result, (str, pathlib.Path)):
            log = str(result)
        else:
            log = {"output": str(result)}

    for p in new_files or []:
        if p not in found:
            found.append(p)

    files, image, video, audio = classify_paths(found)
    return files, image, video, audio, log


def json_safe(obj: Any, depth: int = 0) -> Any:
    if depth > 6:
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v, depth + 1) for v in obj]
    return str(obj)


# ---------------------------------------------------------------------------
# Stato / esecuzione
# ---------------------------------------------------------------------------
def format_status(ok: bool, key: str, message: str, elapsed: Optional[float] = None, files: Optional[List[str]] = None) -> str:
    icon = "✅" if ok else "❌"
    if key == "all":
        label = "Esegui Tutto in Cascata"
    else:
        label = MODULE_DEFS.get(key, {}).get("label", key)

    lines = [f"## {icon} {label}", message]
    if elapsed is not None:
        lines.append(f"**Durata:** {elapsed:.2f}s")
    if files:
        lines.append("**Articoli generati/modificati:**")
        for f in files[:12]:
            lines.append(f"- `{os.path.basename(f)}`")
    lines.append(f"**Aggiornato:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n\n".join(lines)


def make_result(
    ok: bool,
    key: str,
    message: str,
    files: Optional[List[str]] = None,
    image: Optional[str] = None,
    video: Optional[str] = None,
    audio: Optional[str] = None,
    log: Any = None,
    elapsed: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "ok": ok,
        "status": format_status(ok, key, message, elapsed, files),
        "files": files or [],
        "image": image,
        "video": video,
        "audio": audio,
        "log": log,
    }


def result_to_ui(res: Dict[str, Any]):
    return (
        res.get("status", ""),
        res.get("files") or None,
        res.get("image"),
        res.get("video"),
        res.get("audio"),
        json_safe(res.get("log")),
    )


def update_progress(progress, frac: float, desc: Optional[str] = None):
    if progress is None:
        return
    try:
        if desc is not None:
            progress(frac, desc=desc)
        else:
            progress(frac)
    except Exception:
        try:
            progress(frac)
        except Exception:
            pass


def execute_module(key: str, values: Dict[str, Any], progress=None) -> Dict[str, Any]:
    label = MODULE_DEFS[key]["label"]
    update_progress(progress, 0.05, f"Preparazione {label}")

    fn = get_function(key)
    if fn is None:
        err = _LOAD_ERRORS.get(key, "Funzione non trovata")
        return make_result(False, key, f"Modulo non disponibile: {err}", log={"error": err})

    auto = build_auto_context(key, values)
    args, kwargs = build_call_args(fn, values, auto)

    update_progress(progress, 0.15, f"Lancio {label}")
    before = snapshot_files()
    t0 = time.time()

    try:
        result = call_function_safe(fn, args, kwargs)

        if result is False:
            raise RuntimeError("La funzione del modulo ha restituito False.")

        elapsed = time.time() - t0
        update_progress(progress, 0.90, f"Elaborazione output {label}")

        after = snapshot_files()
        new_files = new_or_modified_files(before, after)
        files, image, video, audio, log = normalize_result(result, new_files)

        pulisci_memoria_safe()
        update_progress(progress, 1.0, f"{label} completato")

        return make_result(True, key, "Modulo completato.", files, image, video, audio, log, elapsed)

    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        pulisci_memoria_safe()
        update_progress(progress, 1.0, f"{label} fallito")
        return make_result(
            False,
            key,
            f"Errore: {e}",
            [],
            None,
            None,
            None,
            {"error": str(e), "traceback": tb},
            elapsed,
        )


# ---------------------------------------------------------------------------
# UI specs
# ---------------------------------------------------------------------------
TAB_SPECS = {
    "1a": {
        "label": "1A. Sfondo Scena",
        "description": "Genera lo sfondo della scena. Se il modulo accetta parametri extra, usa il campo JSON.",
        "fields": [
            {"type": "textbox", "key": "prompt", "label": "Prompt scena", "props": {"lines": 3}},
            {"type": "textbox", "key": "negative_prompt", "label": "Negative prompt", "props": {"lines": 2}},
            {"type": "number", "key": "seed", "label": "Seed (-1 = random)", "value": -1},
            {"type": "number", "key": "width", "label": "Larghezza", "value": 1024},
            {"type": "number", "key": "height", "label": "Altezza", "value": 576},
            {"type": "slider", "key": "steps", "label": "Steps", "value": 28, "minimum": 1, "maximum": 100, "step": 1},
            {"type": "slider", "key": "guidance_scale", "label": "Guidance scale", "value": 7.0, "minimum": 0.0, "maximum": 30.0, "step": 0.5},
            {"type": "textbox", "key": "extra_json", "label": "Extra kwargs JSON", "props": {"lines": 2, "placeholder": '{"model": "..."}'}},
        ],
    },
    "1b": {
        "label": "1B. Inserimento Attore",
        "description": "Inserisce l'attore nello sfondo. Se lasci i campi file vuoti, prova a usare l'ultimo output disponibile.",
        "fields": [
            {"type": "image", "key": "image", "label": "Immagine sfondo/base (opzionale)"},
            {"type": "image", "key": "reference_image", "label": "Immagine attore/riferimento (opzionale)"},
            {"type": "textbox", "key": "prompt", "label": "Prompt inserimento attore", "props": {"lines": 3}},
            {"type": "textbox", "key": "negative_prompt", "label": "Negative prompt", "props": {"lines": 2}},
            {"type": "slider", "key": "strength", "label": "Strength / blending", "value": 0.75, "minimum": 0.0, "maximum": 1.0, "step": 0.01},
            {"type": "number", "key": "seed", "label": "Seed (-1 = random)", "value": -1},
            {"type": "textbox", "key": "extra_json", "label": "Extra kwargs JSON", "props": {"lines": 2}},
        ],
    },
    "1c": {
        "label": "1C. Controllo Posa/Mani",
        "description": "Controllo posa/mani. Puoi caricare un keyframe o un'immagine esterna da Nano Banana Pro.",
        "fields": [
            {"type": "image", "key": "image", "label": "Keyframe / immagine base (anche da Nano Banana Pro)"},
            {"type": "image", "key": "pose_image", "label": "Immagine posa/mani (anche da Nano Banana Pro)"},
            {"type": "textbox", "key": "prompt", "label": "Prompt posa/mani", "props": {"lines": 3}},
            {"type": "textbox", "key": "negative_prompt", "label": "Negative prompt", "props": {"lines": 2}},
            {"type": "slider", "key": "strength", "label": "Control strength", "value": 0.80, "minimum": 0.0, "maximum": 1.0, "step": 0.01},
            {"type": "number", "key": "seed", "label": "Seed (-1 = random)", "value": -1},
            {"type": "textbox", "key": "extra_json", "label": "Extra kwargs JSON", "props": {"lines": 2}},
        ],
    },
    "2a": {
        "label": "2A. Animazione Video",
        "description": "Anima un keyframe. Puoi caricare un'immagine esterna da Nano Banana Pro.",
        "fields": [
            {"type": "image", "key": "image", "label": "Immagine da animare (anche da Nano Banana Pro)"},
            {"type": "textbox", "key": "prompt", "label": "Prompt animazione", "props": {"lines": 3}},
            {"type": "textbox", "key": "negative_prompt", "label": "Negative prompt", "props": {"lines": 2}},
            {"type": "slider", "key": "strength", "label": "Motion strength", "value": 0.70, "minimum": 0.0, "maximum": 1.0, "step": 0.01},
            {"type": "number", "key": "duration", "label": "Durata stimata (s)", "value": 3},
            {"type": "number", "key": "fps", "label": "FPS base", "value": 24},
            {"type": "number", "key": "seed", "label": "Seed (-1 = random)", "value": -1},
            {"type": "textbox", "key": "extra_json", "label": "Extra kwargs JSON", "props": {"lines": 2}},
        ],
    },
    "2b": {
        "label": "2B. Continuità Clip",
        "description": "Migliora/estende la continuità tra clip. Se lasci vuoto, usa l'ultima clip disponibile.",
        "fields": [
            {"type": "video", "key": "video", "label": "Clip precedente (opzionale)"},
            {"type": "video", "key": "reference_video", "label": "Video riferimento (opzionale)"},
            {"type": "textbox", "key": "prompt", "label": "Prompt continuità", "props": {"lines": 3}},
            {"type": "textbox", "key": "negative_prompt", "label": "Negative prompt", "props": {"lines": 2}},
            {"type": "slider", "key": "strength", "label": "Strength", "value": 0.65, "minimum": 0.0, "maximum": 1.0, "step": 0.01},
            {"type": "number", "key": "seed", "label": "Seed (-1 = random)", "value": -1},
            {"type": "textbox", "key": "extra_json", "label": "Extra kwargs JSON", "props": {"lines": 2}},
        ],
    },
    "3a": {
        "label": "3A. Voce Italiana (TTS)",
        "description": "Genera la voce italiana. Se il testo non è specificato, può usare il prompt globale.",
        "fields": [
            {"type": "textbox", "key": "text", "label": "Testo da trasformare in voce italiana", "props": {"lines": 4}},
            {"type": "textbox", "key": "voice", "label": "Voce (nome/ID)", "value": "default"},
            {"type": "textbox", "key": "language", "label": "Lingua", "value": "it-IT"},
            {"type": "number", "key": "speed", "label": "Speed", "value": 1.0},
            {"type": "textbox", "key": "extra_json", "label": "Extra kwargs JSON", "props": {"lines": 2}},
        ],
    },
    "3b": {
        "label": "3B. Labiale (Lip-Sync)",
        "description": "Applica lip-sync su video + audio. Se lasci vuoto, usa ultima clip e ultimo audio disponibili.",
        "fields": [
            {"type": "video", "key": "video", "label": "Video con volto (opzionale)"},
            {"type": "audio", "key": "audio", "label": "Audio voce (opzionale)"},
            {"type": "textbox", "key": "prompt", "label": "Note espressione/labiale (opzionale)", "props": {"lines": 2}},
            {"type": "textbox", "key": "extra_json", "label": "Extra kwargs JSON", "props": {"lines": 2}},
        ],
    },
    "4a": {
        "label": "4A. Fluidità 24/30 fps",
        "description": "Interpolazione/frame rate. Se lasci vuoto, usa l'ultima clip disponibile.",
        "fields": [
            {"type": "video", "key": "video", "label": "Video da fluidificare (opzionale)"},
            {"type": "dropdown", "key": "target_fps", "label": "Target FPS", "choices": [24, 30], "value": 30},
            {"type": "textbox", "key": "extra_json", "label": "Extra kwargs JSON", "props": {"lines": 2}},
        ],
    },
    "4b": {
        "label": "4B. Upscale 1080p",
        "description": "Upscale del video. Se lasci vuoto, usa l'ultima clip disponibile.",
        "fields": [
            {"type": "video", "key": "video", "label": "Video da upscalare (opzionale)"},
            {"type": "dropdown", "key": "target_resolution", "label": "Risoluzione target", "choices": ["1920x1080", "1280x720", "3840x2160"], "value": "1920x1080"},
            {"type": "textbox", "key": "extra_json", "label": "Extra kwargs JSON", "props": {"lines": 2}},
        ],
    },
    "5": {
        "label": "5. Montaggio & Sound Design",
        "description": "Montaggio finale e sound design. Puoi passare clip multiple, video principale, voce e musica.",
        "fields": [
            {"type": "files", "key": "clips", "label": "Clip/video da montare (multipli, opzionale)"},
            {"type": "video", "key": "video", "label": "Video principale (opzionale)"},
            {"type": "audio", "key": "audio", "label": "Voce/audio (opzionale)"},
            {"type": "audio", "key": "music", "label": "Musica/SFX (opzionale)"},
            {"type": "textbox", "key": "prompt", "label": "Note montaggio / sound design", "props": {"lines": 3}},
            {"type": "textbox", "key": "extra_json", "label": "Extra kwargs JSON", "props": {"lines": 2}},
        ],
    },
}


def render_field(spec: Dict[str, Any]):
    t = spec["type"]
    label = spec.get("label", spec.get("key", ""))
    value = spec.get("value")
    props = dict(spec.get("props", {}))

    if t == "textbox" or t == "text":
        return gr.Textbox(label=label, value=value, **props)

    if t == "number":
        return gr.Number(label=label, value=value, **props)

    if t == "slider":
        minimum = props.pop("minimum", 0)
        maximum = props.pop("maximum", 100)
        step = props.pop("step", 1)
        return gr.Slider(label=label, value=value, minimum=minimum, maximum=maximum, step=step, **props)

    if t == "image":
        return gr.Image(label=label, value=value, type="filepath", **props)

    if t == "video":
        return gr.Video(label=label, value=value, **props)

    if t == "audio":
        return gr.Audio(label=label, value=value, type="filepath", **props)

    if t == "file":
        return gr.File(label=label, value=value, **props)

    if t == "files":
        file_count = props.pop("file_count", "multiple")
        return gr.File(label=label, value=value, file_count=file_count, **props)

    if t == "dropdown":
        return gr.Dropdown(label=label, value=value, choices=spec.get("choices", []), **props)

    if t == "checkbox":
        return gr.Checkbox(label=label, value=value, **props)

    raise ValueError(f"Tipo componente non supportato: {t}")


def make_output_components():
    status = gr.Markdown("### ⚙️ Pronto.")
    files = gr.File(label="File generati", file_count="multiple", interactive=False)
    with gr.Row():
        image = gr.Image(label="Immagine", type="filepath", interactive=False)
        video = gr.Video(label="Video", interactive=False)
        audio = gr.Audio(label="Audio", type="filepath", interactive=False)
    log = gr.JSON(label="Log / risultato grezzo")
    return [status, files, image, video, audio, log]


def make_runner(key: str, keys: List[str]):
    label = MODULE_DEFS[key]["label"]

    def runner(*args, progress=gr.Progress()):
        values = prepare_values(keys, list(args))
        update_progress(progress, 0.01, f"Avvio {label}")
        res = execute_module(key, values, progress)
        return result_to_ui(res)

    runner.__name__ = f"run_{key.replace('.', '_')}"
    return runner


# ---------------------------------------------------------------------------
# Cascata
# ---------------------------------------------------------------------------
def run_all(
    prompt: str,
    negative_prompt: str,
    text: str,
    voice: str,
    seed: Any,
    image: Optional[str],
    video: Optional[str],
    audio: Optional[str],
    start_step: str,
    end_step: str,
    extra_json: str,
    progress=gr.Progress(),
):
    keys = [
        "prompt",
        "negative_prompt",
        "text",
        "voice",
        "seed",
        "image",
        "video",
        "audio",
        "start_step",
        "end_step",
        "extra_json",
    ]
    args = [prompt, negative_prompt, text, voice, seed, image, video, audio, start_step, end_step, extra_json]
    values = prepare_values(keys, args)

    start = values.get("start_step") or "1a"
    end = values.get("end_step") or "5"

    if start not in STEP_ORDER or end not in STEP_ORDER:
        res = make_result(False, "all", "Selezione modulo iniziale/finale non valida.")
        yield result_to_ui(res)
        return

    start_idx = STEP_ORDER.index(start)
    end_idx = STEP_ORDER.index(end)
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    steps = STEP_ORDER[start_idx:end_idx + 1]
    total = len(steps)

    base_values = {k: v for k, v in values.items() if k not in {"start_step", "end_step"}}

    user_image = clean_value(values.get("image"))
    user_video = clean_value(values.get("video"))
    user_audio = clean_value(values.get("audio"))

    current_image = user_image
    current_video = user_video
    current_audio = user_audio

    generated_image = None
    generated_video = None
    generated_audio = None

    all_files = []
    completed = []

    for idx, step in enumerate(steps, 1):
        label = MODULE_DEFS[step]["label"]
        update_progress(progress, (idx - 1) / max(total, 1), f"Inizio {label}")

        running_status = (
            f"### ▶️ Step {idx}/{total}: {label}\n\n"
            f"**Stato:** in corso...\n\n"
            f"**Aggiornato:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        yield (
            running_status,
            all_files or None,
            current_image,
            current_video,
            current_audio,
            json_safe({"step": step, "status": "running"}),
        )

        relevant_keys = [f["key"] for f in TAB_SPECS[step]["fields"]]
        step_values = {k: v for k, v in base_values.items() if k in relevant_keys}

        if isinstance(values.get("_extra"), dict):
            step_values.update(values["_extra"])
            step_values["_extra"] = values["_extra"]

        # Continuità pipeline: preferisce gli artefatti generati nello step precedente
        if "image" in relevant_keys:
            if generated_image:
                step_values["image"] = generated_image
            elif current_image and not step_values.get("image"):
                step_values["image"] = current_image

        if "video" in relevant_keys:
            if generated_video:
                step_values["video"] = generated_video
            elif current_video and not step_values.get("video"):
                step_values["video"] = current_video

        if "audio" in relevant_keys:
            if generated_audio:
                step_values["audio"] = generated_audio
            elif current_audio and not step_values.get("audio"):
                step_values["audio"] = current_audio

        if step == "2b" and generated_video and not step_values.get("reference_video"):
            step_values["reference_video"] = generated_video

        if step == "3a":
            if not clean_value(step_values.get("text")) and clean_value(step_values.get("prompt")):
                step_values["text"] = step_values["prompt"]
            if clean_value(voice) and not clean_value(step_values.get("voice")):
                step_values["voice"] = voice

        if step == "5" and generated_video and not step_values.get("clips"):
            step_values["clips"] = [generated_video]

        res = execute_module(step, step_values, progress)

        if res["ok"]:
            completed.append(step)
            res_files = res.get("files") or []
            for f in res_files:
                if f not in all_files:
                    all_files.append(f)

            if res.get("image"):
                generated_image = res["image"]
                current_image = res["image"]
            if res.get("video"):
                generated_video = res["video"]
                current_video = res["video"]
            if res.get("audio"):
                generated_audio = res["audio"]
                current_audio = res["audio"]

            if not any([res.get("image"), res.get("video"), res.get("audio")]) and res_files:
                _, img2, vid2, aud2 = classify_paths(res_files)
                generated_image = generated_image or img2
                generated_video = generated_video or vid2
                generated_audio = generated_audio or aud2
                current_image = current_image or img2
                current_video = current_video or vid2
                current_audio = current_audio or aud2

            yield result_to_ui(res)
        else:
            yield result_to_ui(res)
            return

    final_files, final_img, final_vid, final_aud = classify_paths(all_files)
    final_image = current_image or final_img
    final_video = current_video or final_vid
    final_audio = current_audio or final_aud

    final_res = make_result(
        True,
        "all",
        f"Pipeline completata. Moduli eseguiti: {', '.join(completed) if completed else 'nessuno'}.",
        final_files,
        final_image,
        final_video,
        final_audio,
        {"steps": completed, "start": start, "end": end},
    )
    yield result_to_ui(final_res)


# ---------------------------------------------------------------------------
# Gradio Blocks
# ---------------------------------------------------------------------------
with gr.Blocks(title="Pipeline Video AI - 9 Micro-Moduli") as demo:
    gr.Markdown(
        "# Pipeline Video AI - 9 Micro-Moduli\n"
        "Orchestrazione centrale su Kaggle (GPU T4 16GB). Ogni modulo è eseguibile singolarmente oppure in cascata."
    )

    with gr.Tabs():
        for key in STEP_ORDER:
            spec = TAB_SPECS[key]
            with gr.Tab(spec["label"]):
                gr.Markdown(spec.get("description", ""))

                with gr.Row():
                    with gr.Column(scale=1):
                        components = []
                        keys = []

                        for field in spec["fields"]:
                            comp = render_field(field)
                            components.append(comp)
                            keys.append(field["key"])

                        run_btn = gr.Button(
                            f"▶️ Esegui {spec['label']}",
                            variant="primary",
                            elem_id=f"run_{key}",
                        )

                    with gr.Column(scale=1):
                        outputs = make_output_components()

                runner = make_runner(key, keys)
                run_btn.click(
                    fn=runner,
                    inputs=components,
                    outputs=outputs,
                    api_name=f"run_{key}",
                )

        with gr.Tab("Esegui Tutto in Cascata"):
            gr.Markdown(
                "Lancia i moduli uno dopo l'altro, mostrando lo stato di avanzamento. "
                "Se un campo media è vuoto, la pipeline prova a usare l'ultimo artefatto generato."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    all_prompt = gr.Textbox(label="Prompt globale", lines=3)
                    all_negative = gr.Textbox(label="Negative prompt globale", lines=2)
                    all_text = gr.Textbox(label="Testo voce italiana (TTS)", lines=3)
                    all_voice = gr.Textbox(label="Voce (nome/ID)", value="default")
                    all_seed = gr.Number(label="Seed globale (-1 = random)", value=-1)

                    all_image = gr.Image(
                        label="Immagine iniziale / override (anche da Nano Banana Pro)",
                        type="filepath",
                    )
                    all_video = gr.Video(label="Video iniziale / override", type="filepath")
                    all_audio = gr.Audio(label="Audio iniziale / override", type="filepath")

                    step_choices = [(MODULE_DEFS[k]["label"], k) for k in STEP_ORDER]
                    start_dd = gr.Dropdown(label="Modulo iniziale", choices=step_choices, value="1a")
                    end_dd = gr.Dropdown(label="Modulo finale", choices=step_choices, value="5")

                    all_extra = gr.Textbox(
                        label="Extra kwargs JSON (applicato a tutti i moduli)",
                        lines=2,
                        placeholder='{"model": "..."}',
                    )

                    all_btn = gr.Button(
                        "🚀 Esegui Tutto in Cascata",
                        variant="primary",
                        elem_id="run_all",
                    )

                with gr.Column(scale=1):
                    all_outputs = make_output_components()

            all_inputs = [
                all_prompt,
                all_negative,
                all_text,
                all_voice,
                all_seed,
                all_image,
                all_video,
                all_audio,
                start_dd,
                end_dd,
                all_extra,
            ]

            all_btn.click(
                fn=run_all,
                inputs=all_inputs,
                outputs=all_outputs,
                api_name="run_all",
            )

demo.queue()
demo.launch(share=True, debug=True)
