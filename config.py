import os
import gc
import torch

OUTPUT_DIR = "output_pipeline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATHS = {
    "1A_SFONDO": os.path.join(OUTPUT_DIR, "step_1a_sfondo.png"),
    "1B_KEYFRAME": os.path.join(OUTPUT_DIR, "step_1b_keyframe.png"),
    "1C_FIXED": os.path.join(OUTPUT_DIR, "step_1c_fixed.png"),
    "2A_VIDEO": os.path.join(OUTPUT_DIR, "step_2a_video_raw.mp4"),
    "2B_EXTENDED": os.path.join(OUTPUT_DIR, "step_2b_video_ext.mp4"),
    "3A_AUDIO": os.path.join(OUTPUT_DIR, "step_3a_voce.wav"),
    "3B_LIPSYNC": os.path.join(OUTPUT_DIR, "step_3b_lipsync.mp4"),
    "4A_SMOOTH": os.path.join(OUTPUT_DIR, "step_4a_smooth.mp4"),
    "4B_FINAL": os.path.join(OUTPUT_DIR, "step_4b_final_1080p.mp4")
}

def pulisci_memoria(model=None):
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memoria pulita e cache GPU svuotata.")
