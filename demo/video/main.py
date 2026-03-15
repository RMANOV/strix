#!/usr/bin/env python3
"""STRIX Demo Video — Entry point.

Renders a 4-minute photorealistic military demo video:
  1920×1080, 30fps, H.264 CRF 18, AAC audio.

Usage:
    python main.py              # Full render (7200 frames)
    python main.py --scene 1    # Render only Scene 1
    python main.py --preview 3  # Render 3 keyframes as PNGs
"""

import sys
import os
import argparse

# Ensure package imports work from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_PATH, SCENES, FPS
from pipeline import VideoCompositor


def render_preview(scene_num: int, n_frames: int = 3):
    """Render a few keyframes from a scene as PNG files for inspection."""
    from PIL import Image

    start, end = SCENES[scene_num]
    total = end - start

    # Import scene class
    scene_classes = {
        1: "scenes.scene1_radio.Scene1Radio",
        2: "scenes.scene2_recon.Scene2Recon",
        3: "scenes.scene3_loss.Scene3Loss",
        4: "scenes.scene4_adversarial.Scene4Adversarial",
        5: "scenes.scene5_scale.Scene5Scale",
    }
    module_path, class_name = scene_classes[scene_num].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    scene = getattr(mod, class_name)()
    scene.setup()

    out_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(out_dir, exist_ok=True)

    # Keyframe positions
    positions = [int(total * i / max(n_frames - 1, 1))
                 for i in range(n_frames)]

    for i, local_f in enumerate(positions):
        global_f = start + local_f
        rgb = scene.render_frame(local_f, total, global_f)
        img = Image.fromarray(rgb)
        path = os.path.join(out_dir, f"preview_s{scene_num}_{i}.png")
        img.save(path)
        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="STRIX Demo Video Renderer")
    parser.add_argument("--scene", type=int, choices=[1, 2, 3, 4, 5],
                        help="Render only this scene")
    parser.add_argument("--preview", type=int, metavar="N",
                        help="Render N keyframe PNGs instead of video")
    args = parser.parse_args()

    if args.preview:
        scenes_to_preview = [args.scene] if args.scene else list(SCENES.keys())
        for s in scenes_to_preview:
            print(f"Scene {s} preview ({args.preview} frames):")
            render_preview(s, args.preview)
        return

    print("STRIX Demo Video Renderer")
    print(f"  Output:     {OUTPUT_PATH}")
    print(f"  Resolution: 1920×1080 @ {FPS}fps")
    print(f"  Duration:   {len(SCENES)} scenes, 4:00")
    print()

    compositor = VideoCompositor()
    compositor.render_all()


if __name__ == "__main__":
    main()
