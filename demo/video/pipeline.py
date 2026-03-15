"""VideoCompositor — scene sequencing + PyAV H.264/AAC encoding."""

import os
import sys
import av
import numpy as np

from config import (
    WIDTH, HEIGHT, FPS, TOTAL_FRAMES, CRF, OUTPUT_PATH,
    SCENES, CROSSFADE_FRAMES, AUDIO_SAMPLE_RATE,
)
from scenes.scene1_radio import Scene1Radio
from scenes.scene2_recon import Scene2Recon
from scenes.scene3_loss import Scene3Loss
from scenes.scene4_adversarial import Scene4Adversarial
from scenes.scene5_scale import Scene5Scale
from renderer.audio import AudioGenerator


class VideoCompositor:
    """Render all scenes frame-by-frame into an H.264 MP4."""

    def __init__(self, output_path: str = OUTPUT_PATH):
        self.output_path = output_path
        self.scenes = [
            Scene1Radio(),
            Scene2Recon(),
            Scene3Loss(),
            Scene4Adversarial(),
            Scene5Scale(),
        ]
        # Scene frame ranges as list for index access
        self._ranges = list(SCENES.values())
        self.container = None
        self.v_stream = None
        self.a_stream = None

    def _scene_for_frame(self, frame: int):
        """Return (scene_index, local_frame, total_frames)."""
        for i, (start, end) in enumerate(self._ranges):
            if start <= frame < end:
                return i, frame - start, end - start
        # Past end — clamp to last scene
        i = len(self._ranges) - 1
        s, e = self._ranges[i]
        return i, e - s - 1, e - s

    def render_all(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # ── Init encoder ─────────────────────────
        self.container = av.open(self.output_path, 'w')
        self.v_stream = self.container.add_stream('h264', rate=FPS)
        self.v_stream.width = WIDTH
        self.v_stream.height = HEIGHT
        self.v_stream.pix_fmt = 'yuv420p'
        self.v_stream.options = {'crf': str(CRF), 'preset': 'medium'}

        self.a_stream = self.container.add_stream('aac', rate=AUDIO_SAMPLE_RATE)
        self.a_stream.layout = 'stereo'

        # ── Setup scenes ─────────────────────────
        for sc in self.scenes:
            sc.setup()

        # ── Render video frames ──────────────────
        for frame_idx in range(TOTAL_FRAMES):
            rgb = self._render_composited_frame(frame_idx)

            vf = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            vf.pts = frame_idx
            for pkt in self.v_stream.encode(vf):
                self.container.mux(pkt)

            if frame_idx % 300 == 0:
                pct = frame_idx / TOTAL_FRAMES * 100
                print(f"\rVideo: {pct:5.1f}% ({frame_idx}/{TOTAL_FRAMES})",
                      end="", flush=True)

        print("\rVideo: 100.0% — encoding audio...", end="", flush=True)

        # ── Render & encode audio ────────────────
        audio_gen = AudioGenerator()
        audio = audio_gen.generate_full_audio(TOTAL_FRAMES)
        self._encode_audio(audio)

        # ── Flush & close ────────────────────────
        for pkt in self.v_stream.encode():
            self.container.mux(pkt)
        for pkt in self.a_stream.encode():
            self.container.mux(pkt)
        self.container.close()

        file_mb = os.path.getsize(self.output_path) / 1_048_576
        print(f"\rDone: {self.output_path} ({file_mb:.1f} MB)")

    def _render_composited_frame(self, frame_idx: int) -> np.ndarray:
        """Render frame with scene blending and fades."""
        si, local, total = self._scene_for_frame(frame_idx)
        rgb = self.scenes[si].render_frame(local, total, frame_idx).astype(np.float32)

        # ── Crossfade from previous scene ────────
        if si > 0 and local < CROSSFADE_FRAMES:
            prev_start, prev_end = self._ranges[si - 1]
            prev_total = prev_end - prev_start
            prev_local = prev_total - CROSSFADE_FRAMES + local
            prev_rgb = self.scenes[si - 1].render_frame(
                prev_local, prev_total, frame_idx
            ).astype(np.float32)
            alpha = local / CROSSFADE_FRAMES
            rgb = prev_rgb * (1 - alpha) + rgb * alpha

        # ── Fade in (first 3 seconds) ────────────
        if frame_idx < 90:
            rgb *= frame_idx / 90.0

        # ── Fade out (last 4 seconds) ────────────
        if frame_idx > TOTAL_FRAMES - 120:
            remaining = TOTAL_FRAMES - frame_idx
            rgb *= remaining / 120.0

        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _encode_audio(self, audio: np.ndarray):
        """Encode stereo float32 audio (n_samples, 2) into AAC."""
        # PyAV expects (channels, samples) for 'fltp'
        samples_per_chunk = 1024  # AAC frame size
        n_samples = len(audio)
        pts = 0

        for i in range(0, n_samples, samples_per_chunk):
            chunk = audio[i:i + samples_per_chunk]
            if len(chunk) < samples_per_chunk:
                pad = np.zeros((samples_per_chunk - len(chunk), 2),
                               dtype=np.float32)
                chunk = np.concatenate([chunk, pad])

            af = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(chunk.T, dtype=np.float32),
                format='fltp',
                layout='stereo',
            )
            af.sample_rate = AUDIO_SAMPLE_RATE
            af.pts = pts
            pts += samples_per_chunk

            for pkt in self.a_stream.encode(af):
                self.container.mux(pkt)
