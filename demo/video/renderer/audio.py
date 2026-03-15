"""Synthetic ambient audio: drone hum, radio static, beeps, alerts, pads."""

import numpy as np
from scipy.signal import butter, lfilter

from config import AUDIO_SAMPLE_RATE as SR, SCENES, FPS


def _bandpass(data: np.ndarray, low: float, high: float,
              order: int = 4) -> np.ndarray:
    nyq = SR / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, data)


def drone_hum(n_samples: int, amplitude: float = 0.12) -> np.ndarray:
    """Low-frequency drone hum: 80-120 Hz + harmonics."""
    t = np.arange(n_samples) / SR
    sig = (
        np.sin(2 * np.pi * 80 * t) * 0.40 +
        np.sin(2 * np.pi * 120 * t) * 0.30 +
        np.sin(2 * np.pi * 160 * t) * 0.15 +
        np.sin(2 * np.pi * 240 * t) * 0.08 +
        np.random.randn(n_samples) * 0.04
    )
    # Slow amplitude LFO
    lfo = 0.8 + 0.2 * np.sin(2 * np.pi * 0.15 * t)
    return sig * amplitude * lfo


def radio_static(n_samples: int, amplitude: float = 0.08) -> np.ndarray:
    """Band-pass filtered white noise (300-3000 Hz)."""
    noise = np.random.randn(n_samples)
    return _bandpass(noise, 300, 3000) * amplitude


def radio_beep(n_samples: int, freq: float = 1000,
               duration_ms: float = 50) -> np.ndarray:
    """Short sine pulse."""
    out = np.zeros(n_samples)
    dur_samples = int(SR * duration_ms / 1000)
    dur_samples = min(dur_samples, n_samples)
    t = np.arange(dur_samples) / SR
    # Smooth envelope
    env = np.sin(np.pi * np.arange(dur_samples) / dur_samples)
    out[:dur_samples] = np.sin(2 * np.pi * freq * t) * env * 0.3
    return out


def alert_tone(n_samples: int, f_start: float = 800,
               f_end: float = 1200) -> np.ndarray:
    """Frequency sweep alert (200ms)."""
    dur = min(int(SR * 0.2), n_samples)
    out = np.zeros(n_samples)
    t = np.arange(dur) / SR
    freq = f_start + (f_end - f_start) * t / (dur / SR)
    env = np.sin(np.pi * np.arange(dur) / dur)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    out[:dur] = np.sin(phase) * env * 0.25
    return out


def sonar_ping(n_samples: int, freq: float = 2000) -> np.ndarray:
    """Damped sine sonar ping."""
    dur = min(int(SR * 0.3), n_samples)
    out = np.zeros(n_samples)
    t = np.arange(dur) / SR
    env = np.exp(-t * 12)
    out[:dur] = np.sin(2 * np.pi * freq * t) * env * 0.2
    return out


def ui_click(n_samples: int) -> np.ndarray:
    """Short noise burst (10ms)."""
    dur = min(int(SR * 0.01), n_samples)
    out = np.zeros(n_samples)
    t_env = np.arange(dur) / dur
    env = np.exp(-t_env * 8)
    out[:dur] = np.random.randn(dur) * env * 0.15
    return out


def ambient_pad(n_samples: int, amplitude: float = 0.06) -> np.ndarray:
    """Layered low sine waves with slow LFO for atmosphere."""
    t = np.arange(n_samples) / SR
    sig = (
        np.sin(2 * np.pi * 40 * t) * 0.35 +
        np.sin(2 * np.pi * 55 * t) * 0.30 +
        np.sin(2 * np.pi * 70 * t) * 0.20 +
        np.sin(2 * np.pi * 30 * t) * 0.15
    )
    # Slow breathing LFO
    lfo = 0.6 + 0.4 * np.sin(2 * np.pi * 0.08 * t)
    return sig * amplitude * lfo


class AudioGenerator:
    """Generate the full audio track synchronized to scenes."""

    def _add(self, mix: np.ndarray, pos: int, signal: np.ndarray):
        """Safely add signal into mix at position, clipping to bounds."""
        n = len(mix)
        if pos < 0 or pos >= n:
            return
        end = min(pos + len(signal), n)
        length = end - pos
        if length > 0:
            mix[pos:end] += signal[:length]

    def generate_full_audio(self, total_frames: int) -> np.ndarray:
        """Return stereo audio array (n_samples, 2) as float32."""
        total_samples = int(total_frames / FPS * SR)
        mix = np.zeros(total_samples, dtype=np.float64)

        # Continuous layers
        mix += drone_hum(total_samples)
        mix += ambient_pad(total_samples)

        # Scene-specific sounds
        for scene_num, (f_start, f_end) in SCENES.items():
            s_start = int(f_start / FPS * SR)
            if s_start >= total_samples:
                continue

            if scene_num == 1:
                # Radio static during transmission (frames 90–510)
                r_start = s_start + int(90 / FPS * SR)
                r_len = int((510 - 90) / FPS * SR)
                self._add(mix, r_start, radio_static(r_len, 0.06))
                # Beeps at transmission boundaries
                self._add(mix, s_start + int(90 / FPS * SR),
                          radio_beep(int(SR * 0.1)))
                self._add(mix, s_start + int(360 / FPS * SR),
                          radio_beep(int(SR * 0.1)))

            elif scene_num == 3:
                # Alert at drone loss (frame 2880)
                loss_pos = s_start + int((2880 - f_start) / FPS * SR)
                self._add(mix, loss_pos, alert_tone(int(SR * 0.3)))
                # UI clicks during auction (frames 3060–3240)
                for f in range(3060, 3240, 12):
                    click_pos = s_start + int((f - f_start) / FPS * SR)
                    self._add(mix, click_pos, ui_click(int(SR * 0.02)))

            elif scene_num == 4:
                # Sonar ping at enemy detection (frame 4500)
                det_pos = s_start + int((4500 - f_start) / FPS * SR)
                self._add(mix, det_pos, sonar_ping(int(SR * 0.4)))

            elif scene_num == 5:
                # UI clicks at slider stops
                for f in [6480, 6660, 6840]:
                    click_pos = s_start + int((f - f_start) / FPS * SR)
                    self._add(mix, click_pos, ui_click(int(SR * 0.02)))

        # Scene intensity envelope (swell at transitions)
        fade_samples = int(SR * 0.5)
        for scene_num, (f_start, f_end) in SCENES.items():
            s_start = int(f_start / FPS * SR)
            s_end = min(int(f_end / FPS * SR), total_samples)
            n = s_end - s_start
            if s_start >= total_samples or n <= fade_samples * 2:
                continue
            fade_in = np.linspace(0.7, 1.0, min(fade_samples, n))
            fade_out = np.linspace(1.0, 0.7, min(fade_samples, n))
            mix[s_start:s_start + len(fade_in)] *= fade_in
            mix[s_end - len(fade_out):s_end] *= fade_out

        # Soft clip & normalize
        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = mix / max(peak, 0.5) * 0.7
        mix = np.tanh(mix)  # soft clip

        # Stereo (slight decorrelation for width)
        stereo = np.column_stack([
            mix,
            np.roll(mix, int(SR * 0.0003))  # ~0.3ms delay for width
        ])
        return stereo.astype(np.float32)
