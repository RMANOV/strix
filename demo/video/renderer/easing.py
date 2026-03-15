"""Easing functions for smooth animation curves."""

import math


def linear(t: float) -> float:
    return t


def ease_in_quad(t: float) -> float:
    return t * t


def ease_out_quad(t: float) -> float:
    return t * (2 - t)


def ease_in_out_quad(t: float) -> float:
    if t < 0.5:
        return 2 * t * t
    return -1 + (4 - 2 * t) * t


def ease_in_cubic(t: float) -> float:
    return t * t * t


def ease_out_cubic(t: float) -> float:
    t -= 1
    return t * t * t + 1


def ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4 * t * t * t
    t -= 1
    return 1 + 4 * t * t * t


def ease_out_expo(t: float) -> float:
    if t >= 1.0:
        return 1.0
    return 1 - 2 ** (-10 * t)


def ease_in_expo(t: float) -> float:
    if t <= 0.0:
        return 0.0
    return 2 ** (10 * (t - 1))


def ease_out_elastic(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    p = 0.3
    s = p / 4
    return 2 ** (-10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1


def ease_out_back(t: float) -> float:
    s = 1.70158
    t -= 1
    return t * t * ((s + 1) * t + s) + 1


def pulse(t: float, center: float = 0.5, width: float = 0.3) -> float:
    """Smooth pulse: 0→1→0, peaks at `center`."""
    d = abs(t - center) / width
    return max(0.0, 1.0 - d * d)


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def remap(v: float, in_lo: float, in_hi: float,
          out_lo: float = 0.0, out_hi: float = 1.0) -> float:
    """Map value from [in_lo, in_hi] to [out_lo, out_hi], clamped."""
    t = clamp((v - in_lo) / (in_hi - in_lo))
    return lerp(out_lo, out_hi, t)
