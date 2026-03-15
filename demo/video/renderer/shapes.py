"""Shape primitives: drone icon, explosion, waveform, logo, slider."""

import cairo
import math
import numpy as np

from config import (CYAN, GREEN, RED, WHITE, DIM_WHITE, AMBER, DRONE_ICON_SIZE,
                    FONT_FAMILY, FONT_SIZE_SMALL, FONT_SIZE_TINY, WIDTH, HEIGHT)


def drone_icon(ctx: cairo.Context, cx: float, cy: float,
               heading: float = 0.0, color=CYAN, size: float = DRONE_ICON_SIZE):
    """Oriented triangle representing a drone (points in heading direction)."""
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(heading)

    # Triangle pointing right (heading=0 → east)
    s = size
    ctx.move_to(s, 0)
    ctx.line_to(-s * 0.6, -s * 0.5)
    ctx.line_to(-s * 0.6, s * 0.5)
    ctx.close_path()
    ctx.set_source_rgba(*color)
    ctx.fill_preserve()
    ctx.set_source_rgba(*color[:3], 0.6)
    ctx.set_line_width(1)
    ctx.stroke()
    ctx.restore()


def drone_glow(ctx: cairo.Context, cx: float, cy: float,
               color=CYAN, radius: float = 20):
    """Soft radial glow behind a drone."""
    pat = cairo.RadialGradient(cx, cy, 0, cx, cy, radius)
    pat.add_color_stop_rgba(0, *color[:3], 0.25)
    pat.add_color_stop_rgba(1, *color[:3], 0.0)
    ctx.set_source(pat)
    ctx.arc(cx, cy, radius, 0, 2 * math.pi)
    ctx.fill()


def explosion_ring(ctx: cairo.Context, cx: float, cy: float,
                   progress: float, max_radius: float = 80,
                   color=RED):
    """Expanding ring explosion effect. progress: 0→1."""
    r = max_radius * progress
    alpha = max(0.0, 1.0 - progress)

    # Outer ring
    ctx.set_source_rgba(*color[:3], alpha * 0.7)
    ctx.set_line_width(3 * (1 - progress) + 1)
    ctx.arc(cx, cy, r, 0, 2 * math.pi)
    ctx.stroke()

    # Inner filled glow
    pat = cairo.RadialGradient(cx, cy, 0, cx, cy, r * 0.6)
    pat.add_color_stop_rgba(0, 1, 1, 1, alpha * 0.5)
    pat.add_color_stop_rgba(0.5, *color[:3], alpha * 0.3)
    pat.add_color_stop_rgba(1, *color[:3], 0)
    ctx.set_source(pat)
    ctx.arc(cx, cy, r * 0.6, 0, 2 * math.pi)
    ctx.fill()


def screen_flash(ctx: cairo.Context, alpha: float, color=WHITE):
    """Full-screen white/color flash overlay."""
    ctx.set_source_rgba(*color[:3], alpha)
    ctx.rectangle(0, 0, WIDTH, HEIGHT)
    ctx.fill()


def radio_waveform(ctx: cairo.Context, x: float, y: float,
                   w: float, h: float, phase: float,
                   color=CYAN, amplitude: float = 1.0):
    """Animated oscillating radio waveform display."""
    ctx.save()
    ctx.set_source_rgba(*color[:3], 0.8)
    ctx.set_line_width(2)

    n_points = int(w)
    ctx.move_to(x, y + h / 2)
    for i in range(n_points):
        t = i / n_points
        px = x + t * w
        # Composite waveform: voice-like modulated signal
        env = math.sin(math.pi * t) * amplitude
        sig = (
            math.sin(2 * math.pi * 3 * t + phase) * 0.6 +
            math.sin(2 * math.pi * 7 * t + phase * 1.3) * 0.25 +
            math.sin(2 * math.pi * 13 * t + phase * 0.7) * 0.15
        )
        py = y + h / 2 + sig * env * h * 0.4
        ctx.line_to(px, py)
    ctx.stroke()

    # Center line
    ctx.set_source_rgba(*color[:3], 0.15)
    ctx.set_line_width(1)
    ctx.move_to(x, y + h / 2)
    ctx.line_to(x + w, y + h / 2)
    ctx.stroke()
    ctx.restore()


def slider_widget(ctx: cairo.Context, x: float, y: float,
                  w: float, h: float, value: float, max_val: float,
                  color=CYAN, show_labels: bool = True):
    """Horizontal slider with tick marks and value label."""
    ctx.save()
    ratio = min(value / max_val, 1.0)

    # Track background
    ctx.set_source_rgba(*DIM_WHITE[:3], 0.15)
    ctx.rectangle(x, y, w, h)
    ctx.fill()

    # Filled portion
    ctx.set_source_rgba(*color[:3], 0.5)
    ctx.rectangle(x, y, w * ratio, h)
    ctx.fill()

    # Tick marks
    for tick_val in [5, 50, 100, 250, 500]:
        if tick_val <= max_val:
            tx = x + (tick_val / max_val) * w
            ctx.set_source_rgba(*DIM_WHITE[:3], 0.4)
            ctx.set_line_width(1)
            ctx.move_to(tx, y - 4)
            ctx.line_to(tx, y + h + 4)
            ctx.stroke()
            if show_labels:
                ctx.set_font_size(FONT_SIZE_TINY)
                ctx.move_to(tx - 8, y + h + 16)
                ctx.show_text(str(tick_val))

    # Handle
    hx = x + w * ratio
    ctx.set_source_rgba(*color)
    ctx.arc(hx, y + h / 2, h * 0.8, 0, 2 * math.pi)
    ctx.fill()

    # Value label
    if show_labels:
        ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(FONT_SIZE_SMALL)
        ctx.set_source_rgba(*color)
        ctx.move_to(hx + 12, y + h / 2 + 5)
        ctx.show_text(f"{int(value)}")
    ctx.restore()


def confidence_gauge(ctx: cairo.Context, cx: float, cy: float,
                     radius: float, value: float, color=CYAN,
                     label_text: str = ""):
    """Circular arc gauge (0–100%). value: 0.0→1.0."""
    ctx.save()
    start_angle = math.pi * 0.75
    sweep = math.pi * 1.5  # 270 degrees
    end_angle = start_angle + sweep

    # Background arc
    ctx.set_source_rgba(*DIM_WHITE[:3], 0.15)
    ctx.set_line_width(6)
    ctx.arc(cx, cy, radius, start_angle, end_angle)
    ctx.stroke()

    # Filled arc
    fill_end = start_angle + sweep * min(value, 1.0)
    ctx.set_source_rgba(*color)
    ctx.set_line_width(6)
    ctx.arc(cx, cy, radius, start_angle, fill_end)
    ctx.stroke()

    # Value text
    ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(radius * 0.7)
    ctx.set_source_rgba(*color)
    pct_text = f"{int(value * 100)}%"
    ext = ctx.text_extents(pct_text)
    ctx.move_to(cx - ext.x_advance / 2, cy + ext.height / 3)
    ctx.show_text(pct_text)

    # Label
    if label_text:
        ctx.set_font_size(radius * 0.35)
        ctx.set_source_rgba(*DIM_WHITE)
        ext = ctx.text_extents(label_text)
        ctx.move_to(cx - ext.x_advance / 2, cy + radius + 18)
        ctx.show_text(label_text)
    ctx.restore()


def bid_panel(ctx: cairo.Context, x: float, y: float, w: float, h: float,
              values: dict, progress: float, color=CYAN):
    """Slide-in auction bid panel with animated values."""
    ctx.save()
    # Panel background
    ctx.set_source_rgba(0.05, 0.05, 0.1, 0.9)
    ctx.rectangle(x, y, w, h)
    ctx.fill()

    # Border
    ctx.set_source_rgba(*color[:3], 0.6)
    ctx.set_line_width(2)
    ctx.rectangle(x, y, w, h)
    ctx.stroke()

    # Title
    ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(16)
    ctx.set_source_rgba(*color)
    ctx.move_to(x + 12, y + 25)
    ctx.show_text("TASK RE-AUCTION")

    # Formula
    ctx.set_font_size(11)
    ctx.set_source_rgba(*DIM_WHITE)
    formula = "bid = urg×10 + cap×3 + prox×5 + nrg×2 − risk×4"
    ctx.move_to(x + 12, y + 48)
    ctx.show_text(formula)

    # Animated values
    row_y = y + 72
    weights = {"urgency": 10, "capability": 3, "proximity": 5,
               "energy": 2, "risk": -4}
    for key, weight in weights.items():
        val = values.get(key, 0) * min(progress, 1.0)
        score = val * weight
        # Label
        ctx.set_font_size(13)
        ctx.set_source_rgba(*DIM_WHITE)
        ctx.move_to(x + 12, row_y)
        ctx.show_text(f"{key:>11s}: {val:5.1f} × {weight:+d} = ")
        # Score (colored)
        sc = CYAN if score >= 0 else RED
        ctx.set_source_rgba(*sc)
        ctx.show_text(f"{score:+6.1f}")
        row_y += 22

    # Total
    total = sum(values.get(k, 0) * w * min(progress, 1.0)
                for k, w in weights.items())
    ctx.set_line_width(1)
    ctx.set_source_rgba(*color[:3], 0.3)
    ctx.move_to(x + 12, row_y + 2)
    ctx.line_to(x + w - 12, row_y + 2)
    ctx.stroke()
    row_y += 20
    ctx.set_font_size(16)
    ctx.set_source_rgba(*color)
    ctx.move_to(x + 12, row_y)
    ctx.show_text(f"TOTAL BID: {total:.1f}")
    ctx.restore()


def strix_logo(ctx: cairo.Context, cx: float, cy: float,
               font_size: float = 72, color=CYAN, alpha: float = 1.0):
    """Render the STRIX logo text with glow."""
    ctx.save()
    ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(font_size)
    text = "S T R I X"
    ext = ctx.text_extents(text)
    tx = cx - ext.x_advance / 2
    ty = cy + ext.height / 3

    # Glow
    for offset in range(5, 0, -1):
        a = alpha * 0.08 * (6 - offset)
        ctx.set_source_rgba(*color[:3], a)
        for dx in range(-offset, offset + 1, max(1, offset)):
            for dy in range(-offset, offset + 1, max(1, offset)):
                ctx.move_to(tx + dx, ty + dy)
                ctx.show_text(text)

    # Crisp text
    ctx.set_source_rgba(*color[:3], alpha)
    ctx.move_to(tx, ty)
    ctx.show_text(text)

    # Subtitle
    ctx.set_font_size(font_size * 0.25)
    sub = "AUTONOMOUS  DRONE  SWARM  INTELLIGENCE"
    ext2 = ctx.text_extents(sub)
    ctx.set_source_rgba(*color[:3], alpha * 0.6)
    ctx.move_to(cx - ext2.x_advance / 2, ty + font_size * 0.6)
    ctx.show_text(sub)
    ctx.restore()


def kill_zone_marker(ctx: cairo.Context, cx: float, cy: float,
                     radius: float, pulse_phase: float, color=RED):
    """Pulsing red circle marking a kill zone."""
    ctx.save()
    pulse = 0.5 + 0.5 * math.sin(pulse_phase)

    # Pulsing fill
    pat = cairo.RadialGradient(cx, cy, 0, cx, cy, radius)
    pat.add_color_stop_rgba(0, *color[:3], 0.2 * pulse)
    pat.add_color_stop_rgba(0.7, *color[:3], 0.08 * pulse)
    pat.add_color_stop_rgba(1, *color[:3], 0)
    ctx.set_source(pat)
    ctx.arc(cx, cy, radius, 0, 2 * math.pi)
    ctx.fill()

    # Ring
    ctx.set_source_rgba(*color[:3], 0.5 + 0.3 * pulse)
    ctx.set_line_width(2)
    ctx.arc(cx, cy, radius, 0, 2 * math.pi)
    ctx.stroke()

    # Cross
    s = radius * 0.3
    ctx.set_source_rgba(*color[:3], 0.6)
    ctx.set_line_width(1.5)
    ctx.move_to(cx - s, cy - s)
    ctx.line_to(cx + s, cy + s)
    ctx.move_to(cx + s, cy - s)
    ctx.line_to(cx - s, cy + s)
    ctx.stroke()
    ctx.restore()


def vignette(ctx: cairo.Context, w: float = WIDTH, h: float = HEIGHT,
             strength: float = 0.6):
    """Dark vignette overlay for cinematic effect."""
    cx, cy = w / 2, h / 2
    r = math.sqrt(cx * cx + cy * cy)
    pat = cairo.RadialGradient(cx, cy, r * 0.4, cx, cy, r)
    pat.add_color_stop_rgba(0, 0, 0, 0, 0)
    pat.add_color_stop_rgba(1, 0, 0, 0, strength)
    ctx.set_source(pat)
    ctx.rectangle(0, 0, w, h)
    ctx.fill()
