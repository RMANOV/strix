"""HUD overlay: corners, scanlines, bars, gauges, drone status, clock."""

import cairo
import math

from config import (
    WIDTH, HEIGHT, CYAN, GREEN, RED, DIM_WHITE, AMBER, FPS,
    FONT_FAMILY, FONT_SIZE_SMALL, FONT_SIZE_TINY, FONT_SIZE_MEDIUM,
)
from renderer.text_renderer import mission_time_text, label


class HUD:
    """Head-up display overlay layer."""

    def draw_full(self, ctx: cairo.Context, frame_idx: int,
                  global_frame: int = 0):
        """Draw all standard HUD elements."""
        self.draw_corners(ctx)
        self.draw_scanlines(ctx, frame_idx)
        self.draw_mission_time(ctx, global_frame)
        self.draw_coords(ctx)

    # ── Corner brackets ──────────────────────────
    def draw_corners(self, ctx: cairo.Context, color=CYAN, alpha: float = 0.5,
                     size: int = 40, thickness: float = 2, margin: int = 20):
        ctx.save()
        ctx.set_source_rgba(*color[:3], alpha)
        ctx.set_line_width(thickness)

        corners = [
            (margin, margin, 1, 1),                          # top-left
            (WIDTH - margin, margin, -1, 1),                  # top-right
            (margin, HEIGHT - margin, 1, -1),                 # bottom-left
            (WIDTH - margin, HEIGHT - margin, -1, -1),        # bottom-right
        ]
        for cx_, cy_, dx, dy in corners:
            ctx.move_to(cx_, cy_ + dy * size)
            ctx.line_to(cx_, cy_)
            ctx.line_to(cx_ + dx * size, cy_)
            ctx.stroke()
        ctx.restore()

    # ── Scanlines (CRT aesthetic) ────────────────
    def draw_scanlines(self, ctx: cairo.Context, frame_idx: int,
                       alpha: float = 0.03, spacing: int = 3):
        ctx.save()
        ctx.set_source_rgba(0, 0, 0, alpha)
        offset = frame_idx % spacing
        y = offset
        while y < HEIGHT:
            ctx.rectangle(0, y, WIDTH, 1)
            y += spacing
        ctx.fill()
        ctx.restore()

    # ── Mission time ─────────────────────────────
    def draw_mission_time(self, ctx: cairo.Context, global_frame: int):
        seconds = global_frame / FPS
        text = mission_time_text(seconds)
        ctx.save()
        ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(FONT_SIZE_SMALL)
        ctx.set_source_rgba(*CYAN[:3], 0.7)
        ctx.move_to(WIDTH - 180, 40)
        ctx.show_text(text)
        ctx.restore()

    # ── Coordinate display ───────────────────────
    def draw_coords(self, ctx: cairo.Context):
        ctx.save()
        ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(FONT_SIZE_TINY)
        ctx.set_source_rgba(*DIM_WHITE[:3], 0.35)
        ctx.move_to(30, HEIGHT - 30)
        ctx.show_text("37°12'N  38°54'E  ALT: 120m AGL")
        ctx.restore()

    # ── Status bar ───────────────────────────────
    def draw_status_bar(self, ctx: cairo.Context, x: float, y: float,
                        w: float, h: float, value: float, max_val: float,
                        color=CYAN, label_text: str = "",
                        show_value: bool = True):
        """Horizontal fill bar with optional label."""
        ctx.save()
        ratio = min(value / max(max_val, 0.001), 1.0)

        # Background
        ctx.set_source_rgba(*DIM_WHITE[:3], 0.1)
        ctx.rectangle(x, y, w, h)
        ctx.fill()

        # Fill
        ctx.set_source_rgba(*color[:3], 0.6)
        ctx.rectangle(x, y, w * ratio, h)
        ctx.fill()

        # Border
        ctx.set_source_rgba(*color[:3], 0.3)
        ctx.set_line_width(0.5)
        ctx.rectangle(x, y, w, h)
        ctx.stroke()

        # Label
        if label_text:
            ctx.set_font_size(FONT_SIZE_TINY)
            ctx.set_source_rgba(*DIM_WHITE)
            ctx.move_to(x, y - 3)
            ctx.show_text(label_text)

        # Value
        if show_value:
            ctx.set_font_size(FONT_SIZE_TINY)
            ctx.set_source_rgba(*color)
            txt = f"{value:.0f}/{max_val:.0f}"
            ext = ctx.text_extents(txt)
            ctx.move_to(x + w + 4, y + h - 1)
            ctx.show_text(txt)
        ctx.restore()

    # ── Circular gauge ───────────────────────────
    def draw_gauge(self, ctx: cairo.Context, cx: float, cy: float,
                   radius: float, value: float, color=CYAN,
                   label_text: str = ""):
        """Circular arc gauge. value: 0→1."""
        ctx.save()
        start = math.pi * 0.75
        sweep = math.pi * 1.5

        # Background arc
        ctx.set_source_rgba(*DIM_WHITE[:3], 0.1)
        ctx.set_line_width(4)
        ctx.arc(cx, cy, radius, start, start + sweep)
        ctx.stroke()

        # Value arc
        ctx.set_source_rgba(*color)
        ctx.set_line_width(4)
        ctx.arc(cx, cy, radius, start, start + sweep * min(value, 1.0))
        ctx.stroke()

        # Center text
        ctx.set_font_size(radius * 0.6)
        ctx.set_source_rgba(*color)
        pct = f"{int(value * 100)}"
        ext = ctx.text_extents(pct)
        ctx.move_to(cx - ext.x_advance / 2, cy + ext.height / 3)
        ctx.show_text(pct)

        if label_text:
            ctx.set_font_size(FONT_SIZE_TINY)
            ctx.set_source_rgba(*DIM_WHITE[:3], 0.5)
            ext2 = ctx.text_extents(label_text)
            ctx.move_to(cx - ext2.x_advance / 2, cy + radius + 14)
            ctx.show_text(label_text)
        ctx.restore()

    # ── Drone status panel ───────────────────────
    def draw_drone_panel(self, ctx: cairo.Context, x: float, y: float,
                         drone_id: int, battery: float, altitude: float,
                         speed: float, regime: str = "RECON",
                         alive: bool = True):
        """Per-drone status readout: ID, battery, alt, speed, regime."""
        ctx.save()
        panel_w, panel_h = 160, 80
        color = CYAN if alive else RED

        # Background
        ctx.set_source_rgba(0.02, 0.02, 0.06, 0.7)
        ctx.rectangle(x, y, panel_w, panel_h)
        ctx.fill()

        # Border
        ctx.set_source_rgba(*color[:3], 0.4)
        ctx.set_line_width(1)
        ctx.rectangle(x, y, panel_w, panel_h)
        ctx.stroke()

        # ID
        ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(13)
        ctx.set_source_rgba(*color)
        status = regime if alive else "LOST"
        ctx.move_to(x + 6, y + 16)
        ctx.show_text(f"D-{drone_id:02d}  [{status}]")

        # Bars
        bar_x = x + 6
        bar_w = panel_w - 50
        ctx.set_font_size(FONT_SIZE_TINY)

        # Battery
        bat_color = GREEN if battery > 0.3 else (AMBER if battery > 0.1 else RED)
        self.draw_status_bar(ctx, bar_x, y + 26, bar_w, 6,
                             battery * 100, 100, bat_color, "BAT")
        # Altitude
        self.draw_status_bar(ctx, bar_x, y + 44, bar_w, 6,
                             altitude, 500, CYAN, "ALT")
        # Speed
        self.draw_status_bar(ctx, bar_x, y + 62, bar_w, 6,
                             speed, 25, CYAN, "SPD")
        ctx.restore()

    # ── Coverage percentage ──────────────────────
    def draw_coverage_counter(self, ctx: cairo.Context, x: float, y: float,
                              pct: float, color=GREEN):
        ctx.save()
        ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(FONT_SIZE_MEDIUM)
        ctx.set_source_rgba(*color)
        ctx.move_to(x, y)
        ctx.show_text(f"COVERAGE: {pct:.0f}%")
        ctx.restore()

    # ── Scene title ──────────────────────────────
    def draw_scene_label(self, ctx: cairo.Context, text: str,
                         alpha: float = 0.4):
        ctx.save()
        ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(FONT_SIZE_TINY)
        ctx.set_source_rgba(*DIM_WHITE[:3], alpha)
        ctx.move_to(30, 40)
        ctx.show_text(text)
        ctx.restore()

    # ── Split-screen divider ─────────────────────
    def draw_split_divider(self, ctx: cairo.Context, x: float,
                           color=CYAN, alpha: float = 0.5):
        ctx.save()
        ctx.set_source_rgba(*color[:3], alpha)
        ctx.set_line_width(2)
        ctx.move_to(x, 0)
        ctx.line_to(x, HEIGHT)
        ctx.stroke()
        # Small arrows at top and bottom
        s = 6
        for ay in [40, HEIGHT - 40]:
            ctx.move_to(x - s, ay - s)
            ctx.line_to(x, ay)
            ctx.line_to(x - s, ay + s)
            ctx.move_to(x + s, ay - s)
            ctx.line_to(x, ay)
            ctx.line_to(x + s, ay + s)
            ctx.stroke()
        ctx.restore()

    # ── Timer display ────────────────────────────
    def draw_timer(self, ctx: cairo.Context, x: float, y: float,
                   value_sec: float, color=CYAN, label_text: str = ""):
        ctx.save()
        ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_BOLD)
        if label_text:
            ctx.set_font_size(FONT_SIZE_TINY)
            ctx.set_source_rgba(*DIM_WHITE[:3], 0.5)
            ctx.move_to(x, y - 4)
            ctx.show_text(label_text)

        ctx.set_font_size(FONT_SIZE_MEDIUM)
        ctx.set_source_rgba(*color)
        ctx.move_to(x, y + 20)
        ctx.show_text(f"{value_sec:.2f}s")
        ctx.restore()

    # ── Stats overlay ────────────────────────────
    def draw_stats_overlay(self, ctx: cairo.Context, stats: list[str],
                           x: float, y: float, color=CYAN):
        ctx.save()
        ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(FONT_SIZE_MEDIUM)
        for i, line in enumerate(stats):
            ctx.set_source_rgba(*color[:3], 0.9)
            ctx.move_to(x, y + i * 36)
            ctx.show_text(line)
        ctx.restore()
