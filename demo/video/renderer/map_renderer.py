"""Tactical map renderer: grid, zones, paths, drones, coverage."""

import cairo
import math
import numpy as np

from config import (
    CYAN, CYAN_DIM, GREEN, GREEN_DIM, RED, RED_DIM, DIM_WHITE,
    MAP_WORLD_W, MAP_WORLD_H, MAP_GRID_SPACING, SAM_ZONE, BASE_POS,
    DRONE_ICON_SIZE, FONT_FAMILY, FONT_SIZE_TINY,
)
from renderer.shapes import drone_icon, drone_glow, kill_zone_marker


class TacticalMap:
    """World-coordinate tactical map with pixel transform."""

    def __init__(self, viewport: tuple = None):
        """viewport: (x, y, w, h) pixel region on screen."""
        if viewport is None:
            viewport = (0, 0, 1920, 1080)
        self.vp_x, self.vp_y, self.vp_w, self.vp_h = viewport
        self.world_w = MAP_WORLD_W
        self.world_h = MAP_WORLD_H
        self.scale_x = self.vp_w / self.world_w
        self.scale_y = self.vp_h / self.world_h

    def w2p(self, wx: float, wy: float) -> tuple:
        """World coords (meters) → pixel coords."""
        px = self.vp_x + wx * self.scale_x
        py = self.vp_y + wy * self.scale_y
        return px, py

    def w2p_dist(self, wd: float) -> float:
        """World distance → pixel distance (using X scale)."""
        return wd * self.scale_x

    def draw_grid(self, ctx: cairo.Context, spacing: int = MAP_GRID_SPACING,
                  color=CYAN, alpha: float = 0.08):
        """Draw cartographic grid lines."""
        ctx.save()
        ctx.set_source_rgba(*color[:3], alpha)
        ctx.set_line_width(0.5)

        # Vertical lines
        x = 0
        while x <= self.world_w:
            px, _ = self.w2p(x, 0)
            _, py_end = self.w2p(0, self.world_h)
            ctx.move_to(px, self.vp_y)
            ctx.line_to(px, py_end)
            ctx.stroke()
            x += spacing

        # Horizontal lines
        y = 0
        while y <= self.world_h:
            _, py = self.w2p(0, y)
            px_end, _ = self.w2p(self.world_w, 0)
            ctx.move_to(self.vp_x, py)
            ctx.line_to(px_end, py)
            ctx.stroke()
            y += spacing

        # Grid labels (every other line)
        ctx.set_font_size(FONT_SIZE_TINY)
        ctx.set_source_rgba(*DIM_WHITE[:3], 0.25)
        x = 0
        while x <= self.world_w:
            px, py = self.w2p(x, 0)
            ctx.move_to(px + 3, py + 14)
            ctx.show_text(f"{int(x)}m")
            x += spacing * 2
        ctx.restore()

    def draw_sam_zone(self, ctx: cairo.Context, pulse: float = 0.0):
        """Draw SAM restricted corridor with red hatching."""
        sx, sy, sw, sh = SAM_ZONE
        px, py = self.w2p(sx, sy)
        pw = self.w2p_dist(sw)
        ph = sh * self.scale_y

        pulse_alpha = 0.08 + 0.06 * math.sin(pulse)

        # Filled zone
        ctx.save()
        ctx.set_source_rgba(*RED[:3], pulse_alpha)
        ctx.rectangle(px, py, pw, ph)
        ctx.fill()

        # Hatching (diagonal lines) — clipped to zone rect
        ctx.save()
        ctx.rectangle(px, py, pw, ph)
        ctx.clip()
        ctx.set_source_rgba(*RED[:3], 0.12)
        ctx.set_line_width(1)
        step = 12
        for offset in range(-int(ph), int(pw + ph), step):
            ctx.move_to(px + offset, py)
            ctx.line_to(px + offset + ph, py + ph)
            ctx.stroke()
        ctx.restore()

        # Border
        ctx.set_source_rgba(*RED[:3], 0.35 + 0.15 * math.sin(pulse))
        ctx.set_line_width(1.5)
        ctx.rectangle(px, py, pw, ph)
        ctx.stroke()

        # Label
        ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(11)
        ctx.set_source_rgba(*RED[:3], 0.6)
        ctx.move_to(px + 5, py + 15)
        ctx.show_text("SAM CORRIDOR")
        ctx.restore()

    def draw_kill_zone(self, ctx: cairo.Context, wx: float, wy: float,
                       radius_m: float, pulse: float):
        """Draw pulsing kill zone at world coords."""
        px, py = self.w2p(wx, wy)
        pr = self.w2p_dist(radius_m)
        kill_zone_marker(ctx, px, py, pr, pulse)

    def draw_base(self, ctx: cairo.Context, color=GREEN):
        """Draw the base/launch location."""
        px, py = self.w2p(*BASE_POS)
        ctx.save()
        # Diamond shape
        s = 10
        ctx.move_to(px, py - s)
        ctx.line_to(px + s, py)
        ctx.line_to(px, py + s)
        ctx.line_to(px - s, py)
        ctx.close_path()
        ctx.set_source_rgba(*color[:3], 0.7)
        ctx.fill_preserve()
        ctx.set_source_rgba(*color)
        ctx.set_line_width(1.5)
        ctx.stroke()

        ctx.set_font_size(10)
        ctx.set_source_rgba(*color[:3], 0.6)
        ctx.move_to(px + 14, py + 4)
        ctx.show_text("BASE")
        ctx.restore()

    def draw_drone(self, ctx: cairo.Context, wx: float, wy: float,
                   heading: float = 0.0, color=CYAN,
                   size: float = DRONE_ICON_SIZE, glow: bool = True):
        """Draw drone at world position with optional glow."""
        px, py = self.w2p(wx, wy)
        if glow:
            drone_glow(ctx, px, py, color, radius=size * 2)
        drone_icon(ctx, px, py, heading, color, size)

    def draw_path(self, ctx: cairo.Context, waypoints_world: list,
                  color=CYAN, alpha: float = 0.3, dashed: bool = True):
        """Draw path through world-coordinate waypoints."""
        if len(waypoints_world) < 2:
            return
        ctx.save()
        ctx.set_source_rgba(*color[:3], alpha)
        ctx.set_line_width(1.5)
        if dashed:
            ctx.set_dash([6, 4])

        px, py = self.w2p(*waypoints_world[0])
        ctx.move_to(px, py)
        for wp in waypoints_world[1:]:
            px, py = self.w2p(*wp)
            ctx.line_to(px, py)
        ctx.stroke()
        ctx.restore()

    def draw_coverage_sectors(self, ctx: cairo.Context,
                              coverage: np.ndarray,
                              rows: int = 4, cols: int = 6):
        """Draw coverage grid. coverage: (rows, cols) array of 0→1 values."""
        cell_w = self.vp_w / cols
        cell_h = self.vp_h / rows

        for r in range(rows):
            for c in range(cols):
                v = float(coverage[r, c]) if r < coverage.shape[0] and c < coverage.shape[1] else 0
                if v < 0.01:
                    continue
                x = self.vp_x + c * cell_w
                y = self.vp_y + r * cell_h
                ctx.set_source_rgba(*GREEN[:3], v * 0.15)
                ctx.rectangle(x, y, cell_w, cell_h)
                ctx.fill()
                # Border
                ctx.set_source_rgba(*GREEN[:3], v * 0.3)
                ctx.set_line_width(0.5)
                ctx.rectangle(x, y, cell_w, cell_h)
                ctx.stroke()

    def draw_sector_fan(self, ctx: cairo.Context, wx: float, wy: float,
                        heading: float, arc: float = math.pi / 3,
                        range_m: float = 500, color=CYAN, alpha: float = 0.08):
        """Draw a sensor coverage fan from drone position."""
        px, py = self.w2p(wx, wy)
        pr = self.w2p_dist(range_m)
        ctx.save()
        ctx.set_source_rgba(*color[:3], alpha)
        ctx.move_to(px, py)
        ctx.arc(px, py, pr, heading - arc / 2, heading + arc / 2)
        ctx.close_path()
        ctx.fill()
        ctx.restore()

    def draw_enemy_vehicle(self, ctx: cairo.Context, wx: float, wy: float,
                           heading: float = 0.0, color=RED):
        """Draw enemy vehicle icon (larger, red)."""
        px, py = self.w2p(wx, wy)
        ctx.save()
        ctx.translate(px, py)
        ctx.rotate(heading)
        # Box shape for vehicle
        s = 8
        ctx.rectangle(-s, -s * 0.6, s * 2, s * 1.2)
        ctx.set_source_rgba(*color[:3], 0.7)
        ctx.fill_preserve()
        ctx.set_source_rgba(*color)
        ctx.set_line_width(1.5)
        ctx.stroke()
        ctx.restore()
