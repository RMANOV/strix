"""Scene 2: Reconnaissance — split-screen tactical map + particle uncertainty.

Local frames 0–1800 (global 900–2700, 60 seconds at 30 fps).

Phases:
  0–180     Split reveal: cyan divider wipes in from center
  180–1200  Navigation: LEFT map with drones patrolling around SAM, RIGHT particle clouds
  1200–1800 Sensor fusion: particles tighten, coverage reaches 80 %
"""

import math
import numpy as np
import cairo

from config import (
    WIDTH, HEIGHT, FPS, CYAN, GREEN, RED, DIM_WHITE, AMBER,
    NUM_RECON_DRONES, SAM_ZONE, FONT_FAMILY, FONT_SIZE_TINY,
)
from renderer.cairo_surface import CairoFrame
from renderer.map_renderer import TacticalMap
from renderer.particles import ParticleSystem
from renderer.shapes import vignette, drone_icon
from renderer.text_renderer import label, glow_text
from renderer.easing import (
    ease_out_cubic, ease_in_out_cubic, smoothstep, lerp, clamp, remap,
)
from scenes.base import BaseScene


# ---------------------------------------------------------------------------
# Cubic-spline helper (Catmull-Rom → position on closed loop)
# ---------------------------------------------------------------------------

class CubicSpline:
    """Catmull-Rom spline through closed-loop waypoints."""

    def __init__(self, points: list[tuple[float, float]]):
        self.pts = np.array(points, dtype=np.float64)
        self.n = len(points)

    def eval(self, t: float) -> tuple[float, float]:
        """Evaluate spline at parameter *t* (wraps around [0, n))."""
        t = t % self.n
        i = int(t)
        frac = t - i

        p0 = self.pts[(i - 1) % self.n]
        p1 = self.pts[i % self.n]
        p2 = self.pts[(i + 1) % self.n]
        p3 = self.pts[(i + 2) % self.n]

        # Catmull-Rom weights
        tt = frac
        tt2 = tt * tt
        tt3 = tt2 * tt
        q = (
            0.5 * ((-p0 + 3 * p1 - 3 * p2 + p3) * tt3
                   + (2 * p0 - 5 * p1 + 4 * p2 - p3) * tt2
                   + (-p0 + p2) * tt
                   + 2 * p1)
        )
        return float(q[0]), float(q[1])

    def tangent(self, t: float) -> float:
        """Return heading angle (radians) at parameter t."""
        eps = 0.01
        x0, y0 = self.eval(t - eps)
        x1, y1 = self.eval(t + eps)
        return math.atan2(y1 - y0, x1 - x0)


# ---------------------------------------------------------------------------
# Patrol paths — four loops that orbit the map while avoiding the SAM zone
# ---------------------------------------------------------------------------

def _build_patrol_paths() -> list[CubicSpline]:
    """Return 4 CubicSpline patrol loops in world-coordinates."""
    sam_cx = SAM_ZONE[0] + SAM_ZONE[2] / 2  # 1900
    sam_cy = SAM_ZONE[1] + SAM_ZONE[3] / 2  # 1500

    # Drone 0: wide left loop (west side)
    path0 = CubicSpline([
        (400, 500), (600, 1200), (400, 2000), (800, 2500),
        (1200, 2200), (1100, 1500), (1200, 800), (800, 400),
    ])
    # Drone 1: upper right loop (north-east, above SAM)
    path1 = CubicSpline([
        (2400, 300), (3000, 500), (3600, 400), (3800, 900),
        (3400, 700), (2800, 600), (2200, 500), (2000, 300),
    ])
    # Drone 2: lower right loop (south-east, below SAM)
    path2 = CubicSpline([
        (2200, 2200), (2800, 2500), (3400, 2600), (3800, 2300),
        (3600, 1900), (3000, 1800), (2500, 2000), (2200, 2400),
    ])
    # Drone 3: center skimming path (threads between SAM edges)
    path3 = CubicSpline([
        (1000, 600), (1500, 400), (2000, 300), (2600, 500),
        (3000, 1200), (2800, 2000), (2200, 2600), (1500, 2400),
        (900, 1800), (800, 1100),
    ])
    return [path0, path1, path2, path3]


# ---------------------------------------------------------------------------
# Isometric grid helper
# ---------------------------------------------------------------------------

def _draw_iso_grid(ctx: cairo.Context, ox: float, oy: float,
                   w: float, h: float, rows: int = 10, cols: int = 10,
                   color=CYAN, alpha: float = 0.06):
    """Draw a pseudo-isometric ground-plane grid inside the given rect."""
    ctx.save()
    ctx.set_source_rgba(*color[:3], alpha)
    ctx.set_line_width(0.5)

    # Vanishing-point perspective: compress Y toward horizon
    horizon_y = oy + h * 0.15
    bottom_y = oy + h * 0.95
    left_x = ox + w * 0.05
    right_x = ox + w * 0.95

    for r in range(rows + 1):
        t = r / rows
        # Exponential spacing: denser near horizon
        yt = horizon_y + (bottom_y - horizon_y) * (t ** 1.6)
        # Narrow at horizon, wide at bottom
        squeeze = 0.2 + 0.8 * (t ** 1.2)
        lx = ox + w * 0.5 - (w * 0.45) * squeeze
        rx = ox + w * 0.5 + (w * 0.45) * squeeze
        ctx.move_to(lx, yt)
        ctx.line_to(rx, yt)
        ctx.stroke()

    for c in range(cols + 1):
        ct = c / cols
        # Top (horizon) point
        tx = lerp(ox + w * 0.35, ox + w * 0.65, ct)
        # Bottom point
        bx = lerp(left_x, right_x, ct)
        ctx.move_to(tx, horizon_y)
        ctx.line_to(bx, bottom_y)
        ctx.stroke()

    ctx.restore()


# ---------------------------------------------------------------------------
# Scene 2 class
# ---------------------------------------------------------------------------

class Scene2Recon(BaseScene):
    """Reconnaissance: split-screen tactical map + particle uncertainty clouds."""

    TOTAL = 1800  # local frames

    # Phase boundaries (local frames)
    SPLIT_START, SPLIT_END = 0, 180
    NAV_START, NAV_END = 180, 1200
    FUSION_START, FUSION_END = 1200, 1800

    def setup(self):
        # -- Tactical map (left half) --
        self.tac = TacticalMap(viewport=(0, 0, 960, 1080))

        # -- Patrol splines & time offsets --
        self.paths = _build_patrol_paths()
        # Each drone starts at a different spline parameter offset
        self.t_offsets = [0.0, 2.5, 1.0, 4.0]

        # -- Per-drone state --
        self.batteries = [1.0, 0.98, 0.97, 0.99]  # slowly deplete
        self.altitudes = [120.0, 135.0, 110.0, 125.0]
        self.speeds = [12.0, 14.0, 11.0, 13.0]

        # -- Coverage grid (4 rows x 6 cols) --
        self.coverage = np.zeros((4, 6), dtype=np.float64)

        # -- Particle systems (right panel, 200 each) --
        self.particles: list[ParticleSystem] = []
        for _ in range(NUM_RECON_DRONES):
            self.particles.append(ParticleSystem(max_particles=250))

        # Drone colors for right panel (slight variations)
        self.drone_colors = [
            (*CYAN[:3], 0.7),
            (0.0, 0.9, 0.8, 0.7),
            (0.1, 0.7, 1.0, 0.7),
            (0.0, 1.0, 0.9, 0.7),
        ]

        self._initialized = True

    # ── helpers ────────────────────────────────────────────

    def _drone_world_pos(self, drone_idx: int, local_frame: int
                         ) -> tuple[float, float]:
        """Current world-coordinate position of drone on its patrol spline."""
        speed = 0.0025  # spline parameter per frame
        t = self.t_offsets[drone_idx] + local_frame * speed
        return self.paths[drone_idx].eval(t)

    def _drone_heading(self, drone_idx: int, local_frame: int) -> float:
        speed = 0.0025
        t = self.t_offsets[drone_idx] + local_frame * speed
        return self.paths[drone_idx].tangent(t)

    def _update_drone_state(self, local_frame: int):
        """Slowly deplete battery, wobble altitude/speed."""
        dt = 1.0 / FPS
        for i in range(NUM_RECON_DRONES):
            # Battery: ~0.02 %/s depletion
            self.batteries[i] = max(0.3, self.batteries[i] - 0.0002 * dt)
            # Altitude wobble +-5 m
            self.altitudes[i] = 120.0 + 8.0 * math.sin(
                local_frame * 0.02 + i * 1.5)
            # Speed wobble +-2 m/s
            self.speeds[i] = 12.0 + 3.0 * math.sin(
                local_frame * 0.03 + i * 2.0)

    def _update_coverage(self, local_frame: int):
        """Progressively fill the coverage grid based on drone positions."""
        for i in range(NUM_RECON_DRONES):
            wx, wy = self._drone_world_pos(i, local_frame)
            # Map world pos → grid cell
            col = int(clamp(wx / 4000.0 * 6, 0, 5))
            row = int(clamp(wy / 3000.0 * 4, 0, 3))
            # Increase coverage in this cell and neighbors
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    r2, c2 = row + dr, col + dc
                    if 0 <= r2 < 4 and 0 <= c2 < 6:
                        dist = abs(dr) + abs(dc)
                        inc = 0.004 if dist == 0 else 0.0015
                        self.coverage[r2, c2] = min(
                            1.0, self.coverage[r2, c2] + inc)

        # During fusion phase, accelerate fill toward 80 %
        if local_frame >= self.FUSION_START:
            fusion_t = self.phase_progress(
                local_frame, self.FUSION_START, self.FUSION_END)
            target = 0.8 * smoothstep(fusion_t)
            self.coverage = np.maximum(
                self.coverage, target * np.ones_like(self.coverage) * 0.9)

    def _coverage_pct(self) -> float:
        return float(np.mean(self.coverage)) * 100.0

    # ── right-panel particle management ───────────────────

    def _right_panel_pos(self, drone_idx: int, local_frame: int
                         ) -> tuple[float, float]:
        """Map drone world pos → right-panel pixel coords (960..1920, 0..1080)."""
        wx, wy = self._drone_world_pos(drone_idx, local_frame)
        # Map 0..4000 → 990..1890 and 0..3000 → 60..1020
        px = 990.0 + (wx / 4000.0) * 900.0
        py = 60.0 + (wy / 3000.0) * 960.0
        return px, py

    def _emit_particles(self, local_frame: int):
        """Emit uncertainty cloud particles from each drone's position."""
        for i in range(NUM_RECON_DRONES):
            ps = self.particles[i]
            px, py = self._right_panel_pos(i, local_frame)

            # Determine spread based on phase
            if local_frame < self.FUSION_START:
                spread = 60.0
                vel_spread = 15.0
            else:
                fusion_t = self.phase_progress(
                    local_frame, self.FUSION_START, self.FUSION_END)
                spread = lerp(60.0, 12.0, smoothstep(fusion_t))
                vel_spread = lerp(15.0, 3.0, smoothstep(fusion_t))

            # Emit a few particles per frame to maintain the cloud
            if ps.count < 200:
                ps.emit_point(
                    pos=(px, py),
                    vel=(0, 0),
                    count=5,
                    life=2.5,
                    color=self.drone_colors[i],
                    size=2.5,
                    spread=spread,
                    vel_spread=vel_spread,
                )

    def _update_particles(self, local_frame: int):
        """Step particle physics with optional attractor during fusion."""
        dt = 1.0 / FPS
        for i in range(NUM_RECON_DRONES):
            ps = self.particles[i]
            px, py = self._right_panel_pos(i, local_frame)

            attractor = None
            attract_str = 0.0
            if local_frame >= self.FUSION_START:
                fusion_t = self.phase_progress(
                    local_frame, self.FUSION_START, self.FUSION_END)
                attractor = (px, py)
                attract_str = lerp(20.0, 120.0, smoothstep(fusion_t))

            ps.update(
                dt=dt,
                drag=0.96,
                attractor=attractor,
                attract_strength=attract_str,
            )

    # ── rendering ─────────────────────────────────────────

    def _draw_left_panel(self, ctx: cairo.Context, local_frame: int):
        """LEFT half: tactical map with grid, SAM, drones, coverage."""
        ctx.save()
        ctx.rectangle(0, 0, 960, HEIGHT)
        ctx.clip()

        # Grid
        self.tac.draw_grid(ctx)

        # Coverage sectors (behind everything)
        self.tac.draw_coverage_sectors(ctx, self.coverage)

        # SAM zone with pulsing
        pulse = local_frame * 0.08
        self.tac.draw_sam_zone(ctx, pulse)

        # Kill zone marker at center of SAM
        sam_cx = SAM_ZONE[0] + SAM_ZONE[2] / 2
        sam_cy = SAM_ZONE[1] + SAM_ZONE[3] / 2
        self.tac.draw_kill_zone(ctx, sam_cx, sam_cy, 250, pulse)

        # Base
        self.tac.draw_base(ctx)

        # Drones + sensor fans + paths
        for i in range(NUM_RECON_DRONES):
            wx, wy = self._drone_world_pos(i, local_frame)
            heading = self._drone_heading(i, local_frame)

            # Sensor fan
            self.tac.draw_sector_fan(
                ctx, wx, wy, heading,
                arc=math.pi / 3, range_m=500,
                color=GREEN, alpha=0.06,
            )

            # Drone trail: last N positions as dashed path
            trail_pts = []
            for back in range(0, min(local_frame, 120), 4):
                twx, twy = self._drone_world_pos(i, local_frame - back)
                trail_pts.append((twx, twy))
            if len(trail_pts) >= 2:
                self.tac.draw_path(ctx, trail_pts, color=CYAN, alpha=0.12)

            # Drone icon
            self.tac.draw_drone(ctx, wx, wy, heading, color=CYAN, size=10)

        ctx.restore()

    def _draw_right_panel(self, ctx: cairo.Context, local_frame: int):
        """RIGHT half: isometric particle uncertainty view."""
        ctx.save()
        ctx.rectangle(960, 0, 960, HEIGHT)
        ctx.clip()

        # Isometric terrain grid as background
        _draw_iso_grid(ctx, 960, 0, 960, 1080, rows=12, cols=12)

        # Particles (uncertainty clouds)
        for i in range(NUM_RECON_DRONES):
            self.particles[i].draw_with_glow(ctx, alpha_mul=0.8)

            # Small drone marker on right panel
            px, py = self._right_panel_pos(i, local_frame)
            heading = self._drone_heading(i, local_frame)
            drone_icon(ctx, px, py, heading, color=self.drone_colors[i], size=7)

        # "UNCERTAINTY CLOUD" label
        label(ctx, "PARTICLE UNCERTAINTY", 990, 40,
              color=CYAN, font_size=14, bold=True)

        # Fusion progress indicator (during fusion phase)
        if local_frame >= self.FUSION_START:
            fusion_t = self.phase_progress(
                local_frame, self.FUSION_START, self.FUSION_END)
            label(ctx, f"FUSION: {fusion_t * 100:.0f}%", 990, 60,
                  color=GREEN, font_size=12, bold=True)

        ctx.restore()

    def _draw_split_divider(self, ctx: cairo.Context, local_frame: int):
        """Animated cyan split divider."""
        if local_frame < self.SPLIT_END:
            # Animate: divider slides in from center
            t = self.phase_progress(local_frame, self.SPLIT_START, self.SPLIT_END)
            reveal = ease_out_cubic(max(0.0, t))
            # Divider grows from mid-height outward
            mid = HEIGHT / 2
            half_h = mid * reveal
            top = mid - half_h
            bot = mid + half_h

            ctx.save()
            ctx.set_source_rgba(*CYAN[:3], 0.6 * reveal)
            ctx.set_line_width(2)
            ctx.move_to(960, top)
            ctx.line_to(960, bot)
            ctx.stroke()

            # Glow pulse at midpoint
            if reveal > 0.3:
                pat = cairo.RadialGradient(960, mid, 0, 960, mid, 30)
                pat.add_color_stop_rgba(0, *CYAN[:3], 0.4 * reveal)
                pat.add_color_stop_rgba(1, *CYAN[:3], 0.0)
                ctx.set_source(pat)
                ctx.arc(960, mid, 30, 0, 2 * math.pi)
                ctx.fill()
            ctx.restore()
        else:
            # Fully revealed — use HUD helper
            self.hud.draw_split_divider(ctx, 960)

    def _draw_drone_panels(self, ctx: cairo.Context, local_frame: int):
        """Per-drone HUD panels along the bottom."""
        panel_y = HEIGHT - 95
        for i in range(NUM_RECON_DRONES):
            # Distribute panels across the bottom, avoiding the divider
            if i < 2:
                px = 30 + i * 175
            else:
                px = 990 + (i - 2) * 175
            self.hud.draw_drone_panel(
                ctx, px, panel_y,
                drone_id=i + 1,
                battery=self.batteries[i],
                altitude=self.altitudes[i],
                speed=self.speeds[i],
                regime="RECON",
                alive=True,
            )

    def _draw_coverage_hud(self, ctx: cairo.Context, local_frame: int):
        """Coverage percentage counter in top area."""
        pct = self._coverage_pct()
        # Position above the right panel
        self.hud.draw_coverage_counter(ctx, 1400, 1040, pct, color=GREEN)

    # ── main render ───────────────────────────────────────

    def render_frame(self, local_frame: int, total_frames: int,
                     global_frame: int = 0) -> np.ndarray:
        if not self._initialized:
            self.setup()

        frame = self.new_frame()
        ctx = frame.ctx

        # --- State updates ---
        self._update_drone_state(local_frame)
        self._update_coverage(local_frame)
        self._emit_particles(local_frame)
        self._update_particles(local_frame)

        # --- Phase: Split reveal (0–180) ---
        if local_frame < self.SPLIT_END:
            reveal_t = self.phase_progress(
                local_frame, self.SPLIT_START, self.SPLIT_END)
            reveal = ease_out_cubic(max(0.0, reveal_t))

            # Fade in the two panels
            if reveal > 0.1:
                ctx.save()
                ctx.push_group()
                self._draw_left_panel(ctx, local_frame)
                self._draw_right_panel(ctx, local_frame)
                ctx.pop_group_to_source()
                ctx.paint_with_alpha(smoothstep(reveal))
                ctx.restore()

            # Scene title
            if reveal_t > 0.3:
                title_alpha = smoothstep(remap(reveal_t, 0.3, 0.8))
                label(ctx, "SCENE 2 — RECONNAISSANCE", 30, 40,
                      color=DIM_WHITE, font_size=12, bold=False)

        else:
            # --- Phase: Navigation + Fusion ---
            self._draw_left_panel(ctx, local_frame)
            self._draw_right_panel(ctx, local_frame)

        # --- Divider ---
        self._draw_split_divider(ctx, local_frame)

        # --- HUD overlays ---
        self.hud.draw_full(ctx, local_frame, global_frame)
        self.hud.draw_scene_label(ctx, "02 | RECONNAISSANCE")
        self._draw_drone_panels(ctx, local_frame)
        self._draw_coverage_hud(ctx, local_frame)

        # --- Vignette ---
        vignette(ctx, strength=0.5)

        return frame.to_rgb()
