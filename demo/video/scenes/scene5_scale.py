"""Scene 5: Scale + Closing — 100+ drone swarm scaling & STRIX logo reveal.

Local frames 0-900  (global 6300-7200, 30 seconds @ 30fps).

Phases:
    0-180:   Slider at 5 drones, shown on map
  180-360:   Slider animates to 100, drones populate map
  360-540:   Slider to 500, Poisson-disc distributed drones fill map
  540-660:   Stats overlay — capabilities summary
  660-780:   Particles converge into STRIX logo
  780-900:   Fade to black
"""

import math
import random

import cairo
import numpy as np

from config import (
    WIDTH, HEIGHT, FPS, CYAN, GREEN, WHITE, DIM_WHITE, AMBER,
    FONT_FAMILY, FONT_SIZE_SMALL, FONT_SIZE_TINY, FONT_SIZE_MEDIUM,
    MAP_WORLD_W, MAP_WORLD_H,
)
from renderer.cairo_surface import CairoFrame
from renderer.easing import (
    ease_out_cubic, ease_in_out_cubic, smoothstep, clamp, lerp, remap,
)
from renderer.map_renderer import TacticalMap
from renderer.particles import ParticleSystem
from renderer.shapes import slider_widget, drone_icon, drone_glow, strix_logo, vignette
from renderer.text_renderer import label, glow_text
from scenes.base import BaseScene


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def poisson_disc_points(width: float, height: float,
                        min_dist: float, count: int,
                        seed: int = 42) -> list[tuple[float, float]]:
    """Generate up to `count` well-spaced points via rejection sampling."""
    rng = random.Random(seed)
    points: list[tuple[float, float]] = []
    for _ in range(count * 30):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        ok = all((x - px) ** 2 + (y - py) ** 2 > min_dist ** 2
                 for px, py in points)
        if ok:
            points.append((x, y))
        if len(points) >= count:
            break
    return points


def _strix_target_positions(cx: float, cy: float,
                            spacing: float = 12.0) -> list[tuple[float, float]]:
    """Return a list of pixel positions that spell 'STRIX' on a coarse grid.

    Each letter is defined on a 5-tall x variable-width grid, then scaled
    and offset so the whole word is centered at (cx, cy).
    """
    # Each letter: list of (row, col) cells that are "on"
    letters = {
        "S": [(0,1),(0,2),(0,3), (1,0),(1,1), (2,1),(2,2), (3,2),(3,3), (4,0),(4,1),(4,2)],
        "T": [(0,0),(0,1),(0,2),(0,3),(0,4), (1,2),(2,2),(3,2),(4,2)],
        "R": [(0,0),(0,1),(0,2), (1,0),(1,3), (2,0),(2,1),(2,2), (3,0),(3,2), (4,0),(4,3)],
        "I": [(0,0),(0,1),(0,2), (1,1),(2,1),(3,1), (4,0),(4,1),(4,2)],
        "X": [(0,0),(0,4), (1,1),(1,3), (2,2), (3,1),(3,3), (4,0),(4,4)],
    }
    letter_widths = {"S": 4, "T": 5, "R": 4, "I": 3, "X": 5}
    word = "STRIX"
    gap = 2  # columns between letters

    # Total width in grid cells
    total_cols = sum(letter_widths[ch] for ch in word) + gap * (len(word) - 1)
    total_rows = 5

    positions: list[tuple[float, float]] = []
    col_offset = 0
    for ch in word:
        for r, c in letters[ch]:
            px = (col_offset + c - total_cols / 2) * spacing + cx
            py = (r - total_rows / 2) * spacing + cy
            positions.append((px, py))
        col_offset += letter_widths[ch] + gap

    return positions


# ---------------------------------------------------------------------------
# Phase boundaries (local frames)
# ---------------------------------------------------------------------------
_P_SLIDER5   = (0, 180)
_P_SLIDER100 = (180, 360)
_P_SLIDER500 = (360, 540)
_P_STATS     = (540, 660)
_P_LOGO      = (660, 780)
_P_FADE      = (780, 900)


class Scene5Scale(BaseScene):
    """Scale demonstration and closing logo."""

    def __init__(self):
        super().__init__()
        # Tactical map — leave room for slider at top
        self.tmap = TacticalMap(viewport=(0, 80, WIDTH, HEIGHT - 80))

        # Pre-generate drone positions for each tier using Poisson disc
        self._positions_5   = poisson_disc_points(MAP_WORLD_W, MAP_WORLD_H, 600, 5, seed=10)
        self._positions_100 = poisson_disc_points(MAP_WORLD_W, MAP_WORLD_H, 200, 100, seed=20)
        self._positions_500 = poisson_disc_points(MAP_WORLD_W, MAP_WORLD_H, 80, 500, seed=30)

        # Pre-generate stable headings per tier
        self._headings_5   = [random.Random(10).uniform(0, 2 * math.pi) for _ in range(5)]
        self._headings_100 = [random.Random(20 + i).uniform(0, 2 * math.pi) for i in range(100)]
        self._headings_500 = [random.Random(30 + i).uniform(0, 2 * math.pi) for i in range(500)]

        # Particle system for logo convergence
        self.particles = ParticleSystem(max_particles=2000)
        self._logo_targets = _strix_target_positions(WIDTH / 2, HEIGHT / 2, spacing=14.0)
        self._particles_initialized = False

        # Stats overlay lines
        self._stats_lines = [
            "100+ DRONES  \u00b7  AUTONOMOUS SWARM",
            "ZERO GPS DEPENDENCY",
            "SUB-200ms CONSENSUS  \u00b7  ~20 BYTES/MSG",
            "PX4 / ARDUPILOT / DJI / ROS2 COMPATIBLE",
        ]

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------

    def render_frame(self, local_frame: int, total_frames: int,
                     global_frame: int = 0) -> np.ndarray:
        if not self._initialized:
            self.setup()
            self._initialized = True

        frame = self.new_frame()
        ctx = frame.ctx

        # ── Determine slider value and visible drones ────────────
        slider_val, drone_positions, drone_headings = self._get_drones(local_frame)

        # ── Phases ───────────────────────────────────────────────
        in_map_phase = local_frame < _P_LOGO[0]
        in_stats = self.in_phase(local_frame, *_P_STATS)
        in_logo = self.in_phase(local_frame, *_P_LOGO)
        in_fade = self.in_phase(local_frame, *_P_FADE)

        # ── Map + slider (visible up through stats phase) ────────
        if in_map_phase:
            self._draw_map_phase(ctx, local_frame, slider_val,
                                 drone_positions, drone_headings)

        # ── Stats overlay ────────────────────────────────────────
        if in_stats:
            p = self.phase_progress(local_frame, *_P_STATS)
            self._draw_stats(ctx, p)

        # ── Logo convergence ─────────────────────────────────────
        if in_logo or in_fade:
            p_logo = self.phase_progress(local_frame, *_P_LOGO)
            self._draw_logo_phase(ctx, local_frame, p_logo, in_fade)

        # ── HUD (always, dims during fade) ───────────────────────
        fade_alpha = 1.0
        if in_fade:
            p_fade = self.phase_progress(local_frame, *_P_FADE)
            fade_alpha = 1.0 - ease_out_cubic(clamp(p_fade))

        if fade_alpha > 0.01:
            self.hud.draw_full(ctx, local_frame, global_frame)
            vignette(ctx, strength=0.5)

        # ── Fade to black ────────────────────────────────────────
        if in_fade:
            p_fade = self.phase_progress(local_frame, *_P_FADE)
            black_alpha = ease_in_out_cubic(clamp(p_fade))
            ctx.set_source_rgba(0, 0, 0, black_alpha)
            ctx.rectangle(0, 0, WIDTH, HEIGHT)
            ctx.fill()

        return frame.to_rgb()

    # ------------------------------------------------------------------
    # Drone data per frame
    # ------------------------------------------------------------------

    def _get_drones(self, lf: int):
        """Return (slider_value, positions_list, headings_list) for frame."""
        if lf < _P_SLIDER100[0]:
            # Phase 1: steady at 5
            return 5.0, self._positions_5, self._headings_5

        if lf < _P_SLIDER500[0]:
            # Phase 2: 5 -> 100
            t = ease_out_cubic(clamp(self.phase_progress(lf, *_P_SLIDER100)))
            val = lerp(5, 100, t)
            n_show = max(5, int(val))
            return val, self._positions_100[:n_show], self._headings_100[:n_show]

        # Phase 3+: 100 -> 500
        t = ease_out_cubic(clamp(self.phase_progress(lf, *_P_SLIDER500)))
        val = lerp(100, 500, t)
        n_show = max(100, int(val))
        return val, self._positions_500[:n_show], self._headings_500[:n_show]

    # ------------------------------------------------------------------
    # Map rendering
    # ------------------------------------------------------------------

    def _draw_map_phase(self, ctx: cairo.Context, lf: int,
                        slider_val: float,
                        positions: list, headings: list):
        """Tactical map with drones + slider widget + performance note."""
        tmap = self.tmap

        # Grid
        tmap.draw_grid(ctx, alpha=0.06)

        # Drones — fade in recently-appeared ones
        n_drones = len(positions)
        for i, (wx, wy) in enumerate(positions):
            # Per-drone fade-in: newest drones fade in over ~15 frames
            drone_alpha = 1.0
            if n_drones > 5:
                # Only fade in if the drone "just appeared" relative to slider
                fraction = (i + 1) / n_drones
                if fraction > 0.9:
                    # Newest 10% drones — subtle pop-in
                    drone_alpha = clamp(remap(fraction, 0.9, 1.0, 1.0, 0.4))

            heading = headings[i] if i < len(headings) else 0.0
            color = (*CYAN[:3], drone_alpha)

            # Skip glow for large swarms (performance)
            use_glow = n_drones <= 120
            size = 10 if n_drones <= 100 else max(4, 10 - n_drones // 100)
            tmap.draw_drone(ctx, wx, wy, heading, color, size=size, glow=use_glow)

        # Slider at top
        slider_widget(ctx, 100, 30, WIDTH - 200, 20,
                      slider_val, 500, color=CYAN, show_labels=True)

        # Slider title
        label(ctx, "FLEET SIZE", 100, 22, color=DIM_WHITE,
              font_size=FONT_SIZE_TINY, bold=True)

        # Architecture note — constant regardless of scale
        self._draw_arch_note(ctx, lf)

        # Performance metrics — constant at all scales
        self._draw_perf_metrics(ctx, lf)

    def _draw_arch_note(self, ctx: cairo.Context, lf: int):
        """Small note showing O(log N) consensus is architecture-invariant."""
        alpha = 0.6 + 0.15 * math.sin(lf * 0.05)
        ctx.save()
        ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(FONT_SIZE_SMALL)
        ctx.set_source_rgba(*CYAN[:3], alpha)
        ctx.move_to(WIDTH - 300, HEIGHT - 60)
        ctx.show_text("O(log N) CONSENSUS")
        ctx.restore()

    def _draw_perf_metrics(self, ctx: cairo.Context, lf: int):
        """Draw constant performance metrics panel (bottom-right)."""
        ctx.save()
        x, y = WIDTH - 300, HEIGHT - 40
        ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(FONT_SIZE_TINY)
        ctx.set_source_rgba(*GREEN[:3], 0.6)
        ctx.move_to(x, y)
        ctx.show_text("< 200ms latency  |  ~20 bytes/msg")
        ctx.restore()

    # ------------------------------------------------------------------
    # Stats overlay
    # ------------------------------------------------------------------

    def _draw_stats(self, ctx: cairo.Context, progress: float):
        """Capability summary overlay fading in line by line."""
        # Semi-transparent background panel
        panel_alpha = smoothstep(clamp(progress * 3))
        ctx.save()
        ctx.set_source_rgba(0.02, 0.02, 0.06, 0.75 * panel_alpha)
        px, py, pw, ph = WIDTH // 2 - 380, HEIGHT // 2 - 100, 760, 220
        ctx.rectangle(px, py, pw, ph)
        ctx.fill()
        # Border
        ctx.set_source_rgba(*CYAN[:3], 0.3 * panel_alpha)
        ctx.set_line_width(1)
        ctx.rectangle(px, py, pw, ph)
        ctx.stroke()
        ctx.restore()

        # Lines appear one by one
        n_lines = len(self._stats_lines)
        for i, line in enumerate(self._stats_lines):
            # Each line starts at progress i/n, fully visible at (i+1)/n
            line_start = i / n_lines
            line_end = (i + 0.6) / n_lines
            line_alpha = smoothstep(clamp(remap(progress, line_start, line_end)))
            if line_alpha < 0.01:
                continue

            ly = HEIGHT // 2 - 60 + i * 42
            ctx.save()
            ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                                 cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(FONT_SIZE_MEDIUM)
            # Center horizontally
            ext = ctx.text_extents(line)
            lx = WIDTH / 2 - ext.x_advance / 2
            ctx.set_source_rgba(*CYAN[:3], 0.9 * line_alpha)
            ctx.move_to(lx, ly)
            ctx.show_text(line)
            ctx.restore()

    # ------------------------------------------------------------------
    # Logo phase
    # ------------------------------------------------------------------

    def _draw_logo_phase(self, ctx: cairo.Context, lf: int,
                         p_logo: float, in_fade: bool):
        """Particles converge to STRIX text, then crisp logo appears."""
        dt = 1.0 / FPS

        # Initialize particles once at logo phase start
        if not self._particles_initialized and p_logo >= 0:
            self._init_logo_particles()
            self._particles_initialized = True

        if self._particles_initialized:
            # Convergence: ramp up attract_strength over the phase
            converge_t = clamp(p_logo)
            strength = lerp(20, 300, ease_out_cubic(converge_t))
            drag = lerp(0.96, 0.90, converge_t)

            # Attractor = center of screen (particles find their individual
            # targets via grouped attraction — we use the center as a global
            # attractor plus manual per-target nudging in _update_logo_particles)
            self._update_logo_particles(dt, converge_t, strength, drag)

            # Draw particles with glow
            glow_alpha = 1.0
            if in_fade:
                p_fade = self.phase_progress(lf, *_P_FADE)
                glow_alpha = 1.0 - ease_out_cubic(clamp(p_fade))
            self.particles.draw_with_glow(ctx, alpha_mul=glow_alpha)

            # Once well converged, overlay crisp STRIX logo
            if converge_t > 0.7:
                logo_alpha = smoothstep(remap(converge_t, 0.7, 1.0))
                if in_fade:
                    p_fade = self.phase_progress(lf, *_P_FADE)
                    logo_alpha *= 1.0 - ease_out_cubic(clamp(p_fade))
                strix_logo(ctx, WIDTH / 2, HEIGHT / 2, font_size=72,
                           color=CYAN, alpha=logo_alpha)

    def _init_logo_particles(self):
        """Emit particles spread across the screen, one per target position."""
        self.particles.clear()
        n_targets = len(self._logo_targets)

        # Emit more particles than targets for visual richness
        # First batch: one particle per letter pixel target
        for tx, ty in self._logo_targets:
            # Start from random screen position
            sx = random.uniform(50, WIDTH - 50)
            sy = random.uniform(50, HEIGHT - 50)
            self.particles.emit_point(
                pos=(sx, sy),
                vel=(random.uniform(-30, 30), random.uniform(-30, 30)),
                count=1,
                life=20.0,  # long life — they live through the whole phase
                color=(*CYAN[:3], 0.9),
                size=3.0,
                spread=0,
                vel_spread=10,
            )

        # Extra ambient particles for sparkle
        extra = min(400, self.particles.max - self.particles.count)
        for _ in range(extra):
            sx = random.uniform(0, WIDTH)
            sy = random.uniform(0, HEIGHT)
            self.particles.emit_point(
                pos=(sx, sy),
                vel=(random.uniform(-20, 20), random.uniform(-20, 20)),
                count=1,
                life=18.0,
                color=(*CYAN[:3], 0.4),
                size=1.5,
                spread=0,
                vel_spread=5,
            )

    def _update_logo_particles(self, dt: float, converge_t: float,
                               strength: float, drag: float):
        """Move particles toward their target positions in the STRIX text."""
        n = self.particles.count
        if n == 0:
            return

        n_targets = len(self._logo_targets)

        # For the first n_targets particles, attract each toward its
        # specific target position. For the rest, attract toward center.
        alive = min(n, n_targets)
        for i in range(alive):
            if i >= n:
                break
            tx, ty = self._logo_targets[i]
            dx = tx - self.particles.pos[i, 0]
            dy = ty - self.particles.pos[i, 1]
            dist = math.sqrt(dx * dx + dy * dy) + 1.0
            force = strength * dt / max(dist * 0.02, 1.0)
            self.particles.vel[i, 0] += (dx / dist) * force
            self.particles.vel[i, 1] += (dy / dist) * force

        # Ambient particles attract to center
        center = np.array([WIDTH / 2, HEIGHT / 2])
        if n > n_targets:
            ambient_slice = slice(n_targets, n)
            delta = center - self.particles.pos[ambient_slice]
            dist = np.linalg.norm(delta, axis=1, keepdims=True)
            dist = np.maximum(dist, 5.0)
            force = delta / dist * strength * 0.3 * dt
            self.particles.vel[ambient_slice] += force

        # Velocity damping
        self.particles.vel[:n] *= drag

        # Position update
        self.particles.pos[:n] += self.particles.vel[:n] * dt

        # Keep particles alive (reset life decay)
        self.particles.life[:n] = np.maximum(self.particles.life[:n], 0.5)
