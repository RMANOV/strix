"""Scene 1: Radio Command — frames 0-900 (30 seconds).

Phases:
  0-90     Fade in: dark map + cyan grid
  90-360   Radio waveform + typewriter sergeant command
  360-510  STRIX confirmation + mission parameters
  510-900  4 drones deploy from base, fan out with particle trails
"""

import math
import numpy as np
from scipy.interpolate import CubicSpline

from config import (
    WIDTH, HEIGHT, FPS, CYAN, GREEN, RED, DIM_WHITE, AMBER,
    SAM_ZONE, BASE_POS, FONT_SIZE_MEDIUM, FONT_SIZE_SMALL,
)
from scenes.base import BaseScene
from renderer.cairo_surface import CairoFrame
from renderer.map_renderer import TacticalMap
from renderer.text_renderer import typewriter, multiline_typewriter, label
from renderer.shapes import radio_waveform, vignette, drone_icon, drone_glow
from renderer.particles import ParticleSystem
from renderer.easing import (
    ease_out_cubic, ease_in_out_cubic, smoothstep, clamp, lerp,
)


# ── Text lines ───────────────────────────────────────────────────────────────

RADIO_LINES = [
    "STRIX, I need eyes on the ridge north of",
    "checkpoint Charlie. Avoid the SAM corridor.",
    "Report in 10 minutes.",
]

STRIX_LINES = [
    "COPY. DEPLOYING 4x RECON.",
    "SAM CORRIDOR MARKED.",
    "ETA: 47 SECONDS.",
]


# ── Drone path waypoints (world coords) ──────────────────────────────────────
# Each path starts at BASE_POS and fans out to a different area,
# routing around the SAM corridor (x 1600-2200, y 800-2200).

_B = BASE_POS  # (200, 2600)

DRONE_WAYPOINTS = [
    # Drone 0 — sweeps north-west, stays left of SAM
    [_B, (400, 2200), (600, 1600), (800, 1000), (1000, 500), (1400, 300)],
    # Drone 1 — goes north through center-left, curves east above SAM
    [_B, (500, 2100), (900, 1500), (1200, 800), (1600, 400), (2200, 300)],
    # Drone 2 — loops far east, passes below SAM then rises right of it
    [_B, (600, 2400), (1200, 2500), (2400, 2400), (3000, 1800), (3200, 1000)],
    # Drone 3 — far east sweep, upper right quadrant
    [_B, (800, 2500), (1500, 2600), (2600, 2300), (3400, 1400), (3700, 600)],
]

DRONE_COLORS = [CYAN, GREEN, (*CYAN[:3], 1.0), (*GREEN[:3], 1.0)]


def _build_spline(waypoints: list[tuple], n_samples: int = 500):
    """Build a CubicSpline for smooth drone path interpolation.

    Returns callable(t) -> (x, y) where t in [0, 1].
    """
    pts = np.array(waypoints, dtype=np.float64)
    # Cumulative chord-length parameterization
    dists = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cum = np.concatenate([[0], np.cumsum(dists)])
    cum /= cum[-1]  # normalize to [0, 1]

    spline_x = CubicSpline(cum, pts[:, 0], bc_type="clamped")
    spline_y = CubicSpline(cum, pts[:, 1], bc_type="clamped")

    def interp(t: float):
        t = float(np.clip(t, 0, 1))
        return float(spline_x(t)), float(spline_y(t))

    return interp


class Scene1Radio(BaseScene):
    """Scene 1: Radio Command — sergeant orders, STRIX confirms, drones deploy."""

    def setup(self):
        self.tac_map = TacticalMap()
        self.particles = ParticleSystem()
        self.drone_splines = [_build_spline(wp) for wp in DRONE_WAYPOINTS]
        # Stagger launch: drone i starts moving at frame offset i * 30
        self.launch_stagger = 30  # frames between successive launches
        self._initialized = True

    def render_frame(self, local_frame: int, total_frames: int,
                     global_frame: int = 0) -> np.ndarray:
        if not self._initialized:
            self.setup()

        frame = self.new_frame()
        ctx = frame.ctx

        # ── Phase boundaries ──────────────────────────────────────────────
        PH_FADE_START, PH_FADE_END = 0, 90
        PH_RADIO_START, PH_RADIO_END = 90, 360
        PH_CONFIRM_START, PH_CONFIRM_END = 360, 510
        PH_DEPLOY_START, PH_DEPLOY_END = 510, 900

        # ── Master fade-in alpha ──────────────────────────────────────────
        if local_frame < PH_FADE_END:
            master_alpha = smoothstep(self.phase_progress(
                local_frame, PH_FADE_START, PH_FADE_END))
        else:
            master_alpha = 1.0

        # ── Map background ────────────────────────────────────────────────
        self.tac_map.draw_grid(ctx, alpha=0.08 * master_alpha)
        self.tac_map.draw_sam_zone(ctx, pulse=local_frame * 0.05)
        self.tac_map.draw_base(ctx)

        # ── Phase: Radio waveform + typewriter ────────────────────────────
        if self.in_phase(local_frame, PH_RADIO_START, PH_RADIO_END):
            p = self.phase_progress(local_frame, PH_RADIO_START, PH_RADIO_END)

            # Waveform — centered top area
            wf_x, wf_y, wf_w, wf_h = 460, 80, 1000, 100
            amp = 0.3 + 0.7 * math.sin(math.pi * p)  # swell then fade
            radio_waveform(ctx, wf_x, wf_y, wf_w, wf_h,
                           phase=local_frame * 0.15, amplitude=amp)

            # "SGT" label
            label(ctx, "SGT KOVACS // CH-7 ENCRYPTED", wf_x, wf_y - 8,
                  color=(*AMBER[:3], 0.6), font_size=12, bold=True)

            # Typewriter text — below waveform
            text_x, text_y = 480, 220
            multiline_typewriter(ctx, RADIO_LINES, text_x, text_y,
                                 progress=ease_out_cubic(p),
                                 color=(*AMBER[:3], 0.9),
                                 font_size=FONT_SIZE_MEDIUM,
                                 line_spacing=1.6)

        # Show radio text completed (persist after phase)
        if local_frame >= PH_RADIO_END:
            text_x, text_y = 480, 220
            multiline_typewriter(ctx, RADIO_LINES, text_x, text_y,
                                 progress=1.0,
                                 color=(*AMBER[:3], 0.5),
                                 font_size=FONT_SIZE_MEDIUM,
                                 line_spacing=1.6)

        # ── Phase: STRIX confirmation ─────────────────────────────────────
        if self.in_phase(local_frame, PH_CONFIRM_START, PH_CONFIRM_END):
            p = self.phase_progress(
                local_frame, PH_CONFIRM_START, PH_CONFIRM_END)

            # STRIX label
            label(ctx, "STRIX // AUTONOMOUS RESPONSE", 480, 370,
                  color=(*CYAN[:3], 0.7), font_size=12, bold=True)

            # Confirmation typewriter
            multiline_typewriter(ctx, STRIX_LINES, 480, 395,
                                 progress=ease_out_cubic(p),
                                 color=CYAN,
                                 font_size=FONT_SIZE_MEDIUM,
                                 line_spacing=1.6)

            # Mission params box (slides in)
            box_alpha = smoothstep(clamp((p - 0.4) / 0.3))
            if box_alpha > 0.01:
                bx, by, bw, bh = 480, 520, 420, 130
                ctx.set_source_rgba(0.02, 0.02, 0.06, 0.75 * box_alpha)
                ctx.rectangle(bx, by, bw, bh)
                ctx.fill()
                ctx.set_source_rgba(*CYAN[:3], 0.3 * box_alpha)
                ctx.set_line_width(1)
                ctx.rectangle(bx, by, bw, bh)
                ctx.stroke()

                params = [
                    "MISSION    : RECON-7741",
                    "ASSETS     : 4x MICRO-UAV",
                    "AVOID      : SAM CORRIDOR (1600-2200E)",
                    "OBJECTIVE  : RIDGE NORTH OF CP CHARLIE",
                    "TIME LIMIT : 10:00",
                ]
                for i, line in enumerate(params):
                    line_alpha = smoothstep(
                        clamp((p - 0.45 - i * 0.08) / 0.15)) * box_alpha
                    if line_alpha > 0.01:
                        label(ctx, line, bx + 12, by + 22 + i * 22,
                              color=(*DIM_WHITE[:3], 0.8 * line_alpha),
                              font_size=14)

        # Persist STRIX text after confirm phase
        if local_frame >= PH_CONFIRM_END:
            # Dim the confirmation text
            label(ctx, "STRIX // AUTONOMOUS RESPONSE", 480, 370,
                  color=(*CYAN[:3], 0.35), font_size=12, bold=True)
            multiline_typewriter(ctx, STRIX_LINES, 480, 395,
                                 progress=1.0,
                                 color=(*CYAN[:3], 0.35),
                                 font_size=FONT_SIZE_MEDIUM,
                                 line_spacing=1.6)

        # ── Phase: Deploy drones ──────────────────────────────────────────
        if self.in_phase(local_frame, PH_DEPLOY_START, PH_DEPLOY_END):
            deploy_frame = local_frame - PH_DEPLOY_START
            deploy_duration = PH_DEPLOY_END - PH_DEPLOY_START  # 390 frames

            # "DEPLOYING" label with pulsing dot
            blink = (local_frame % 20) < 12
            deploy_label = "DEPLOYING..." if blink else "DEPLOYING"
            label(ctx, deploy_label, 30, 80,
                  color=(*GREEN[:3], 0.8), font_size=FONT_SIZE_SMALL,
                  bold=True)

            for drone_idx in range(4):
                # Stagger: each drone launches a bit later
                drone_start = drone_idx * self.launch_stagger
                drone_local = deploy_frame - drone_start
                if drone_local < 0:
                    continue

                # Drone progress along its spline
                # Reserve some frames for the drone to travel its full path
                flight_frames = deploy_duration - drone_start
                t = ease_in_out_cubic(
                    clamp(drone_local / max(flight_frames, 1)))

                spline = self.drone_splines[drone_idx]
                wx, wy = spline(t)

                # Heading: tangent direction
                t_next = min(t + 0.01, 1.0)
                wx2, wy2 = spline(t_next)
                heading = math.atan2(wy2 - wy, wx2 - wx)

                # Color: alternate cyan / green
                drone_color = CYAN if drone_idx % 2 == 0 else GREEN

                # Draw path (already traversed portion)
                path_pts = []
                n_trail = 40
                for k in range(n_trail + 1):
                    tk = t * k / n_trail
                    path_pts.append(spline(tk))
                self.tac_map.draw_path(ctx, path_pts, color=drone_color,
                                       alpha=0.2, dashed=True)

                # Draw drone
                self.tac_map.draw_drone(ctx, wx, wy, heading,
                                        color=drone_color, glow=True)

                # Drone ID label
                px, py = self.tac_map.w2p(wx, wy)
                label(ctx, f"D-{drone_idx + 1:02d}", px + 16, py - 8,
                      color=(*drone_color[:3], 0.6), font_size=10)

                # Emit particles at drone position (screen coords)
                if drone_local % 2 == 0:
                    self.particles.emit_point(
                        pos=(px, py),
                        vel=(-math.cos(heading) * 15, -math.sin(heading) * 15),
                        count=3,
                        life=1.2,
                        color=(*drone_color[:3], 0.5),
                        size=2.0,
                        spread=3.0,
                        vel_spread=8.0,
                    )

            # Update and draw particles
            dt = 1.0 / FPS
            self.particles.update(dt, drag=0.96)
            self.particles.draw_with_glow(ctx, alpha_mul=0.7)

        # ── HUD overlay ───────────────────────────────────────────────────
        self.hud.draw_full(ctx, local_frame, global_frame)
        self.hud.draw_scene_label(ctx, "SCENE 1 // RADIO COMMAND",
                                  alpha=0.3 * master_alpha)

        # ── Vignette ──────────────────────────────────────────────────────
        vignette(ctx, strength=0.5 * master_alpha)

        # ── Global fade-in for first phase ────────────────────────────────
        if master_alpha < 1.0:
            # Darken the whole frame proportionally
            ctx.set_source_rgba(0, 0, 0, 1.0 - master_alpha)
            ctx.rectangle(0, 0, WIDTH, HEIGHT)
            ctx.fill()

        return frame.to_rgb()
