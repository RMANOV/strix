"""Scene 3: Loss & Recovery — drone destroyed, auction, re-formation.

Frames 0–1800 local (global 2700–4500, 60 seconds @ 30 FPS).

Phases:
  0–180    Calm patrol: 4 drones in tight formation, tactical map
  180–240  Hit!: drone-2 explosion + screen flash
  240–360  Kill zone: pulsing red marker at loss location
  360–540  Auction: slide-in bid panel with animated numbers
  540–660  Re-form: surviving drones avoid kill zone, timer 0.87s
  660–900  Reserve: 5th drone launches from base
  900–1350 AA detect: enemy AA position, confidence gauge fills
  1350–1800 Stable: new formation, metrics overlay
"""

import math
import numpy as np

from config import (
    WIDTH, HEIGHT, FPS, CYAN, GREEN, RED, WHITE, DIM_WHITE, AMBER,
    BID_WEIGHTS, BASE_POS, SAM_ZONE, FONT_SIZE_MEDIUM, FONT_SIZE_SMALL,
)
from scenes.base import BaseScene
from renderer.cairo_surface import CairoFrame
from renderer.map_renderer import TacticalMap
from renderer.particles import ParticleSystem
from renderer.shapes import (
    drone_icon, drone_glow, explosion_ring, screen_flash,
    bid_panel, kill_zone_marker, confidence_gauge, vignette,
)
from renderer.text_renderer import label, glow_text, digit_roll
from renderer.easing import (
    ease_out_cubic, ease_out_expo, ease_in_out_cubic, smoothstep,
    clamp, lerp, remap,
)


# ── Patrol waypoints (world coords) ────────────────────────────────
# 4 drones in a diamond patrol pattern, looping
_PATROL_CENTER = (2400.0, 1500.0)
_PATROL_RADIUS = 300.0

def _patrol_pos(drone_idx: int, t: float) -> tuple:
    """Diamond formation orbiting patrol center. t in [0, 1) is loop phase."""
    offsets = [
        (0.0, -1.0),   # D-00  north
        (1.0, 0.0),    # D-01  east  ← this one gets hit
        (0.0, 1.0),    # D-02  south
        (-1.0, 0.0),   # D-03  west
    ]
    # Gentle circular orbit of the formation center
    orbit_angle = t * 2 * math.pi
    cx = _PATROL_CENTER[0] + math.cos(orbit_angle) * 150
    cy = _PATROL_CENTER[1] + math.sin(orbit_angle) * 100
    dx, dy = offsets[drone_idx]
    return (cx + dx * _PATROL_RADIUS, cy + dy * _PATROL_RADIUS)


# Kill zone (where drone-1 was destroyed)
_KILL_WX, _KILL_WY = 2700.0, 1500.0
_KILL_RADIUS_M = 350.0

# Enemy AA position (detected in phase 7)
_AA_WX, _AA_WY = 2900.0, 1200.0

# Bid values for re-auction
_BID_VALUES = {
    "urgency": 8.0,
    "capability": 7.0,
    "proximity": 6.0,
    "energy": 9.0,
    "risk": 3.0,
}

# Reserve drone starts from base
_RESERVE_BASE = BASE_POS  # (200, 2600)

# Post-recovery formation (avoids kill zone, shifted west)
_SAFE_FORMATION = [
    (2000.0, 1200.0),  # D-00
    # D-01 is dead
    (2000.0, 1800.0),  # D-02
    (1700.0, 1500.0),  # D-03
    (2300.0, 1500.0),  # D-04 (reserve)
]


class Scene3Loss(BaseScene):
    """Loss & Recovery scene: explosion → auction → re-form → reserve."""

    def __init__(self):
        super().__init__()
        self.tmap = TacticalMap()  # full-screen map
        self.particles = ParticleSystem()
        self._explosion_emitted = False

        # Drone state: 5 drones total (index 1 dies at frame 180)
        # [wx, wy, heading, alive, battery, altitude, speed]
        self.drones = [
            {"id": 0, "alive": True, "battery": 0.82, "alt": 120, "spd": 14.0},
            {"id": 1, "alive": True, "battery": 0.75, "alt": 120, "spd": 14.0},
            {"id": 2, "alive": True, "battery": 0.88, "alt": 120, "spd": 14.0},
            {"id": 3, "alive": True, "battery": 0.79, "alt": 120, "spd": 14.0},
            {"id": 4, "alive": True, "battery": 1.00, "alt": 0,   "spd": 0.0},  # reserve
        ]
        # Drone positions (world coords) — updated each frame
        self.drone_pos = [(0.0, 0.0)] * 5
        self.drone_heading = [0.0] * 5

    # ── Drone position logic ──────────────────────────────

    def _update_drone_positions(self, lf: int):
        """Compute world positions for all drones at local frame lf."""
        dt = 1.0 / FPS

        if lf < 180:
            # Phase 1: calm patrol — all 4 in formation orbit
            orbit_t = (lf / 180.0) * 0.3  # slow orbit, ~30% of a loop
            for i in range(4):
                self.drone_pos[i] = _patrol_pos(i, orbit_t)
                # Heading: tangent to orbit
                next_t = orbit_t + 0.01
                nx, ny = _patrol_pos(i, next_t)
                cx, cy = self.drone_pos[i]
                self.drone_heading[i] = math.atan2(ny - cy, nx - cx)

        elif lf < 240:
            # Phase 2: Hit — drone 1 frozen at kill position, others scatter slightly
            t_hit = (lf - 180) / 60.0
            orbit_t = 0.3  # frozen at hit moment
            for i in range(4):
                if i == 1:
                    self.drone_pos[i] = (_KILL_WX, _KILL_WY)
                else:
                    base = _patrol_pos(i, orbit_t)
                    # Small scatter reaction
                    scatter = min(t_hit * 2, 1.0)
                    dx = (base[0] - _KILL_WX) * scatter * 0.15
                    dy = (base[1] - _KILL_WY) * scatter * 0.15
                    self.drone_pos[i] = (base[0] + dx, base[1] + dy)
                    self.drone_heading[i] = math.atan2(dy, dx)

        elif lf < 540:
            # Phases 3–4: kill zone visible, auction — drones hold position
            orbit_t = 0.3
            for i in range(4):
                if i == 1:
                    continue  # dead
                base = _patrol_pos(i, orbit_t)
                offset_x = (base[0] - _KILL_WX) * 0.15
                offset_y = (base[1] - _KILL_WY) * 0.15
                self.drone_pos[i] = (base[0] + offset_x, base[1] + offset_y)

        elif lf < 660:
            # Phase 5: Re-form — smooth transition to safe formation
            t = ease_in_out_cubic(clamp((lf - 540) / 120.0))
            orbit_t = 0.3
            safe_idx = 0
            for i in range(4):
                if i == 1:
                    continue
                base = _patrol_pos(i, orbit_t)
                offset_x = (base[0] - _KILL_WX) * 0.15
                offset_y = (base[1] - _KILL_WY) * 0.15
                start = (base[0] + offset_x, base[1] + offset_y)
                target = _SAFE_FORMATION[safe_idx]
                wx = lerp(start[0], target[0], t)
                wy = lerp(start[1], target[1], t)
                self.drone_pos[i] = (wx, wy)
                self.drone_heading[i] = math.atan2(
                    target[1] - start[1], target[0] - start[0]
                )
                safe_idx += 1

        elif lf < 900:
            # Phase 6: Reserve launch — existing drones hold, drone 4 flies from base
            for i in range(4):
                if i == 1:
                    continue
                safe_idx = [0, -1, 1, 2][i]
                self.drone_pos[i] = _SAFE_FORMATION[safe_idx]

            # Reserve drone: lerp from base to formation slot 3
            t = ease_out_cubic(clamp((lf - 660) / 240.0))
            bx, by = float(_RESERVE_BASE[0]), float(_RESERVE_BASE[1])
            tx, ty = _SAFE_FORMATION[3]
            self.drone_pos[4] = (lerp(bx, tx, t), lerp(by, ty, t))
            self.drone_heading[4] = math.atan2(ty - by, tx - bx)
            self.drones[4]["alt"] = lerp(0, 120, min(t * 1.5, 1.0))
            self.drones[4]["spd"] = lerp(0, 18, min(t * 1.2, 1.0))

        else:
            # Phases 7–8: all 4 active drones in safe formation
            safe_map = {0: 0, 2: 1, 3: 2, 4: 3}
            for i, si in safe_map.items():
                self.drone_pos[i] = _SAFE_FORMATION[si]

            # Gentle breathing motion
            breath = math.sin(lf * 0.02) * 15
            for i in safe_map:
                wx, wy = self.drone_pos[i]
                self.drone_pos[i] = (wx + breath * (0.5 if i % 2 == 0 else -0.5),
                                     wy + breath * (0.3 if i < 3 else -0.3))

    # ── Main render ───────────────────────────────────────

    def render_frame(self, local_frame: int, total_frames: int,
                     global_frame: int = 0) -> np.ndarray:
        if not self._initialized:
            self.setup()
            self._initialized = True

        lf = local_frame
        frame = self.new_frame()
        ctx = frame.ctx

        # Update drone alive state
        if lf >= 180:
            self.drones[1]["alive"] = False

        # Update positions
        self._update_drone_positions(lf)

        # ── Tactical map background ───────────────────
        self.tmap.draw_grid(ctx)
        self.tmap.draw_sam_zone(ctx, pulse=lf * 0.05)
        self.tmap.draw_base(ctx)

        # Kill zone (visible after hit)
        if lf >= 240:
            self.tmap.draw_kill_zone(
                ctx, _KILL_WX, _KILL_WY, _KILL_RADIUS_M, pulse=lf * 0.1
            )

        # ── Draw drone paths (before hit: patrol circle hint) ─────
        if lf < 180:
            # Faint orbit path
            orbit_pts = [_patrol_pos(0, t / 36.0) for t in range(37)]
            self.tmap.draw_path(ctx, orbit_pts, color=CYAN, alpha=0.1, dashed=True)

        # ── Draw drones ───────────────────────────────
        active_indices = self._active_drone_indices(lf)
        for i in active_indices:
            d = self.drones[i]
            if not d["alive"]:
                continue
            wx, wy = self.drone_pos[i]
            color = CYAN if i != 4 else GREEN
            self.tmap.draw_drone(ctx, wx, wy, self.drone_heading[i],
                                 color=color, size=14)
            # Sensor fan for active patrol drones
            if lf < 180 or lf >= 660:
                self.tmap.draw_sector_fan(
                    ctx, wx, wy, self.drone_heading[i],
                    arc=math.pi / 4, range_m=400, color=color, alpha=0.05
                )

        # ── Phase-specific rendering ──────────────────

        # Phase 2: Explosion (180–240)
        if self.in_phase(lf, 180, 300):
            self._render_explosion(ctx, lf)

        # Phase 3: Kill zone label (240–360)
        if self.in_phase(lf, 240, 360):
            kpx, kpy = self.tmap.w2p(_KILL_WX, _KILL_WY)
            alpha = smoothstep(clamp((lf - 240) / 30.0))
            label(ctx, "KILL ZONE", kpx - 30, kpy - self.tmap.w2p_dist(_KILL_RADIUS_M) - 10,
                  color=(*RED[:3], alpha * 0.8), font_size=13, bold=True)

        # Phase 4: Auction panel (360–540)
        if self.in_phase(lf, 340, 600):
            self._render_auction(ctx, lf)

        # Phase 5: Re-form timer (540–660)
        if self.in_phase(lf, 540, 700):
            self._render_reform_timer(ctx, lf)

        # Phase 6: Reserve launch label (660–900)
        if self.in_phase(lf, 660, 900):
            self._render_reserve_launch(ctx, lf)

        # Phase 7: AA detection (900–1350)
        if self.in_phase(lf, 900, 1350):
            self._render_aa_detection(ctx, lf)

        # Phase 8: Stable metrics (1350–1800)
        if lf >= 1350:
            self._render_stable_overlay(ctx, lf)

        # ── Particles ─────────────────────────────────
        self.particles.update(1.0 / FPS, drag=0.96, gravity=15.0)
        self.particles.draw_with_glow(ctx)

        # ── Screen flash (at moment of hit) ───────────
        if self.in_phase(lf, 180, 195):
            flash_alpha = 0.5 * (1.0 - (lf - 180) / 15.0)
            screen_flash(ctx, flash_alpha, color=WHITE)

        # ── Drone status panels ───────────────────────
        self._render_drone_panels(ctx, lf)

        # ── HUD overlay ──────────────────────────────
        self.hud.draw_full(ctx, lf, global_frame)
        self.hud.draw_scene_label(ctx, "SCENE 3 — LOSS & RECOVERY")

        # ── Vignette ──────────────────────────────────
        vignette(ctx, strength=0.5)

        return frame.to_rgb()

    # ── Active drone indices helper ───────────────────

    def _active_drone_indices(self, lf: int) -> list:
        """Which drone indices should be rendered at this frame."""
        if lf < 180:
            return [0, 1, 2, 3]
        elif lf < 660:
            return [0, 2, 3]  # drone 1 dead, reserve not yet launched
        else:
            return [0, 2, 3, 4]  # reserve joined

    # ── Phase renderers ───────────────────────────────

    def _render_explosion(self, ctx, lf: int):
        """Explosion ring + debris particles at drone-1's death location."""
        t = clamp((lf - 180) / 60.0)  # 0→1 over 60 frames

        px, py = self.tmap.w2p(_KILL_WX, _KILL_WY)
        max_r = self.tmap.w2p_dist(250)

        # Expanding ring
        explosion_ring(ctx, px, py, ease_out_expo(t), max_radius=max_r, color=RED)

        # Secondary ring (delayed)
        if lf >= 190:
            t2 = clamp((lf - 190) / 50.0)
            explosion_ring(ctx, px, py, ease_out_expo(t2),
                           max_radius=max_r * 0.6, color=AMBER)

        # Emit debris particles (once at hit frame)
        if not self._explosion_emitted:
            self.particles.emit_ring(
                center=(px, py), count=200, life=2.5,
                color=(*RED[:3], 0.9), size=3.0, speed=180.0
            )
            self.particles.emit_ring(
                center=(px, py), count=80, life=1.8,
                color=(1.0, 0.8, 0.2, 0.8), size=2.0, speed=120.0
            )
            self._explosion_emitted = True

    def _render_auction(self, ctx, lf: int):
        """Slide-in bid panel from right edge."""
        panel_w, panel_h = 310, 240
        # Slide in from off-screen right
        slide_t = ease_out_cubic(clamp((lf - 340) / 45.0))
        # Slide out at end
        if lf >= 560:
            slide_t *= 1.0 - ease_out_cubic(clamp((lf - 560) / 30.0))

        panel_x = WIDTH - panel_w * slide_t
        panel_y = 80

        # Bid value animation progress (ramps up during visible phase)
        bid_progress = ease_in_out_cubic(clamp((lf - 380) / 120.0))

        bid_panel(ctx, panel_x, panel_y, panel_w, panel_h,
                  _BID_VALUES, bid_progress, color=CYAN)

        # "WINNER: D-03" label after auction completes
        if lf >= 480 and slide_t > 0.5:
            win_alpha = smoothstep(clamp((lf - 480) / 30.0)) * slide_t
            label(ctx, "WINNER: D-03 → SECTOR LEAD",
                  panel_x + 12, panel_y + panel_h + 24,
                  color=(*GREEN[:3], win_alpha), font_size=14, bold=True)

    def _render_reform_timer(self, ctx, lf: int):
        """Show re-allocation timer counting to 0.87s."""
        t = ease_out_expo(clamp((lf - 540) / 60.0))
        timer_val = 0.87 * t

        # Timer position: center-bottom area
        tx = WIDTH // 2 - 60
        ty = HEIGHT - 160

        self.hud.draw_timer(ctx, tx, ty, timer_val, color=CYAN,
                            label_text="RE-ALLOCATION TIME")

        # Formation path hints (dashed lines showing re-route)
        if lf < 660:
            reform_t = clamp((lf - 540) / 120.0)
            safe_idx = 0
            for i in [0, 2, 3]:
                wx, wy = self.drone_pos[i]
                target = _SAFE_FORMATION[safe_idx]
                # Draw path from current to target
                self.tmap.draw_path(
                    ctx, [(wx, wy), target],
                    color=CYAN, alpha=0.25 * (1.0 - reform_t), dashed=True
                )
                safe_idx += 1

    def _render_reserve_launch(self, ctx, lf: int):
        """Reserve drone launch visualization."""
        t = clamp((lf - 660) / 240.0)

        # "RESERVE LAUNCH" label near base
        base_px, base_py = self.tmap.w2p(*_RESERVE_BASE)
        appear = smoothstep(clamp((lf - 660) / 30.0))
        fade = 1.0 - smoothstep(clamp((lf - 850) / 50.0))
        alpha = appear * fade

        label(ctx, "RESERVE D-04 LAUNCHING", base_px + 20, base_py - 20,
              color=(*GREEN[:3], alpha * 0.8), font_size=13, bold=True)

        # Trajectory line from base to target
        trail_alpha = 0.2 * alpha
        target = _SAFE_FORMATION[3]
        self.tmap.draw_path(
            ctx, [_RESERVE_BASE, target],
            color=GREEN, alpha=trail_alpha, dashed=True
        )

        # Altitude indicator
        alt = self.drones[4]["alt"]
        if alt > 0 and alpha > 0.1:
            d4_px, d4_py = self.tmap.w2p(*self.drone_pos[4])
            label(ctx, f"ALT {alt:.0f}m", d4_px + 18, d4_py - 8,
                  color=(*GREEN[:3], alpha * 0.6), font_size=11)

    def _render_aa_detection(self, ctx, lf: int):
        """Enemy AA detection with scanning pattern and confidence gauge."""
        t = clamp((lf - 900) / 450.0)

        aa_px, aa_py = self.tmap.w2p(_AA_WX, _AA_WY)

        # Scanning pulse rings (cyan, from detecting drones)
        scan_phase = (lf - 900) * 0.08
        for i, didx in enumerate([0, 2, 3]):
            if not self.drones[didx]["alive"]:
                continue
            dwx, dwy = self.drone_pos[didx]
            dpx, dpy = self.tmap.w2p(dwx, dwy)
            # Expanding scan ring toward AA
            ring_t = ((scan_phase + i * 0.7) % 3.0) / 3.0
            if ring_t < 1.0:
                ring_r = ring_t * self.tmap.w2p_dist(600)
                ring_a = 0.15 * (1.0 - ring_t)
                ctx.set_source_rgba(*CYAN[:3], ring_a)
                ctx.set_line_width(1.5)
                ctx.arc(dpx, dpy, ring_r, 0, 2 * math.pi)
                ctx.stroke()

        # Enemy AA marker (red diamond, pulsing)
        if t > 0.15:
            reveal = smoothstep(clamp((t - 0.15) / 0.3))
            pulse_a = 0.4 + 0.3 * math.sin(lf * 0.12)

            # Red pulsing circle at AA position
            ctx.save()
            ctx.set_source_rgba(*RED[:3], reveal * pulse_a * 0.15)
            ctx.arc(aa_px, aa_py, self.tmap.w2p_dist(150), 0, 2 * math.pi)
            ctx.fill()
            ctx.set_source_rgba(*RED[:3], reveal * pulse_a)
            ctx.set_line_width(2)
            ctx.arc(aa_px, aa_py, self.tmap.w2p_dist(150), 0, 2 * math.pi)
            ctx.stroke()
            ctx.restore()

            # "ENEMY AA" label
            label(ctx, "ENEMY AA", aa_px + self.tmap.w2p_dist(160), aa_py,
                  color=(*RED[:3], reveal * 0.7), font_size=13, bold=True)

            # Draw enemy vehicle icon
            self.tmap.draw_enemy_vehicle(ctx, _AA_WX, _AA_WY,
                                         heading=math.pi * 0.25, color=RED)

        # Confidence gauge (bottom-left area)
        gauge_value = ease_out_cubic(t) * 0.78  # fills to 78%
        gauge_cx = 140
        gauge_cy = HEIGHT - 200
        confidence_gauge(ctx, gauge_cx, gauge_cy, radius=50,
                         value=gauge_value, color=CYAN,
                         label_text="AA CONFIDENCE")

        # Detection data readout
        if t > 0.3:
            data_alpha = smoothstep(clamp((t - 0.3) / 0.2))
            data_x = 80
            data_y = HEIGHT - 120
            lines = [
                f"TYPE:  S-300 VARIANT",
                f"RANGE: ~25km",
                f"CONF:  {gauge_value * 100:.0f}%",
            ]
            for i, line in enumerate(lines):
                label(ctx, line, data_x, data_y + i * 18,
                      color=(*CYAN[:3], data_alpha * 0.6), font_size=12)

    def _render_stable_overlay(self, ctx, lf: int):
        """Final stable phase — formation + metrics."""
        t = smoothstep(clamp((lf - 1350) / 60.0))

        # Stats overlay (top-left area, below scene label)
        stats = [
            "SWARM STATUS: NOMINAL",
            "ACTIVE: 4/5 DRONES",
            f"REALLOC TIME: 0.87s",
            f"THREAT MAPPED: 78%",
        ]
        alpha = t * 0.85
        stats_x = 30
        stats_y = 80
        ctx.select_font_face("monospace", 0, 1)
        ctx.set_font_size(FONT_SIZE_SMALL)
        for i, s in enumerate(stats):
            color = GREEN if "NOMINAL" in s else CYAN
            ctx.set_source_rgba(*color[:3], alpha)
            ctx.move_to(stats_x, stats_y + i * 28)
            ctx.show_text(s)

        # "NO SINGLE POINT OF FAILURE" glow text
        if lf >= 1500:
            glow_t = smoothstep(clamp((lf - 1500) / 60.0))
            glow_text(ctx, "NO SINGLE POINT OF FAILURE",
                      WIDTH // 2 - 250, HEIGHT // 2 + 30,
                      color=(*CYAN[:3], glow_t * 0.9),
                      font_size=28, glow_alpha=glow_t * 0.2)

    # ── Drone status panels ───────────────────────────

    def _render_drone_panels(self, ctx, lf: int):
        """Per-drone status readouts along the right edge."""
        panel_x = WIDTH - 180
        panel_y_start = 300
        panel_spacing = 95

        visible_drones = self._panel_drone_indices(lf)
        for slot, didx in enumerate(visible_drones):
            d = self.drones[didx]
            py = panel_y_start + slot * panel_spacing

            # Determine regime label
            if not d["alive"]:
                regime = "LOST"
            elif lf < 180:
                regime = "PATROL"
            elif lf < 540:
                regime = "EVADE" if didx != 1 else "LOST"
            elif lf < 660:
                regime = "REFORM"
            elif didx == 4 and lf < 900:
                regime = "LAUNCH"
            elif lf < 1350:
                regime = "DETECT"
            else:
                regime = "RECON"

            # Battery drain: slow decay over scene
            batt = d["battery"]
            if d["alive"] and lf > 0:
                batt = max(0.1, batt - lf * 0.00003)

            self.hud.draw_drone_panel(
                ctx, panel_x, py,
                drone_id=d["id"],
                battery=batt,
                altitude=d["alt"],
                speed=d["spd"],
                regime=regime,
                alive=d["alive"],
            )

    def _panel_drone_indices(self, lf: int) -> list:
        """Which drones to show panels for."""
        if lf < 660:
            return [0, 1, 2, 3]
        else:
            return [0, 1, 2, 3, 4]
