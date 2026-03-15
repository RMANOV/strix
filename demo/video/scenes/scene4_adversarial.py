"""Scene 4: Adversarial Prediction — enemy intent classification & preemptive repositioning.

Local frames 0–1800 (global 4500–6300, 60 seconds at 30 fps).

Phases:
  0–300    Enemy appears: red particle cloud forms around enemy vehicle
  300–900  Intent split: cloud forks into 3 branches (DEFENDING/ATTACKING/RETREATING)
  900–1440 Prediction: ATTACKING branch dominates, ghost position ahead of enemy
  1440–1800 Preempt: swarm repositions, countdown 47→15s, confidence 87%
"""

import cairo
import math
import numpy as np

from config import (
    WIDTH, HEIGHT, FPS, CYAN, GREEN, RED, BLUE, YELLOW, AMBER,
    DIM_WHITE, WHITE, SAM_ZONE, FONT_SIZE_MEDIUM, FONT_SIZE_SMALL,
    FONT_SIZE_TINY,
)
from scenes.base import BaseScene
from renderer.map_renderer import TacticalMap
from renderer.particles import ParticleSystem
from renderer.shapes import vignette, confidence_gauge
from renderer.text_renderer import glow_text, label
from renderer.easing import ease_out_cubic, smoothstep, lerp, clamp


# ── Constants ───────────────────────────────────────────────────────
TOTAL = 1800

# Phase boundaries (local frames)
PH_APPEAR_START, PH_APPEAR_END = 0, 300
PH_SPLIT_START, PH_SPLIT_END = 300, 900
PH_PREDICT_START, PH_PREDICT_END = 900, 1440
PH_PREEMPT_START, PH_PREEMPT_END = 1440, 1800

# Enemy path: world-coord waypoints (slow traverse left→right across map)
ENEMY_PATH = [
    (800, 1200),
    (1200, 1100),
    (1600, 1050),
    (2000, 1000),
    (2400, 950),
    (2800, 900),
]

# Drone starting positions (4 drones, carried from scene 3)
DRONE_HOME = [
    (500, 2000),
    (700, 1800),
    (900, 2200),
    (600, 2400),
]

# Preempt target positions — drones surround predicted enemy path
DRONE_INTERCEPT = [
    (2600, 700),
    (2800, 1100),
    (2400, 1200),
    (2200, 800),
]

# Intent branch target offsets (relative to enemy pos, world coords)
INTENT_DEFEND_OFFSET = (0, 0)        # clusters near enemy
INTENT_ATTACK_OFFSET = (600, -300)   # toward friendly positions
INTENT_RETREAT_OFFSET = (-500, 200)  # away from engagement

# Colors for intent branches
COLOR_DEFEND = (*BLUE[:3], 0.85)
COLOR_ATTACK = (*RED[:3], 0.85)
COLOR_RETREAT = (*YELLOW[:3], 0.85)


def _interp_path(path: list[tuple], t: float) -> tuple[float, float]:
    """Interpolate along a polyline. t: 0→1."""
    t = clamp(t)
    if len(path) < 2:
        return path[0]
    seg_count = len(path) - 1
    idx = t * seg_count
    i = int(idx)
    if i >= seg_count:
        return path[-1]
    frac = idx - i
    ax, ay = path[i]
    bx, by = path[i + 1]
    return (ax + (bx - ax) * frac, ay + (by - ay) * frac)


def _heading_on_path(path: list[tuple], t: float) -> float:
    """Tangent angle along the path."""
    t0 = clamp(t - 0.01)
    t1 = clamp(t + 0.01)
    x0, y0 = _interp_path(path, t0)
    x1, y1 = _interp_path(path, t1)
    return math.atan2(y1 - y0, x1 - x0)


class Scene4Adversarial(BaseScene):
    """Adversarial intent prediction and preemptive swarm repositioning."""

    def __init__(self):
        super().__init__()
        self.tmap = TacticalMap()

        # Three particle systems for intent branches (150 particles each)
        self.ps_defend = ParticleSystem(max_particles=150)
        self.ps_attack = ParticleSystem(max_particles=150)
        self.ps_retreat = ParticleSystem(max_particles=150)

        # Single red cloud for initial appearance phase
        self.ps_enemy_cloud = ParticleSystem(max_particles=200)

        # Emission cooldown trackers
        self._emit_tick = 0

    # ── Enemy position at given local frame ──────────────────────
    def _enemy_pos(self, local_frame: int) -> tuple[float, float]:
        """Enemy traverses path slowly over entire scene."""
        t = clamp(local_frame / TOTAL)
        return _interp_path(ENEMY_PATH, t * 0.8)  # only traverse ~80% of path

    def _enemy_heading(self, local_frame: int) -> float:
        t = clamp(local_frame / TOTAL)
        return _heading_on_path(ENEMY_PATH, t * 0.8)

    # ── Drone positions (lerp from home to intercept during preempt) ─
    def _drone_pos(self, idx: int, local_frame: int) -> tuple[float, float]:
        if local_frame < PH_PREEMPT_START:
            return DRONE_HOME[idx]
        t = smoothstep(clamp(
            (local_frame - PH_PREEMPT_START) / (PH_PREEMPT_END - PH_PREEMPT_START)
        ))
        hx, hy = DRONE_HOME[idx]
        ix, iy = DRONE_INTERCEPT[idx]
        return (lerp(hx, ix, t), lerp(hy, iy, t))

    def _drone_heading(self, idx: int, local_frame: int) -> float:
        """Point drones toward predicted enemy position during preempt."""
        dx, dy = self._drone_pos(idx, local_frame)
        if local_frame >= PH_PREEMPT_START:
            ex, ey = self._predicted_enemy_pos(local_frame)
        else:
            ex, ey = self._enemy_pos(local_frame)
        return math.atan2(ey - dy, ex - dx)

    def _predicted_enemy_pos(self, local_frame: int) -> tuple[float, float]:
        """Ghost position: enemy's future predicted position (ahead of actual)."""
        # Predict ~120 frames ahead
        future_frame = min(local_frame + 120, TOTAL)
        t = clamp(future_frame / TOTAL)
        return _interp_path(ENEMY_PATH, t * 0.8)

    # ── Particle emission ────────────────────────────────────────
    def _emit_particles(self, local_frame: int):
        """Manage particle emission across phases."""
        ex, ey = self._enemy_pos(local_frame)
        epx, epy = self.tmap.w2p(ex, ey)

        # Phase 1: enemy appears — red cloud around enemy
        if PH_APPEAR_START <= local_frame < PH_APPEAR_END:
            t = self.phase_progress(local_frame, PH_APPEAR_START, PH_APPEAR_END)
            if self._emit_tick % 2 == 0:
                count = int(3 + 5 * t)
                self.ps_enemy_cloud.emit_cluster(
                    (epx, epy), radius=30 + 20 * t, count=count,
                    life=1.5, color=(*RED[:3], 0.6), size=2.5,
                )

        # Phase 2: intent split — gradually fork into 3 colored clouds
        if PH_SPLIT_START <= local_frame < PH_SPLIT_END:
            t = self.phase_progress(local_frame, PH_SPLIT_START, PH_SPLIT_END)
            split_t = ease_out_cubic(t)

            # Compute intent target positions (pixel coords)
            def_wx = ex + INTENT_DEFEND_OFFSET[0] * split_t
            def_wy = ey + INTENT_DEFEND_OFFSET[1] * split_t
            atk_wx = ex + INTENT_ATTACK_OFFSET[0] * split_t
            atk_wy = ey + INTENT_ATTACK_OFFSET[1] * split_t
            ret_wx = ex + INTENT_RETREAT_OFFSET[0] * split_t
            ret_wy = ey + INTENT_RETREAT_OFFSET[1] * split_t

            def_px, def_py = self.tmap.w2p(def_wx, def_wy)
            atk_px, atk_py = self.tmap.w2p(atk_wx, atk_wy)
            ret_px, ret_py = self.tmap.w2p(ret_wx, ret_wy)

            if self._emit_tick % 3 == 0:
                # DEFENDING (blue) — stays near enemy
                self.ps_defend.emit_cluster(
                    (def_px, def_py), radius=20 + 15 * split_t, count=4,
                    life=1.2, color=COLOR_DEFEND, size=2.0,
                )
                # ATTACKING (red) — spreads toward friendlies
                self.ps_attack.emit_cluster(
                    (atk_px, atk_py), radius=20 + 20 * split_t, count=4,
                    life=1.2, color=COLOR_ATTACK, size=2.2,
                )
                # RETREATING (yellow) — spreads away
                self.ps_retreat.emit_cluster(
                    (ret_px, ret_py), radius=20 + 15 * split_t, count=4,
                    life=1.2, color=COLOR_RETREAT, size=2.0,
                )

            # Fade the initial enemy cloud
            # (let it decay naturally, stop emitting)

        # Phase 3: prediction — ATTACKING grows, others fade
        if PH_PREDICT_START <= local_frame < PH_PREDICT_END:
            t = self.phase_progress(local_frame, PH_PREDICT_START, PH_PREDICT_END)

            atk_wx = ex + INTENT_ATTACK_OFFSET[0]
            atk_wy = ey + INTENT_ATTACK_OFFSET[1]
            atk_px, atk_py = self.tmap.w2p(atk_wx, atk_wy)

            # ATTACKING cloud: more particles, brighter
            if self._emit_tick % 2 == 0:
                self.ps_attack.emit_cluster(
                    (atk_px, atk_py), radius=35, count=6,
                    life=1.5, color=(*RED[:3], 0.9), size=2.8,
                )

            # DEFENDING & RETREATING: fewer emissions → natural fade
            if self._emit_tick % 8 == 0 and t < 0.5:
                def_px, def_py = self.tmap.w2p(
                    ex + INTENT_DEFEND_OFFSET[0], ey + INTENT_DEFEND_OFFSET[1]
                )
                ret_px, ret_py = self.tmap.w2p(
                    ex + INTENT_RETREAT_OFFSET[0], ey + INTENT_RETREAT_OFFSET[1]
                )
                self.ps_defend.emit_cluster(
                    (def_px, def_py), radius=15, count=2,
                    life=0.8, color=COLOR_DEFEND, size=1.5,
                )
                self.ps_retreat.emit_cluster(
                    (ret_px, ret_py), radius=15, count=2,
                    life=0.8, color=COLOR_RETREAT, size=1.5,
                )

        # Phase 4: preempt — attacking cloud follows predicted path
        if PH_PREEMPT_START <= local_frame < PH_PREEMPT_END:
            pred_x, pred_y = self._predicted_enemy_pos(local_frame)
            pred_px, pred_py = self.tmap.w2p(pred_x, pred_y)
            if self._emit_tick % 3 == 0:
                self.ps_attack.emit_cluster(
                    (pred_px, pred_py), radius=25, count=4,
                    life=1.0, color=(*RED[:3], 0.7), size=2.2,
                )

    # ── Main render ──────────────────────────────────────────────
    def render_frame(self, local_frame: int, total_frames: int,
                     global_frame: int = 0) -> np.ndarray:
        if not self._initialized:
            self.setup()
            self._initialized = True

        frame = self.new_frame()
        ctx = frame.ctx
        dt = 1.0 / FPS

        # ── Emit particles ───────────────────────────
        self._emit_particles(local_frame)
        self._emit_tick += 1

        # ── Update particle physics ──────────────────
        ex, ey = self._enemy_pos(local_frame)
        epx, epy = self.tmap.w2p(ex, ey)

        self.ps_enemy_cloud.update(dt, drag=0.96, attractor=(epx, epy),
                                   attract_strength=40.0)
        self.ps_defend.update(dt, drag=0.95)
        self.ps_attack.update(dt, drag=0.95)
        self.ps_retreat.update(dt, drag=0.95)

        # ── Tactical map base layer ──────────────────
        pulse = local_frame * 0.05
        self.tmap.draw_grid(ctx)
        self.tmap.draw_sam_zone(ctx, pulse)
        self.tmap.draw_base(ctx)

        # ── Friendly drones ──────────────────────────
        for i in range(4):
            dwx, dwy = self._drone_pos(i, local_frame)
            dheading = self._drone_heading(i, local_frame)
            self.tmap.draw_drone(ctx, dwx, dwy, heading=dheading,
                                 color=CYAN, size=10, glow=True)

            # Sensor fans (during preempt, oriented toward prediction)
            if local_frame >= PH_PREEMPT_START:
                self.tmap.draw_sector_fan(
                    ctx, dwx, dwy, heading=dheading,
                    arc=math.pi / 3, range_m=400, color=CYAN, alpha=0.06,
                )

        # ── Enemy vehicle ────────────────────────────
        eheading = self._enemy_heading(local_frame)

        # Fade-in for initial appearance
        if local_frame < PH_APPEAR_END:
            appear_t = smoothstep(
                self.phase_progress(local_frame, PH_APPEAR_START, PH_APPEAR_END)
            )
            enemy_alpha = appear_t
        else:
            enemy_alpha = 1.0

        if enemy_alpha > 0.05:
            ctx.save()
            # Slight red glow behind enemy
            epx_d, epy_d = self.tmap.w2p(ex, ey)
            pat = cairo.RadialGradient(
                epx_d, epy_d, 0, epx_d, epy_d, 25)
            pat.add_color_stop_rgba(0, *RED[:3], 0.15 * enemy_alpha)
            pat.add_color_stop_rgba(1, *RED[:3], 0.0)
            ctx.set_source(pat)
            ctx.arc(epx_d, epy_d, 25, 0, 2 * math.pi)
            ctx.fill()
            ctx.restore()

            self.tmap.draw_enemy_vehicle(ctx, ex, ey, heading=eheading,
                                         color=(*RED[:3], enemy_alpha))

        # ── Ghost outline (predicted future position) ─
        if PH_PREDICT_START <= local_frame:
            pred_wx, pred_wy = self._predicted_enemy_pos(local_frame)
            pred_px, pred_py = self.tmap.w2p(pred_wx, pred_wy)

            if local_frame < PH_PREDICT_END:
                ghost_alpha = smoothstep(self.phase_progress(
                    local_frame, PH_PREDICT_START, PH_PREDICT_END
                )) * 0.5
            else:
                ghost_alpha = 0.5

            # Dashed outline rectangle at predicted position
            ctx.save()
            ctx.translate(pred_px, pred_py)
            ctx.rotate(eheading)
            s = 10
            ctx.set_dash([4, 4])
            ctx.set_line_width(1.5)
            ctx.set_source_rgba(*AMBER[:3], ghost_alpha)
            ctx.rectangle(-s, -s * 0.6, s * 2, s * 1.2)
            ctx.stroke()
            ctx.restore()

            # "PREDICTED" label near ghost
            label(ctx, "PREDICTED", pred_px + 16, pred_py - 10,
                  color=(*AMBER[:3], ghost_alpha * 0.8),
                  font_size=FONT_SIZE_TINY, bold=True)

            # Connecting dashed line between actual and predicted
            ctx.save()
            actual_px, actual_py = self.tmap.w2p(ex, ey)
            ctx.set_dash([3, 5])
            ctx.set_line_width(1)
            ctx.set_source_rgba(*AMBER[:3], ghost_alpha * 0.4)
            ctx.move_to(actual_px, actual_py)
            ctx.line_to(pred_px, pred_py)
            ctx.stroke()
            ctx.restore()

        # ── Particle rendering ───────────────────────
        # Enemy cloud (phase 1 primarily, decays naturally)
        if self.ps_enemy_cloud.count > 0:
            self.ps_enemy_cloud.draw_with_glow(ctx, alpha_mul=0.8)

        # Intent branches
        # Compute alpha multipliers based on phase
        defend_alpha = 1.0
        attack_alpha = 1.0
        retreat_alpha = 1.0

        if local_frame >= PH_PREDICT_START:
            fade_t = smoothstep(self.phase_progress(
                local_frame, PH_PREDICT_START, PH_PREDICT_END
            ))
            defend_alpha = max(0.1, 1.0 - fade_t * 0.85)
            retreat_alpha = max(0.1, 1.0 - fade_t * 0.85)
            attack_alpha = min(1.5, 1.0 + fade_t * 0.5)  # grow stronger

        if self.ps_defend.count > 0:
            self.ps_defend.draw_with_glow(ctx, alpha_mul=defend_alpha)
        if self.ps_attack.count > 0:
            self.ps_attack.draw_with_glow(ctx, alpha_mul=min(attack_alpha, 1.0))
        if self.ps_retreat.count > 0:
            self.ps_retreat.draw_with_glow(ctx, alpha_mul=retreat_alpha)

        # ── Intent labels ────────────────────────────
        if PH_SPLIT_START <= local_frame:
            split_t = ease_out_cubic(clamp(self.phase_progress(
                local_frame, PH_SPLIT_START, PH_SPLIT_END
            )))

            # Compute label positions
            def_wx = ex + INTENT_DEFEND_OFFSET[0] * split_t
            def_wy = ey + INTENT_DEFEND_OFFSET[1] * split_t
            atk_wx = ex + INTENT_ATTACK_OFFSET[0] * split_t
            atk_wy = ey + INTENT_ATTACK_OFFSET[1] * split_t
            ret_wx = ex + INTENT_RETREAT_OFFSET[0] * split_t
            ret_wy = ey + INTENT_RETREAT_OFFSET[1] * split_t

            def_px, def_py = self.tmap.w2p(def_wx, def_wy)
            atk_px, atk_py = self.tmap.w2p(atk_wx, atk_wy)
            ret_px, ret_py = self.tmap.w2p(ret_wx, ret_wy)

            label_alpha = min(split_t * 2, 1.0)

            # Pulsing glow for ATTACKING label during prediction phase
            atk_glow = 0.3
            if local_frame >= PH_PREDICT_START:
                atk_glow = 0.3 + 0.3 * (0.5 + 0.5 * math.sin(local_frame * 0.1))

            glow_text(ctx, "DEFENDING", def_px - 40, def_py - 25,
                      color=(*BLUE[:3], label_alpha * defend_alpha),
                      font_size=FONT_SIZE_SMALL, glow_alpha=0.2)

            glow_text(ctx, "ATTACKING", atk_px - 40, atk_py - 25,
                      color=(*RED[:3], label_alpha * min(attack_alpha, 1.0)),
                      font_size=FONT_SIZE_SMALL, glow_alpha=atk_glow)

            glow_text(ctx, "RETREATING", ret_px - 50, ret_py - 25,
                      color=(*YELLOW[:3], label_alpha * retreat_alpha),
                      font_size=FONT_SIZE_SMALL, glow_alpha=0.2)

            # Probability percentages next to labels during prediction
            if local_frame >= PH_PREDICT_START:
                pred_t = smoothstep(self.phase_progress(
                    local_frame, PH_PREDICT_START, PH_PREDICT_END
                ))
                atk_pct = lerp(33, 78, pred_t)
                def_pct = lerp(33, 14, pred_t)
                ret_pct = lerp(33, 8, pred_t)

                label(ctx, f"{atk_pct:.0f}%", atk_px + 60, atk_py - 25,
                      color=(*RED[:3], 0.9), font_size=FONT_SIZE_TINY, bold=True)
                label(ctx, f"{def_pct:.0f}%", def_px + 60, def_py - 25,
                      color=(*BLUE[:3], 0.5 * defend_alpha),
                      font_size=FONT_SIZE_TINY, bold=True)
                label(ctx, f"{ret_pct:.0f}%", ret_px + 70, ret_py - 25,
                      color=(*YELLOW[:3], 0.5 * retreat_alpha),
                      font_size=FONT_SIZE_TINY, bold=True)

        # ── Preempt HUD elements ─────────────────────
        if local_frame >= PH_PREEMPT_START:
            preempt_t = smoothstep(self.phase_progress(
                local_frame, PH_PREEMPT_START, PH_PREEMPT_END
            ))

            # Countdown timer: 47s → 15s
            timer_val = lerp(47.0, 15.0, preempt_t)
            timer_color = AMBER if timer_val > 25 else RED
            self.hud.draw_timer(ctx, 60, HEIGHT - 120, timer_val,
                                color=timer_color, label_text="INTERCEPT ETA")

            # Confidence gauge: fill to 87%
            conf_val = lerp(0.0, 0.87, ease_out_cubic(preempt_t))
            gauge_color = CYAN if conf_val < 0.7 else GREEN
            confidence_gauge(ctx, WIDTH - 120, HEIGHT - 140, radius=50,
                             value=conf_val, color=gauge_color,
                             label_text="CONFIDENCE")

            # Reposition status label
            if preempt_t < 0.8:
                glow_text(ctx, "REPOSITIONING", WIDTH // 2 - 100, 80,
                          color=AMBER, font_size=FONT_SIZE_MEDIUM,
                          glow_alpha=0.15 + 0.15 * math.sin(local_frame * 0.15))
            else:
                glow_text(ctx, "INTERCEPT READY", WIDTH // 2 - 110, 80,
                          color=GREEN, font_size=FONT_SIZE_MEDIUM, glow_alpha=0.3)

            # Drone path trails to intercept positions
            for i in range(4):
                self.tmap.draw_path(
                    ctx,
                    [DRONE_HOME[i], DRONE_INTERCEPT[i]],
                    color=CYAN, alpha=0.15, dashed=True,
                )

        # ── Scene label ──────────────────────────────
        self.hud.draw_scene_label(ctx, "SCENE 4 — ADVERSARIAL PREDICTION")

        # ── HUD overlay ──────────────────────────────
        self.hud.draw_corners(ctx)
        self.hud.draw_scanlines(ctx, local_frame)
        self.hud.draw_mission_time(ctx, global_frame)
        self.hud.draw_coords(ctx)

        # ── Vignette ─────────────────────────────────
        vignette(ctx)

        return frame.to_rgb()
