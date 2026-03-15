"""Numpy-vectorized particle system for trails, explosions, clouds."""

import numpy as np
import cairo
import math

from config import MAX_PARTICLES


class ParticleSystem:
    """Struct-of-arrays particle system. All physics is vectorized numpy."""

    def __init__(self, max_particles: int = MAX_PARTICLES):
        self.max = max_particles
        self.count = 0
        # Struct of arrays
        self.pos = np.zeros((max_particles, 2), dtype=np.float64)
        self.vel = np.zeros((max_particles, 2), dtype=np.float64)
        self.life = np.zeros(max_particles, dtype=np.float64)
        self.max_life = np.ones(max_particles, dtype=np.float64)
        self.color = np.zeros((max_particles, 4), dtype=np.float64)  # RGBA
        self.size = np.ones(max_particles, dtype=np.float64) * 2.0

    def _alloc(self, n: int) -> slice:
        """Reserve `n` particle slots, return slice."""
        start = self.count
        end = min(start + n, self.max)
        self.count = end
        return slice(start, end)

    def emit_point(self, pos: tuple, vel: tuple, count: int,
                   life: float, color: tuple, size: float = 2.0,
                   spread: float = 0.0, vel_spread: float = 0.0):
        """Emit particles from a single point."""
        s = self._alloc(count)
        n = s.stop - s.start
        if n <= 0:
            return
        self.pos[s] = pos
        self.vel[s] = vel
        if spread > 0:
            self.pos[s] += np.random.randn(n, 2) * spread
        if vel_spread > 0:
            self.vel[s] += np.random.randn(n, 2) * vel_spread
        self.life[s] = 1.0
        self.max_life[s] = life + np.random.rand(n) * life * 0.3
        self.color[s] = color
        self.size[s] = size + np.random.rand(n) * size * 0.3

    def emit_ring(self, center: tuple, count: int, life: float,
                  color: tuple, size: float = 2.0, speed: float = 100.0):
        """Emit particles in an expanding ring (explosion pattern)."""
        s = self._alloc(count)
        n = s.stop - s.start
        if n <= 0:
            return
        angles = np.random.uniform(0, 2 * np.pi, n)
        self.pos[s, 0] = center[0]
        self.pos[s, 1] = center[1]
        spd = speed * (0.5 + np.random.rand(n) * 0.5)
        self.vel[s, 0] = np.cos(angles) * spd
        self.vel[s, 1] = np.sin(angles) * spd
        self.life[s] = 1.0
        self.max_life[s] = life + np.random.rand(n) * life * 0.4
        self.color[s] = color
        self.size[s] = size + np.random.rand(n) * size * 0.5

    def emit_cluster(self, center: tuple, radius: float, count: int,
                     life: float, color: tuple, size: float = 2.0):
        """Emit a cloud of particles within a circular region."""
        s = self._alloc(count)
        n = s.stop - s.start
        if n <= 0:
            return
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n)) * radius
        self.pos[s, 0] = center[0] + np.cos(angles) * radii
        self.pos[s, 1] = center[1] + np.sin(angles) * radii
        self.vel[s] = np.random.randn(n, 2) * 5.0
        self.life[s] = 1.0
        self.max_life[s] = life + np.random.rand(n) * life * 0.5
        self.color[s] = color
        self.size[s] = size + np.random.rand(n) * size * 0.3

    def emit_line(self, start: tuple, end: tuple, count: int,
                  life: float, color: tuple, size: float = 1.5):
        """Emit particles along a line segment."""
        s = self._alloc(count)
        n = s.stop - s.start
        if n <= 0:
            return
        t = np.random.uniform(0, 1, n)
        self.pos[s, 0] = start[0] + t * (end[0] - start[0])
        self.pos[s, 1] = start[1] + t * (end[1] - start[1])
        self.vel[s] = np.random.randn(n, 2) * 3.0
        self.life[s] = 1.0
        self.max_life[s] = life
        self.color[s] = color
        self.size[s] = size

    def update(self, dt: float, drag: float = 0.98,
               gravity: float = 0.0,
               attractor: tuple = None, attract_strength: float = 50.0,
               repeller: tuple = None, repel_strength: float = 30.0):
        """Physics step for all live particles."""
        if self.count == 0:
            return

        n = self.count
        alive = self.life[:n] > 0
        if not np.any(alive):
            self.count = 0
            return

        # Position update
        self.pos[:n][alive] += self.vel[:n][alive] * dt

        # Velocity damping
        self.vel[:n][alive] *= drag

        # Gravity (downward)
        if gravity != 0:
            self.vel[:n][alive, 1] += gravity * dt

        # Attractor
        if attractor is not None:
            delta = np.array(attractor) - self.pos[:n][alive]
            dist = np.linalg.norm(delta, axis=1, keepdims=True)
            dist = np.maximum(dist, 5.0)  # prevent singularity
            force = delta / dist * attract_strength * dt
            self.vel[:n][alive] += force

        # Repeller
        if repeller is not None:
            delta = self.pos[:n][alive] - np.array(repeller)
            dist = np.linalg.norm(delta, axis=1, keepdims=True)
            dist = np.maximum(dist, 5.0)
            force = delta / dist * repel_strength / (dist + 1) * dt
            self.vel[:n][alive] += force

        # Life decay
        self.life[:n] -= dt / self.max_life[:n]

        # Compact: remove dead particles
        alive_mask = self.life[:n] > 0
        alive_count = int(np.sum(alive_mask))
        if alive_count < n:
            self.pos[:alive_count] = self.pos[:n][alive_mask]
            self.vel[:alive_count] = self.vel[:n][alive_mask]
            self.life[:alive_count] = self.life[:n][alive_mask]
            self.max_life[:alive_count] = self.max_life[:n][alive_mask]
            self.color[:alive_count] = self.color[:n][alive_mask]
            self.size[:alive_count] = self.size[:n][alive_mask]
            self.count = alive_count

    def draw(self, ctx: cairo.Context, alpha_mul: float = 1.0):
        """Render particles as colored circles.

        Color-bucketed to minimize cairo state changes.
        Skips particles with alpha < 0.02.
        """
        if self.count == 0:
            return

        n = self.count
        # Effective alpha: base color alpha × life fraction × multiplier
        alphas = self.color[:n, 3] * self.life[:n] * alpha_mul
        visible = alphas > 0.02
        if not np.any(visible):
            return

        # Bucket by quantized color for fewer state changes
        vis_idx = np.where(visible)[0]
        # Quantize RGB to reduce unique colors
        qr = (self.color[vis_idx, 0] * 4).astype(int)
        qg = (self.color[vis_idx, 1] * 4).astype(int)
        qb = (self.color[vis_idx, 2] * 4).astype(int)
        color_keys = qr * 25 + qg * 5 + qb

        for key in np.unique(color_keys):
            bucket = vis_idx[color_keys == key]
            if len(bucket) == 0:
                continue
            # Use the first particle's RGB as representative
            r, g, b = self.color[bucket[0], :3]

            for i in bucket:
                a = float(alphas[i])
                s = float(self.size[i]) * float(self.life[i])
                if s < 0.3:
                    continue
                px, py = float(self.pos[i, 0]), float(self.pos[i, 1])
                ctx.set_source_rgba(r, g, b, a)
                ctx.arc(px, py, s, 0, 2 * math.pi)
                ctx.fill()

    def draw_with_glow(self, ctx: cairo.Context, alpha_mul: float = 1.0):
        """Draw particles with additive glow halos."""
        if self.count == 0:
            return

        n = self.count
        alphas = self.color[:n, 3] * self.life[:n] * alpha_mul
        visible = alphas > 0.02
        vis_idx = np.where(visible)[0]

        for i in vis_idx:
            px, py = float(self.pos[i, 0]), float(self.pos[i, 1])
            r, g, b = self.color[i, :3]
            a = float(alphas[i])
            s = float(self.size[i]) * float(self.life[i])
            if s < 0.3:
                continue

            # Glow halo
            pat = cairo.RadialGradient(px, py, 0, px, py, s * 3)
            pat.add_color_stop_rgba(0, r, g, b, a * 0.3)
            pat.add_color_stop_rgba(1, r, g, b, 0)
            ctx.set_source(pat)
            ctx.arc(px, py, s * 3, 0, 2 * math.pi)
            ctx.fill()

            # Core
            ctx.set_source_rgba(r, g, b, a)
            ctx.arc(px, py, s, 0, 2 * math.pi)
            ctx.fill()

    def clear(self):
        """Remove all particles."""
        self.count = 0
