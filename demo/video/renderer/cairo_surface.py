"""CairoFrame — cairo ImageSurface ↔ numpy bridge.

Provides layer-based compositing for the rendering pipeline:
  scene.render_frame() builds CairoFrame layers → composite → to_rgb() → PyAV
"""

import cairo
import numpy as np

from config import WIDTH, HEIGHT, BG


class CairoFrame:
    """Single ARGB32 frame backed by a cairo ImageSurface."""

    __slots__ = ("width", "height", "surface", "ctx")

    def __init__(self, width: int = WIDTH, height: int = HEIGHT):
        self.width = width
        self.height = height
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self.ctx = cairo.Context(self.surface)

    def clear(self, color=None):
        """Fill with solid color (default: transparent black)."""
        self.ctx.save()
        self.ctx.set_operator(cairo.OPERATOR_SOURCE)
        if color:
            self.ctx.set_source_rgba(*color)
        else:
            self.ctx.set_source_rgba(0, 0, 0, 0)
        self.ctx.paint()
        self.ctx.restore()

    def fill_bg(self):
        """Fill with the standard dark background."""
        self.clear(BG)

    def to_rgb(self) -> np.ndarray:
        """Convert to numpy RGB uint8 array (H, W, 3).

        Cairo ARGB32 on little-endian stores bytes as B, G, R, A.
        We extract channels [2, 1, 0] to get R, G, B.
        """
        self.surface.flush()
        buf = self.surface.get_data()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(
            self.height, self.width, 4
        )
        # BGRA → RGB (copy to own the data, since buf is a memoryview)
        return arr[:, :, [2, 1, 0]].copy()

    def composite_over(self, other: "CairoFrame"):
        """Paint `other` on top of this frame (OPERATOR_OVER)."""
        self.ctx.save()
        self.ctx.set_source_surface(other.surface, 0, 0)
        self.ctx.set_operator(cairo.OPERATOR_OVER)
        self.ctx.paint()
        self.ctx.restore()

    def composite_alpha(self, other: "CairoFrame", alpha: float):
        """Paint `other` on top with global opacity."""
        self.ctx.save()
        self.ctx.set_source_surface(other.surface, 0, 0)
        self.ctx.set_operator(cairo.OPERATOR_OVER)
        self.ctx.paint_with_alpha(alpha)
        self.ctx.restore()

    def set_clip_rect(self, x: float, y: float, w: float, h: float):
        """Set a rectangular clip region."""
        self.ctx.rectangle(x, y, w, h)
        self.ctx.clip()

    def reset_clip(self):
        self.ctx.reset_clip()

    @staticmethod
    def set_color(ctx: cairo.Context, rgba: tuple):
        """Helper: set source color from an RGBA tuple."""
        ctx.set_source_rgba(*rgba)
