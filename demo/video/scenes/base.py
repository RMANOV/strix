"""Base scene abstract class."""

from abc import ABC, abstractmethod
import numpy as np

from config import WIDTH, HEIGHT, BG
from renderer.cairo_surface import CairoFrame
from renderer.hud import HUD
from renderer.easing import clamp


class BaseScene(ABC):
    """Abstract base for all demo scenes.

    Each scene instance persists across frames, allowing stateful
    animation (particle systems, counters, etc.).
    """

    def __init__(self):
        self.hud = HUD()
        self._initialized = False

    def setup(self):
        """Called once before the first frame. Override for init."""
        pass

    @abstractmethod
    def render_frame(self, local_frame: int, total_frames: int,
                     global_frame: int = 0) -> np.ndarray:
        """Render one frame → RGB numpy array (H, W, 3) uint8.

        Args:
            local_frame: frame index within this scene (0-based).
            total_frames: total frames in this scene.
            global_frame: absolute frame index in the video.
        """
        pass

    def progress(self, local_frame: int, total_frames: int) -> float:
        """Scene progress 0.0 → 1.0."""
        return clamp(local_frame / max(total_frames - 1, 1))

    def phase_progress(self, local_frame: int,
                       phase_start: int, phase_end: int) -> float:
        """Progress within a sub-phase. <0 = not started, >1 = done."""
        if local_frame < phase_start:
            return -1.0
        if local_frame >= phase_end:
            return 1.0
        return (local_frame - phase_start) / max(phase_end - phase_start - 1, 1)

    def in_phase(self, local_frame: int,
                 phase_start: int, phase_end: int) -> bool:
        return phase_start <= local_frame < phase_end

    def new_frame(self) -> CairoFrame:
        """Create a fresh frame filled with background color."""
        frame = CairoFrame(WIDTH, HEIGHT)
        frame.fill_bg()
        return frame
