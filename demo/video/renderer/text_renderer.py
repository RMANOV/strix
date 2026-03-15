"""Text rendering: typewriter, glow, digit roll, labels."""

import cairo
import math
from config import FONT_FAMILY, FONT_SIZE_MEDIUM, FONT_SIZE_SMALL, CYAN, WHITE, DIM_WHITE


def typewriter(ctx: cairo.Context, text: str, x: float, y: float,
               progress: float, color=CYAN, font_size: float = FONT_SIZE_MEDIUM,
               cursor: bool = True):
    """Reveal `text` character-by-character based on `progress` (0→1).

    Shows a blinking block cursor at the reveal edge.
    """
    ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(font_size)

    n_chars = max(1, int(len(text) * min(progress, 1.0)))
    visible = text[:n_chars]

    # Draw visible text
    ctx.set_source_rgba(*color)
    ctx.move_to(x, y)
    ctx.show_text(visible)

    # Blinking cursor
    if cursor and progress < 1.0:
        ext = ctx.text_extents(visible)
        cursor_x = x + ext.x_advance
        # Blink at ~3 Hz using progress as proxy for time
        blink = (int(progress * 60) % 2 == 0)
        if blink:
            ctx.rectangle(cursor_x + 2, y - font_size * 0.8,
                          font_size * 0.55, font_size * 0.9)
            ctx.set_source_rgba(*color[:3], 0.8)
            ctx.fill()


def glow_text(ctx: cairo.Context, text: str, x: float, y: float,
              color=CYAN, font_size: float = FONT_SIZE_MEDIUM,
              glow_alpha: float = 0.3, glow_spread: int = 3):
    """Draw text with a soft glow halo behind it.

    Renders multiple offset copies at low alpha (glow), then a crisp top layer.
    """
    ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(font_size)

    # Glow passes (offsets in a small grid)
    for dx in range(-glow_spread, glow_spread + 1):
        for dy in range(-glow_spread, glow_spread + 1):
            d = math.sqrt(dx * dx + dy * dy)
            if d > glow_spread:
                continue
            a = glow_alpha * (1 - d / (glow_spread + 1))
            ctx.set_source_rgba(*color[:3], a)
            ctx.move_to(x + dx, y + dy)
            ctx.show_text(text)

    # Crisp foreground
    ctx.set_source_rgba(*color)
    ctx.move_to(x, y)
    ctx.show_text(text)


def label(ctx: cairo.Context, text: str, x: float, y: float,
          color=DIM_WHITE, font_size: float = FONT_SIZE_SMALL,
          bold: bool = False, align: str = "left"):
    """Simple text label with optional alignment."""
    ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(font_size)
    ctx.set_source_rgba(*color)

    if align != "left":
        ext = ctx.text_extents(text)
        if align == "center":
            x -= ext.x_advance / 2
        elif align == "right":
            x -= ext.x_advance

    ctx.move_to(x, y)
    ctx.show_text(text)


def multiline_typewriter(ctx: cairo.Context, lines: list[str],
                         x: float, y: float, progress: float,
                         color=CYAN, font_size: float = FONT_SIZE_MEDIUM,
                         line_spacing: float = 1.5):
    """Typewriter across multiple lines, progressing sequentially."""
    total_chars = sum(len(l) for l in lines)
    if total_chars == 0:
        return

    chars_to_show = int(total_chars * min(progress, 1.0))
    shown = 0
    for i, line in enumerate(lines):
        line_y = y + i * font_size * line_spacing
        if shown >= chars_to_show:
            break
        remaining = chars_to_show - shown
        line_progress = min(remaining / max(len(line), 1), 1.0)
        is_last_line = (shown + len(line) >= chars_to_show)
        typewriter(ctx, line, x, line_y, line_progress, color, font_size,
                   cursor=is_last_line)
        shown += len(line)


def digit_roll(ctx: cairo.Context, value: float, target: float,
               x: float, y: float, progress: float,
               color=CYAN, font_size: float = FONT_SIZE_MEDIUM,
               fmt: str = "{:.1f}"):
    """Animate a number rolling from `value` to `target`."""
    current = value + (target - value) * min(progress, 1.0)
    text = fmt.format(current)
    ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(font_size)
    ctx.set_source_rgba(*color)
    ctx.move_to(x, y)
    ctx.show_text(text)


def mission_time_text(seconds: float) -> str:
    """Format seconds as MM:SS.ms mission clock."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    ms = int((seconds % 1) * 100)
    return f"T+{m:02d}:{s:02d}.{ms:02d}"
