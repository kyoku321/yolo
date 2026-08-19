"""Visualization: draw recognition boxes + labels on a shelf image."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .pipeline import Recognition

MATCH_COLOR = (34, 197, 94)      # green
UNKNOWN_COLOR = (239, 68, 68)    # red


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_recognitions(image: Image.Image,
                      results: list[Recognition]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    w, _ = canvas.size
    font = _font(max(14, w // 90))
    line = max(2, w // 500)

    for r in results:
        color = MATCH_COLOR if r.is_match else UNKNOWN_COLOR
        x1, y1, x2, y2 = r.box.xyxy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line)

        label = r.label
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), label, fill=(255, 255, 255), font=font)

    return canvas


def summary(results: list[Recognition]) -> str:
    total = len(results)
    matched = sum(1 for r in results if r.is_match)
    lines = [f"Detected {total} products · matched {matched} · "
             f"unknown {total - matched}", ""]
    counts: dict[str, int] = {}
    for r in results:
        if r.is_match:
            counts[r.name] = counts.get(r.name, 0) + 1
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"• {name}: {n}")
    return "\n".join(lines)
