"""Synthetic images so the tool can be tried without hunting for a dataset."""

from __future__ import annotations

import random
from pathlib import Path

SHAPES = ("rect", "ellipse", "triangle")
COLORS = ("#e05252", "#4c9aff", "#34d399", "#fbbf24", "#a78bfa")


def make_demo_images(target: Path, count: int = 6, size: int = 640, seed: int = 7) -> int:
    """Draw ``count`` images of coloured shapes on noise-free backgrounds."""
    from PIL import Image, ImageDraw

    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for index in range(count):
        image = Image.new("RGB", (size, int(size * 0.75)), "#1b2230")
        draw = ImageDraw.Draw(image)
        for _ in range(rng.randint(2, 5)):
            width, height = image.size
            x1 = rng.randint(0, width - 120)
            y1 = rng.randint(0, height - 120)
            x2 = x1 + rng.randint(60, 180)
            y2 = y1 + rng.randint(60, 140)
            color = rng.choice(COLORS)
            shape = rng.choice(SHAPES)
            if shape == "rect":
                draw.rectangle([x1, y1, x2, y2], fill=color)
            elif shape == "ellipse":
                draw.ellipse([x1, y1, x2, y2], fill=color)
            else:
                draw.polygon([(x1, y2), ((x1 + x2) / 2, y1), (x2, y2)], fill=color)
        image.save(target / f"demo_{index:02d}.png")
    return count
