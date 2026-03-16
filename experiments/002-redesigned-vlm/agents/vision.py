"""Vision utilities — render grids as images for VLM consumption."""

import base64
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Official ARC color palette
PALETTE = [
    (0xFF, 0xFF, 0xFF),  # 0 White
    (0xCC, 0xCC, 0xCC),  # 1 Off-white
    (0x99, 0x99, 0x99),  # 2 Neutral light
    (0x66, 0x66, 0x66),  # 3 Neutral
    (0x33, 0x33, 0x33),  # 4 Off-black
    (0x00, 0x00, 0x00),  # 5 Black
    (0xE5, 0x3A, 0xA3),  # 6 Magenta
    (0xFF, 0x7B, 0xCC),  # 7 Magenta light
    (0xF9, 0x3C, 0x31),  # 8 Red
    (0x1E, 0x93, 0xFF),  # 9 Blue
    (0x88, 0xD8, 0xF1),  # 10 Blue light
    (0xFF, 0xDC, 0x00),  # 11 Yellow
    (0xFF, 0x85, 0x1B),  # 12 Orange
    (0x92, 0x12, 0x31),  # 13 Maroon
    (0x4F, 0xCC, 0x30),  # 14 Green
    (0xA3, 0x56, 0xD6),  # 15 Purple
]

SCALE = 4


def grid_to_image(grid: list[list[int]]) -> Image.Image:
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    img = Image.new("RGB", (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            pixels[x, y] = PALETTE[grid[y][x] & 0xF]
    return img.resize((w * SCALE, h * SCALE), Image.NEAREST)


def make_side_by_side(before_grid: list[list[int]], after_grid: list[list[int]], label_before: str = "BEFORE", label_after: str = "AFTER") -> Image.Image:
    """Create a side-by-side comparison image with labels."""
    img_a = grid_to_image(before_grid)
    img_b = grid_to_image(after_grid)
    gap = 20
    total_w = img_a.width + gap + img_b.width
    total_h = img_a.height + 25  # room for labels
    canvas = Image.new("RGB", (total_w, total_h), (40, 40, 40))
    canvas.paste(img_a, (0, 25))
    canvas.paste(img_b, (img_a.width + gap, 25))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((img_a.width // 2 - 30, 4), label_before, fill=(255, 255, 255), font=font)
    draw.text((img_a.width + gap + img_b.width // 2 - 20, 4), label_after, fill=(255, 255, 255), font=font)

    return canvas


def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def grid_to_b64(grid: list[list[int]]) -> str:
    return image_to_b64(grid_to_image(grid))


def side_by_side_b64(before: list[list[int]], after: list[list[int]], label_before: str = "BEFORE", label_after: str = "AFTER") -> str:
    return image_to_b64(make_side_by_side(before, after, label_before, label_after))


def diff_to_b64(before: list[list[int]], after: list[list[int]]) -> str:
    h = len(before)
    w = len(before[0]) if h > 0 else 0
    img = Image.new("RGB", (w, h), (0, 0, 0))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            if before[y][x] != after[y][x]:
                pixels[x, y] = (255, 0, 0)
    return image_to_b64(img.resize((w * SCALE, h * SCALE), Image.NEAREST))


def image_block(b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}
