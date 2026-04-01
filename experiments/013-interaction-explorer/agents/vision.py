"""Vision utilities for GPT-5.4 Responses API.

- Quantized grid rendering (nearest-neighbor, crisp pixels)
- Diff highlighting between frames
- Base64 encoding for input_image format
"""

import base64
import io

from PIL import Image, ImageDraw, ImageFont

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

SCALE = 8  # 64*8 = 512px

Grid = list[list[int]]


def grid_to_image(grid: Grid) -> Image.Image:
    h, w = len(grid), len(grid[0])
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = PALETTE[grid[y][x] & 0xF]
    return img.resize((w * SCALE, h * SCALE), Image.NEAREST)


def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def diff_highlight_image(before: Grid, after: Grid) -> Image.Image:
    """Render AFTER frame with changed cells outlined in red."""
    img = grid_to_image(after)
    draw = ImageDraw.Draw(img)
    h, w = len(after), len(after[0])
    for y in range(h):
        for x in range(w):
            if before[y][x] != after[y][x]:
                x1, y1 = x * SCALE, y * SCALE
                x2, y2 = x1 + SCALE - 1, y1 + SCALE - 1
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
    return img


def side_by_side(before: Grid, after: Grid) -> Image.Image:
    """Previous frame + current frame with diff highlights, side by side."""
    img_before = grid_to_image(before)
    img_after = diff_highlight_image(before, after)
    gap = 20
    label_h = 30
    total_w = img_before.width + gap + img_after.width
    total_h = max(img_before.height, img_after.height) + label_h
    canvas = Image.new("RGB", (total_w, total_h), (40, 40, 40))
    canvas.paste(img_before, (0, label_h))
    canvas.paste(img_after, (img_before.width + gap, label_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((img_before.width // 2 - 30, 6), "PREVIOUS", fill=(255, 255, 255), font=font)
    draw.text((img_before.width + gap + img_after.width // 2 - 60, 6), "CURRENT (red=changed)", fill=(255, 100, 100), font=font)
    return canvas


def grid_b64(grid: Grid) -> str:
    return image_to_b64(grid_to_image(grid))


def diff_b64(before: Grid, after: Grid) -> str:
    return image_to_b64(side_by_side(before, after))


# GPT-5.4 Responses API format
def input_text(text: str) -> dict:
    return {"type": "input_text", "text": text}


def input_image_b64(b64: str) -> dict:
    return {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}
