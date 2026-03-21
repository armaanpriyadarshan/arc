"""Tests for image rendering (SCALE=8 retained for model quality)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.vision import grid_to_image, diff_highlight_image, image_to_b64, SCALE


def _make_grid(val=0):
    return [[val] * 64 for _ in range(64)]


def test_scale_is_8():
    assert SCALE == 8, f"Expected SCALE=8, got {SCALE}"


def test_scale_8_produces_512x512():
    grid = _make_grid(5)
    img = grid_to_image(grid)
    assert img.size == (512, 512), f"Expected 512x512, got {img.size}"


def test_diff_outline_visible():
    """Changed cell should have red outline with cell color still visible in center."""
    before = _make_grid(5)  # all black
    after = _make_grid(5)
    after[10][10] = 8  # change one cell to red

    img = diff_highlight_image(before, after)
    px = img.load()

    # At SCALE=8, cell (10,10) maps to pixels (80,80)-(87,87)
    x0, y0 = 10 * SCALE, 10 * SCALE

    # Outline pixel (edge) should be red
    assert px[x0, y0] == (255, 0, 0), f"Edge pixel should be red, got {px[x0, y0]}"

    # Center pixel should show the cell's actual color, not outline red
    center_x, center_y = x0 + 4, y0 + 4
    center_color = px[center_x, center_y]
    assert center_color != (255, 0, 0), (
        "Center pixel should show cell color, not outline red"
    )
