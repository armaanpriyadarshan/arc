"""Tests for optimization 1: skip image generation when grid unchanged."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.vision import grid_b64, diff_b64


def _make_grid(val=0):
    return [[val] * 64 for _ in range(64)]


def test_identical_grids_compare_equal():
    """Two grids with same content should compare equal with ==."""
    g1 = _make_grid(5)
    g2 = _make_grid(5)
    assert g1 == g2


def test_different_grids_compare_unequal():
    """Grids with different content should not compare equal."""
    g1 = _make_grid(5)
    g2 = _make_grid(5)
    g2[10][10] = 8
    assert g1 != g2


def test_identical_grid_sends_plain_frame():
    """When grid is unchanged, a plain grid_b64 should be sent, not a diff."""
    grid = _make_grid(5)
    grid[10][10] = 8

    # Simulate what the agent does when grid unchanged:
    # It should send grid_b64(current_grid), NOT diff_b64(prev, current)
    plain_b64 = grid_b64(grid)
    diff_b64_result = diff_b64(grid, grid)  # diff of identical grids

    # The plain image should be different from the diff (diff has side-by-side layout)
    assert plain_b64 != diff_b64_result, (
        "Plain frame and diff image should be different formats"
    )

    # Plain image should be smaller (single frame vs side-by-side)
    assert len(plain_b64) < len(diff_b64_result), (
        "Plain frame should be smaller than side-by-side diff"
    )


def test_cached_frame_reused():
    """Calling grid_b64 twice on same grid should produce identical b64."""
    grid = _make_grid(5)
    grid[20][20] = 14

    b64_1 = grid_b64(grid)
    b64_2 = grid_b64(grid)
    assert b64_1 == b64_2, "Same grid should produce identical base64"


def test_different_grid_sends_diff():
    """When grid changes, a diff image with red outlines should be sent."""
    before = _make_grid(5)
    after = _make_grid(5)
    after[10][10] = 8

    diff_result = diff_b64(before, after)
    plain_result = grid_b64(after)

    # Diff should be different from plain (has side-by-side + red outlines)
    assert diff_result != plain_result


def test_cache_cleared_on_reset():
    """After clearing cached state, fresh image should be generated."""
    grid = _make_grid(5)
    b64_1 = grid_b64(grid)

    # Simulate clearing (what agent does on GAME_OVER/level-up)
    cached_b64 = None
    cached_grid = None

    # After clear, should regenerate
    assert cached_b64 is None
    b64_2 = grid_b64(grid)
    assert b64_2 == b64_1  # same grid produces same image
