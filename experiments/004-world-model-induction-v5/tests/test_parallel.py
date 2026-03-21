"""Tests for optimization 8: parallel image generation + symbolic analysis."""

import sys
import os
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.symbolic import grid_to_symbolic, symbolic_cache_clear
from agents.vision import grid_b64, diff_b64


def _make_grid(val=0):
    return [[val] * 64 for _ in range(64)]


def _place_block(grid, r, c, size=3, color=8):
    for dr in range(size):
        for dc in range(size):
            if r + dr < 64 and c + dc < 64:
                grid[r + dr][c + dc] = color


def setup_function():
    symbolic_cache_clear()


def test_parallel_same_results_as_sequential():
    """Parallel execution should produce identical results to sequential."""
    grid = _make_grid(0)
    _place_block(grid, 10, 10, 5, 8)
    _place_block(grid, 30, 30, 4, 14)
    symbolic_cache_clear()

    # Sequential
    seq_symbolic = grid_to_symbolic(grid)
    symbolic_cache_clear()
    seq_b64 = grid_b64(grid)

    # Parallel
    symbolic_cache_clear()
    pool = ThreadPoolExecutor(max_workers=2)
    sym_future = pool.submit(grid_to_symbolic, grid)
    img_future = pool.submit(grid_b64, grid)

    par_symbolic = sym_future.result()
    par_b64 = img_future.result()
    pool.shutdown(wait=True)

    assert seq_symbolic == par_symbolic, "Symbolic results should match"
    assert seq_b64 == par_b64, "Image results should match"


def test_parallel_diff_same_results():
    """Parallel diff_b64 should produce same result as sequential."""
    before = _make_grid(0)
    _place_block(before, 10, 10, 5, 8)
    after = _make_grid(0)
    _place_block(after, 12, 12, 5, 8)

    seq_diff = diff_b64(before, after)

    pool = ThreadPoolExecutor(max_workers=2)
    diff_future = pool.submit(diff_b64, before, after)
    par_diff = diff_future.result()
    pool.shutdown(wait=True)

    assert seq_diff == par_diff, "Diff results should match"


def test_parallel_error_fallback():
    """If image generation raises, we should be able to catch it."""
    def failing_func():
        raise ValueError("Simulated image failure")

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(failing_func)

    caught = False
    try:
        future.result(timeout=5)
    except ValueError:
        caught = True

    pool.shutdown(wait=True)
    assert caught, "Exception from thread should propagate to caller"


def test_thread_pool_shutdown():
    """ThreadPoolExecutor.shutdown(wait=False) should not hang."""
    pool = ThreadPoolExecutor(max_workers=2)
    grid = _make_grid(5)
    pool.submit(grid_b64, grid)
    pool.shutdown(wait=False)
    # If we get here without hanging, the test passes
