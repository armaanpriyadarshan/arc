"""Tests for optimization 2: LRU cache on grid_to_symbolic."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.symbolic import (
    grid_to_symbolic,
    symbolic_cache_info,
    symbolic_cache_clear,
    _grid_to_hashable,
)


def _make_grid(val=0):
    return [[val] * 64 for _ in range(64)]


def _place_block(grid, r, c, size=3, color=8):
    """Place a colored block on the grid."""
    for dr in range(size):
        for dc in range(size):
            if r + dr < 64 and c + dc < 64:
                grid[r + dr][c + dc] = color


def setup_function():
    """Clear cache before each test."""
    symbolic_cache_clear()


def test_same_grid_returns_cached():
    """Calling grid_to_symbolic twice on the same grid should hit cache."""
    grid = _make_grid(0)
    _place_block(grid, 10, 10, 5, 8)

    info_before = symbolic_cache_info()
    result1 = grid_to_symbolic(grid)
    info_after_first = symbolic_cache_info()
    assert info_after_first.misses == info_before.misses + 1

    result2 = grid_to_symbolic(grid)
    info_after_second = symbolic_cache_info()
    assert info_after_second.hits == info_after_first.hits + 1

    # Results should be equal
    assert result1 == result2


def test_same_grid_faster_on_cache_hit():
    """Second call on same grid should be significantly faster."""
    grid = _make_grid(0)
    _place_block(grid, 10, 10, 5, 8)
    _place_block(grid, 30, 30, 5, 14)

    # Warm up
    grid_to_symbolic(grid)
    symbolic_cache_clear()

    # First call (cache miss)
    t0 = time.perf_counter()
    grid_to_symbolic(grid)
    first_time = time.perf_counter() - t0

    # Second call (cache hit)
    t0 = time.perf_counter()
    grid_to_symbolic(grid)
    second_time = time.perf_counter() - t0

    # Cache hit should be at least 2x faster
    assert second_time < first_time / 2, (
        f"Cache hit ({second_time:.6f}s) should be >2x faster than miss ({first_time:.6f}s)"
    )


def test_different_grid_recomputes():
    """Modifying a cell should produce different results."""
    grid_a = _make_grid(0)
    _place_block(grid_a, 10, 10, 5, 8)
    result_a = grid_to_symbolic(grid_a)

    grid_b = _make_grid(0)
    _place_block(grid_b, 20, 20, 5, 8)  # different position
    result_b = grid_to_symbolic(grid_b)

    # Results should differ (different object positions)
    assert result_a != result_b


def test_cache_eviction():
    """Cache maxsize=4: 5th distinct grid should evict the first."""
    grids = []
    for i in range(5):
        g = _make_grid(0)
        _place_block(g, i * 10, i * 10, 3, 8)
        grids.append(g)

    for g in grids:
        grid_to_symbolic(g)

    info = symbolic_cache_info()
    assert info.currsize == 4, f"Cache should hold max 4, got {info.currsize}"

    # First grid should have been evicted — calling it again should be a miss
    hits_before = symbolic_cache_info().hits
    grid_to_symbolic(grids[0])
    hits_after = symbolic_cache_info().hits
    # If it was evicted, this is a miss (hits unchanged)
    # If it wasn't, this is a hit (hits + 1)
    # With LRU, the first grid IS the least recently used, so it should be evicted
    assert hits_after == hits_before, "First grid should have been evicted from cache"


def test_hashable_conversion():
    """Same grid content should produce same hashable key."""
    grid1 = _make_grid(5)
    grid2 = _make_grid(5)
    grid3 = _make_grid(8)  # different

    key1 = _grid_to_hashable(grid1)
    key2 = _grid_to_hashable(grid2)
    key3 = _grid_to_hashable(grid3)

    assert key1 == key2, "Identical grids should have same hashable key"
    assert key1 != key3, "Different grids should have different hashable keys"
    assert isinstance(key1, tuple), "Key should be a tuple"
    assert isinstance(key1[0], tuple), "Key rows should be tuples"


def test_cache_info_hits():
    """Verify cache_info properly tracks hits and misses."""
    grid = _make_grid(0)
    _place_block(grid, 5, 5, 3, 14)

    info0 = symbolic_cache_info()
    grid_to_symbolic(grid)  # miss
    info1 = symbolic_cache_info()
    grid_to_symbolic(grid)  # hit
    info2 = symbolic_cache_info()

    assert info1.misses == info0.misses + 1
    assert info2.hits == info1.hits + 1
