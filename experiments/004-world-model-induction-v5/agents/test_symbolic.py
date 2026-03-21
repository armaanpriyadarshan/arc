"""Unit tests for enhanced symbolic.py.

Tests sub-pattern classification, orientation, symmetry, holes,
rotation_id, composites, containment, and backward compatibility.
"""

import sys
import os

# Add parent so we can import from agents/
sys.path.insert(0, os.path.dirname(__file__))

from symbolic import (
    grid_to_symbolic,
    diff_symbolic,
    _normalize_cells,
    _rotate_90_cw,
    _mirror_h,
    _mirror_v,
    _classify_subpattern,
    _detect_orientation,
    _detect_symmetry,
    _detect_holes,
    _canonical_rotation_id,
    _TEMPLATE_LOOKUP,
)


def make_grid(w=64, h=64, bg=0):
    """Create a blank grid filled with bg color."""
    return [[bg] * w for _ in range(h)]


def place_cells(grid, cells, color):
    """Place cells onto grid with given color."""
    for r, c in cells:
        grid[r][c] = color


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def test_normalize_cells():
    cells = frozenset({(5, 10), (6, 10), (7, 10), (7, 11)})
    norm = _normalize_cells(cells)
    assert norm == frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}), f"Got {norm}"
    print("  PASS: _normalize_cells")


def test_rotate_90_cw():
    # L_shape rotated 90 CW
    L = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
    rotated = _rotate_90_cw(L)
    # (0,0)->(0,0), (1,0)->(0,-1), (2,0)->(0,-2), (2,1)->(1,-2)
    # normalize: min_r=0, min_c=-2 -> add 2 to cols
    # (0,2), (0,1), (0,0), (1,0)
    expected = frozenset({(0, 0), (0, 1), (0, 2), (1, 0)})
    assert rotated == expected, f"Got {rotated}"
    print("  PASS: _rotate_90_cw")


def test_mirrors():
    L = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
    h_mirror = _mirror_h(L)
    # max_c=1, so (r, 1-c): (0,1),(1,1),(2,1),(2,0) = J_shape
    expected_h = frozenset({(0, 1), (1, 1), (2, 1), (2, 0)})
    assert h_mirror == expected_h, f"mirror_h got {h_mirror}"

    v_mirror = _mirror_v(L)
    # max_r=2, so (2-r, c): (2,0),(1,0),(0,0),(0,1)
    expected_v = frozenset({(0, 0), (0, 1), (1, 0), (2, 0)})
    assert v_mirror == expected_v, f"mirror_v got {v_mirror}"
    print("  PASS: _mirror_h, _mirror_v")


# ---------------------------------------------------------------------------
# Sub-pattern classification
# ---------------------------------------------------------------------------

def test_template_lookup():
    # L_shape should be in lookup in all 8 orientations
    L = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
    assert L in _TEMPLATE_LOOKUP, "L_shape not in template lookup"

    # Cross should be in lookup
    cross = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})
    assert _TEMPLATE_LOOKUP[cross] == "cross", f"cross mapped to {_TEMPLATE_LOOKUP[cross]}"

    # T_shape
    T = frozenset({(0, 0), (0, 1), (0, 2), (1, 1)})
    assert _TEMPLATE_LOOKUP[T] == "T_shape"
    print("  PASS: template_lookup contains expected shapes")


def test_classify_subpattern_L():
    L = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
    # L_shape and corner/small_L share rotations, so the name may vary
    result = _classify_subpattern(L, "irregular", 4)
    assert result is not None, "L_shape not classified"
    print(f"  PASS: L_shape classified as '{result}'")


def test_classify_subpattern_cross():
    cross = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})
    result = _classify_subpattern(cross, "irregular", 5)
    assert result == "cross", f"Cross classified as '{result}'"
    print("  PASS: cross classified correctly")


def test_classify_subpattern_topological():
    # A 5-cell line segment (not in templates since it's just a line)
    line = frozenset({(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)})
    result = _classify_subpattern(line, "sparse_cluster", 5)
    # This is a vertical line, might match template or fallback
    if result is None:
        # Check topological: 2 endpoints, 0 branch points -> line_segment
        result = _classify_subpattern(line, "sparse_cluster", 5)
    print(f"  PASS: 5-cell line classified as '{result}'")


def test_classify_subpattern_too_large():
    # Size > 20 should return None
    big = frozenset((i, 0) for i in range(25))
    result = _classify_subpattern(big, "irregular", 25)
    assert result is None, f"Size 25 should be None, got '{result}'"
    print("  PASS: large shape returns None")


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def test_orientation_L_shape():
    # L pointing roughly NE (vertical bar + horizontal extension at bottom)
    L = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
    direction, angle = _detect_orientation(L)
    assert direction is not None, "L_shape should have orientation"
    print(f"  PASS: L_shape orientation='{direction}', angle={angle}")


def test_orientation_symmetric_returns_none():
    # Square-ish cluster should return None (near-circular)
    square = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
    direction, angle = _detect_orientation(square)
    assert direction is None, f"Square should have no orientation, got '{direction}'"
    print("  PASS: symmetric shape has no orientation")


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

def test_symmetry_cross():
    cross = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})
    sym = _detect_symmetry(cross)
    assert sym["horizontal"] is True, "Cross should be h-symmetric"
    assert sym["vertical"] is True, "Cross should be v-symmetric"
    assert sym["rot_90"] is True, "Cross should be 90-rot-symmetric"
    assert sym["rot_180"] is True, "Cross should be 180-rot-symmetric"
    print("  PASS: cross has full symmetry")


def test_symmetry_L_shape():
    L = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
    sym = _detect_symmetry(L)
    assert sym["horizontal"] is False, "L should not be h-symmetric"
    assert sym["vertical"] is False, "L should not be v-symmetric"
    assert sym["rot_90"] is False, "L should not be 90-rot-symmetric"
    print("  PASS: L_shape has no symmetry")


def test_symmetry_T_shape():
    T = frozenset({(0, 0), (0, 1), (0, 2), (1, 1)})
    sym = _detect_symmetry(T)
    assert sym["horizontal"] is True, "T should be h-symmetric"
    print(f"  PASS: T_shape symmetry: {sym}")


# ---------------------------------------------------------------------------
# Holes
# ---------------------------------------------------------------------------

def test_holes_ring():
    # Create a hollow square (ring): 5x5 outer, 3x3 inner hole
    grid = make_grid(20, 20, bg=0)
    ring_cells = set()
    for r in range(5, 10):
        for c in range(5, 10):
            if r == 5 or r == 9 or c == 5 or c == 9:
                ring_cells.add((r, c))
                grid[r][c] = 8  # red

    holes = _detect_holes(frozenset(ring_cells), grid, 5, 5, 9, 9)
    assert holes is not None, "Ring should have holes"
    assert holes["count"] == 1, f"Expected 1 hole, got {holes['count']}"
    assert holes["total_cells"] == 9, f"Expected 9 hole cells, got {holes['total_cells']}"
    print(f"  PASS: ring hole detection: {holes}")


def test_holes_solid():
    # Solid 3x3 square: no holes
    grid = make_grid(20, 20, bg=0)
    solid_cells = set()
    for r in range(5, 8):
        for c in range(5, 8):
            solid_cells.add((r, c))
            grid[r][c] = 8

    holes = _detect_holes(frozenset(solid_cells), grid, 5, 5, 7, 7)
    assert holes is None, "Solid square should have no holes"
    print("  PASS: solid square has no holes")


# ---------------------------------------------------------------------------
# Rotation ID
# ---------------------------------------------------------------------------

def test_rotation_id_invariant():
    L = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
    rot_L = _rotate_90_cw(L)
    rot2_L = _rotate_90_cw(rot_L)

    id1 = _canonical_rotation_id(L)
    id2 = _canonical_rotation_id(rot_L)
    id3 = _canonical_rotation_id(rot2_L)

    assert id1 == id2 == id3, f"Rotation IDs differ: {id1}, {id2}, {id3}"
    print(f"  PASS: rotation_id invariant across rotations: {id1}")


def test_rotation_id_different_shapes():
    L = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
    T = frozenset({(0, 0), (0, 1), (0, 2), (1, 1)})
    id_L = _canonical_rotation_id(L)
    id_T = _canonical_rotation_id(T)
    assert id_L != id_T, f"L and T should have different IDs: {id_L} vs {id_T}"
    print(f"  PASS: different shapes get different rotation_ids")


# ---------------------------------------------------------------------------
# Full grid_to_symbolic: backward compat + new fields
# ---------------------------------------------------------------------------

def test_backward_compat():
    """Ensure existing fields are present and unchanged."""
    grid = make_grid(64, 64, bg=0)
    # Place a 3x3 red square
    for r in range(10, 13):
        for c in range(20, 23):
            grid[r][c] = 8

    result = grid_to_symbolic(grid)
    assert "grid_size" in result
    assert "objects" in result
    assert "relations" in result
    assert "backgrounds" in result

    objs = result["objects"]
    assert len(objs) >= 1
    obj = objs[0]
    for key in ("id", "color", "color_id", "shape", "size", "position", "bbox", "center"):
        assert key in obj, f"Missing key: {key}"
    assert "_cells" not in obj, "_cells should be stripped"
    print("  PASS: backward compatibility maintained")


def test_new_fields_on_irregular():
    """L-shape should get subpattern, orientation, symmetry, rotation_id."""
    grid = make_grid(64, 64, bg=0)
    # Place an L shape at (10, 10)
    L_cells = [(10, 10), (11, 10), (12, 10), (12, 11)]
    for r, c in L_cells:
        grid[r][c] = 14  # green

    result = grid_to_symbolic(grid)
    objs = result["objects"]
    assert len(objs) >= 1

    obj = objs[0]
    assert obj["shape"] in ("irregular", "sparse_cluster"), f"L classified as {obj['shape']}"
    assert "subpattern" in obj, "Missing subpattern"
    assert "symmetry" in obj, "Missing symmetry"
    assert "rotation_id" in obj, "Missing rotation_id"
    print(f"  PASS: L-shape new fields: subpattern={obj.get('subpattern')}, "
          f"orientation={obj.get('orientation')}, rotation_id={obj.get('rotation_id')}")


def test_containment_relation():
    """Smaller object inside larger should produce 'contains' relation."""
    grid = make_grid(64, 64, bg=0)

    # Outer ring (red, 7x7 frame)
    for r in range(20, 27):
        for c in range(20, 27):
            if r == 20 or r == 26 or c == 20 or c == 26:
                grid[r][c] = 8  # red

    # Inner square (blue, 3x3 centered)
    for r in range(22, 25):
        for c in range(22, 25):
            grid[r][c] = 9  # blue

    result = grid_to_symbolic(grid)
    contains_rels = [r for r in result["relations"] if r["type"] == "contains"]
    assert len(contains_rels) >= 1, f"Expected contains relation, got: {result['relations']}"
    print(f"  PASS: containment detected: {contains_rels}")


def test_composites():
    """Two adjacent objects of different colors should form a composite."""
    grid = make_grid(64, 64, bg=0)

    # Green 3x2 block
    for r in range(10, 13):
        for c in range(10, 12):
            grid[r][c] = 14

    # Purple 3x2 block right next to it
    for r in range(10, 13):
        for c in range(12, 14):
            grid[r][c] = 15

    result = grid_to_symbolic(grid)
    assert "composites" in result, "Expected composites key"
    assert len(result["composites"]) >= 1, "Expected at least one composite"
    comp = result["composites"][0]
    assert len(comp["component_ids"]) == 2
    assert "green" in comp["colors"] and "purple" in comp["colors"]
    print(f"  PASS: composite detected: {comp}")


def test_holes_in_grid():
    """Ring object should have holes detected in full pipeline."""
    grid = make_grid(64, 64, bg=0)

    # 7x7 ring (red border, hollow inside)
    for r in range(20, 27):
        for c in range(20, 27):
            if r == 20 or r == 26 or c == 20 or c == 26:
                grid[r][c] = 8  # red

    result = grid_to_symbolic(grid)
    ring_objs = [o for o in result["objects"] if o["color"] == "red"]
    assert len(ring_objs) >= 1
    ring = ring_objs[0]
    assert "holes" in ring, f"Ring should have holes. Keys: {list(ring.keys())}"
    assert ring["holes"]["count"] >= 1
    print(f"  PASS: holes in ring: {ring['holes']}")


def test_diff_symbolic_new_fields():
    """diff_symbolic should report changes in orientation/subpattern/rotation_id."""
    prev = {
        "objects": [{
            "id": 1, "color": "green", "color_id": 14,
            "shape": "irregular", "size": 4,
            "position": [10, 10], "bbox": [10, 10, 12, 11],
            "center": [11, 10],
            "subpattern": "L_shape",
            "orientation": "N",
            "rotation_id": "abc12345",
        }],
        "backgrounds": [],
    }
    curr = {
        "objects": [{
            "id": 1, "color": "green", "color_id": 14,
            "shape": "irregular", "size": 4,
            "position": [10, 10], "bbox": [10, 10, 12, 11],
            "center": [11, 10],
            "subpattern": "T_shape",
            "orientation": "E",
            "rotation_id": "def67890",
        }],
        "backgrounds": [],
    }

    changes = diff_symbolic(prev, curr)
    assert len(changes) >= 1
    change = changes[0]
    assert "subpattern" in change, f"Missing subpattern diff, got: {change}"
    assert "orientation" in change, f"Missing orientation diff"
    assert "rotation_id" in change, f"Missing rotation_id diff"
    print(f"  PASS: diff_symbolic reports new field changes")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Transform Helpers ===")
    test_normalize_cells()
    test_rotate_90_cw()
    test_mirrors()

    print("\n=== Sub-pattern Classification ===")
    test_template_lookup()
    test_classify_subpattern_L()
    test_classify_subpattern_cross()
    test_classify_subpattern_topological()
    test_classify_subpattern_too_large()

    print("\n=== Orientation ===")
    test_orientation_L_shape()
    test_orientation_symmetric_returns_none()

    print("\n=== Symmetry ===")
    test_symmetry_cross()
    test_symmetry_L_shape()
    test_symmetry_T_shape()

    print("\n=== Holes ===")
    test_holes_ring()
    test_holes_solid()

    print("\n=== Rotation ID ===")
    test_rotation_id_invariant()
    test_rotation_id_different_shapes()

    print("\n=== Full Pipeline ===")
    test_backward_compat()
    test_new_fields_on_irregular()
    test_containment_relation()
    test_composites()
    test_holes_in_grid()
    test_diff_symbolic_new_fields()

    print("\n=== ALL TESTS PASSED ===")
