"""Convert a 64x64 int grid into a symbolic state representation.

Finds connected components, classifies shapes, determines spatial relations.
Output is a dict the model can reason over without pixel-squinting.

v5: grid_to_symbolic is memoized via LRU cache (maxsize=4) keyed on grid
content. Callers MUST NOT mutate the returned dict.
"""

import hashlib
import math
from functools import lru_cache

COLOR_NAMES = {
    0: "white", 1: "light_gray", 2: "gray", 3: "dark_gray",
    4: "near_black", 5: "black", 6: "magenta", 7: "pink",
    8: "red", 9: "blue", 10: "light_blue", 11: "yellow",
    12: "orange", 13: "maroon", 14: "green", 15: "purple",
}

Grid = list[list[int]]


# ---------------------------------------------------------------------------
# Step 1: Transform helpers
# ---------------------------------------------------------------------------

def _normalize_cells(cells):
    """Translate cells so min row/col = 0, return frozenset."""
    if not cells:
        return frozenset()
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _rotate_90_cw(norm):
    """Rotate 90° clockwise: (r,c) -> (c, -r), then normalize."""
    rotated = frozenset((c, -r) for r, c in norm)
    return _normalize_cells(rotated)


def _mirror_h(norm):
    """Flip left-right: (r, max_c - c)."""
    if not norm:
        return frozenset()
    max_c = max(c for r, c in norm)
    return frozenset((r, max_c - c) for r, c in norm)


def _mirror_v(norm):
    """Flip top-bottom: (max_r - r, c)."""
    if not norm:
        return frozenset()
    max_r = max(r for r, c in norm)
    return frozenset((max_r - r, c) for r, c in norm)


# ---------------------------------------------------------------------------
# Step 2: Template lookup for sub-pattern classification
# ---------------------------------------------------------------------------

_BASE_SHAPES = {
    "L_shape": frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
    "J_shape": frozenset({(0, 1), (1, 1), (2, 1), (2, 0)}),
    "T_shape": frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
    "S_shape": frozenset({(0, 1), (0, 2), (1, 0), (1, 1)}),
    "Z_shape": frozenset({(0, 0), (0, 1), (1, 1), (1, 2)}),
    "O_shape": frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    "cross": frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}),
    "small_L": frozenset({(0, 0), (1, 0), (1, 1)}),
    "corner": frozenset({(0, 0), (0, 1), (1, 0)}),
    "big_L": frozenset({(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)}),
    "diagonal_2": frozenset({(0, 0), (1, 1)}),
    "diagonal_3": frozenset({(0, 0), (1, 1), (2, 2)}),
    "antidiag_3": frozenset({(0, 2), (1, 1), (2, 0)}),
}


def _build_template_lookup():
    """Build lookup mapping frozenset -> shape name for all 8 orientations."""
    lookup = {}
    for name, base in _BASE_SHAPES.items():
        current = base
        for _ in range(4):
            lookup[current] = name
            lookup[_mirror_h(current)] = name
            current = _rotate_90_cw(current)
    return lookup


_TEMPLATE_LOOKUP = _build_template_lookup()


def _classify_subpattern(norm_cells, shape, size):
    """Classify sub-pattern via template match or topological fallback."""
    if size > 20:
        return None

    match = _TEMPLATE_LOOKUP.get(norm_cells)
    if match is not None:
        return match

    # Topological fallback for small unmatched shapes (size <= 8)
    if size > 8:
        return None

    endpoints = 0
    branch_points = 0
    for r, c in norm_cells:
        nbrs = sum(1 for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    if (r + dr, c + dc) in norm_cells)
        if nbrs == 1:
            endpoints += 1
        elif nbrs >= 3:
            branch_points += 1

    if branch_points > 0:
        return "hook"
    elif endpoints == 2:
        return "line_segment"
    elif endpoints == 0 and size >= 4:
        return "loop"
    return None


# ---------------------------------------------------------------------------
# Step 3: Orientation detection (simplified PCA, no numpy)
# ---------------------------------------------------------------------------

def _detect_orientation(norm_cells):
    """Detect orientation via PCA on cell coordinates.

    Returns (compass_direction, axis_angle_degrees) or (None, None).
    """
    n = len(norm_cells)
    if n < 3:
        return None, None

    cells = list(norm_cells)
    mean_r = sum(r for r, c in cells) / n
    mean_c = sum(c for r, c in cells) / n

    cov_rr = sum((r - mean_r) ** 2 for r, c in cells) / n
    cov_cc = sum((c - mean_c) ** 2 for r, c in cells) / n
    cov_rc = sum((r - mean_r) * (c - mean_c) for r, c in cells) / n

    # Eigenvalues of 2x2 covariance matrix
    trace = cov_rr + cov_cc
    if trace < 1e-6:
        return None, None
    det = cov_rr * cov_cc - cov_rc * cov_rc
    discriminant = max(0.0, trace * trace - 4 * det)
    sqrt_disc = discriminant ** 0.5
    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2

    # No dominant axis if eigenvalues are similar (near-circular spread)
    if lambda1 < 1e-6 or lambda2 / lambda1 > 0.8:
        return None, None

    # Dominant eigenvector angle
    theta = 0.5 * math.atan2(2 * cov_rc, cov_cc - cov_rr)

    # Direction vector in grid coords: dc=cos(theta), dr=sin(theta)
    dc = math.cos(theta)
    dr = math.sin(theta)

    # Skewness along principal axis to pick "pointing" end
    projections = [(r - mean_r) * dr + (c - mean_c) * dc for r, c in cells]
    mean_p = sum(projections) / n
    var_p = sum((p - mean_p) ** 2 for p in projections) / n
    if var_p > 1e-6:
        skew = sum((p - mean_p) ** 3 for p in projections) / (n * var_p ** 1.5)
    else:
        skew = 0

    # Positive skew -> narrow end toward positive -> shape points that way
    if skew < 0:
        dc, dr = -dc, -dr

    # Convert to compass (math coords: x=East=dc, y=North=-dr)
    compass_angle = math.degrees(math.atan2(-dr, dc)) % 360
    COMPASS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = round(compass_angle / 45) % 8
    direction = COMPASS[idx]

    axis_angle = round(math.degrees(theta) % 180, 1)
    return direction, axis_angle


# ---------------------------------------------------------------------------
# Step 4: Symmetry detection
# ---------------------------------------------------------------------------

def _detect_symmetry(norm_cells):
    """Check horizontal mirror, vertical mirror, 90° and 180° rotation symmetry."""
    rot90 = _rotate_90_cw(norm_cells)
    rot180 = _rotate_90_cw(rot90)
    return {
        "horizontal": _mirror_h(norm_cells) == norm_cells,
        "vertical": _mirror_v(norm_cells) == norm_cells,
        "rot_90": rot90 == norm_cells,
        "rot_180": rot180 == norm_cells,
    }


# ---------------------------------------------------------------------------
# Step 5: Hole / void detection
# ---------------------------------------------------------------------------

def _detect_holes(cells_set, grid, min_r, min_c, max_r, max_c):
    """Flood-fill from bbox border; unreachable non-object cells are holes."""
    bh = max_r - min_r + 1
    bw = max_c - min_c + 1

    # Boolean mask: True = object cell
    mask = [[False] * bw for _ in range(bh)]
    for r, c in cells_set:
        mask[r - min_r][c - min_c] = True

    # Flood-fill from border (mark reachable non-object cells)
    reachable = [[False] * bw for _ in range(bh)]
    queue = []
    for r in range(bh):
        for c in range(bw):
            if (r == 0 or r == bh - 1 or c == 0 or c == bw - 1) and not mask[r][c]:
                queue.append((r, c))
                reachable[r][c] = True

    qi = 0
    while qi < len(queue):
        cr, cc = queue[qi]
        qi += 1
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < bh and 0 <= nc < bw and not reachable[nr][nc] and not mask[nr][nc]:
                reachable[nr][nc] = True
                queue.append((nr, nc))

    # Collect hole cells (non-object, non-reachable)
    hole_cells = []
    for r in range(bh):
        for c in range(bw):
            if not mask[r][c] and not reachable[r][c]:
                hole_cells.append((r + min_r, c + min_c))

    if not hole_cells:
        return None

    # Count separate hole regions via BFS
    hole_set = set(hole_cells)
    visited_h = set()
    hole_count = 0
    for cell in hole_cells:
        if cell in visited_h:
            continue
        hole_count += 1
        stack = [cell]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited_h:
                continue
            visited_h.add((cr, cc))
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in hole_set and (nr, nc) not in visited_h:
                    stack.append((nr, nc))

    # Colors inside the holes
    colors = set()
    for r, c in hole_cells:
        colors.add(COLOR_NAMES.get(grid[r][c], f"color_{grid[r][c]}"))

    return {
        "count": hole_count,
        "total_cells": len(hole_cells),
        "colors": sorted(colors),
    }


# ---------------------------------------------------------------------------
# Step 6: Canonical rotation ID
# ---------------------------------------------------------------------------

def _canonical_rotation_id(norm_cells):
    """Hash invariant to rotation — two rotated copies share the same ID."""
    orientations = []
    current = norm_cells
    for _ in range(4):
        orientations.append(tuple(sorted(current)))
        current = _rotate_90_cw(current)
    canonical = min(orientations)
    return hashlib.md5(str(canonical).encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Step 7: Multi-color composite grouping
# ---------------------------------------------------------------------------

def _find_composites(objects):
    """Group adjacent objects of different colors into composites via dilation."""
    n = len(objects)
    if n < 2:
        return []

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        if "_cells" not in objects[i]:
            continue
        dilated = set()
        for r, c in objects[i]["_cells"]:
            dilated.update([(r, c), (r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])
        for j in range(i + 1, n):
            if "_cells" not in objects[j]:
                continue
            if dilated & objects[j]["_cells"]:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    composites = []
    comp_id = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        colors = set(objects[i]["color"] for i in indices)
        if len(colors) < 2:
            continue

        comp_id += 1
        component_ids = sorted(objects[i]["id"] for i in indices)
        total_size = sum(objects[i]["size"] for i in indices)

        all_rows, all_cols = [], []
        for i in indices:
            if "_cells" in objects[i]:
                for r, c in objects[i]["_cells"]:
                    all_rows.append(r)
                    all_cols.append(c)

        if all_rows:
            bbox = {"min_row": min(all_rows), "min_col": min(all_cols), "max_row": max(all_rows), "max_col": max(all_cols)}
            center = {"row": sum(all_rows) // len(all_rows), "col": sum(all_cols) // len(all_cols)}
        else:
            bbox = dict(objects[indices[0]]["bbox"])
            center = dict(objects[indices[0]]["center"])

        composites.append({
            "composite_id": comp_id,
            "component_ids": component_ids,
            "colors": sorted(colors),
            "total_size": total_size,
            "bbox": bbox,
            "center": center,
        })

    return composites


# ---------------------------------------------------------------------------
# Step 10: Main — grid_to_symbolic (enhanced, with LRU cache)
# ---------------------------------------------------------------------------

def _grid_to_hashable(grid: Grid) -> tuple[tuple[int, ...], ...]:
    """Convert mutable grid to an immutable hashable key for caching."""
    return tuple(tuple(row) for row in grid)


@lru_cache(maxsize=4)
def _cached_grid_to_symbolic(grid_key: tuple[tuple[int, ...], ...], min_size: int) -> dict:
    """Cached inner implementation — keyed on immutable grid tuple."""
    grid = [list(row) for row in grid_key]
    return _grid_to_symbolic_impl(grid, min_size)


def grid_to_symbolic(grid: Grid, min_size: int = 2) -> dict:
    """Convert grid to symbolic state with objects and relations.

    Results are LRU-cached (maxsize=4) by grid content. Callers MUST NOT
    mutate the returned dict.
    """
    key = _grid_to_hashable(grid)
    return _cached_grid_to_symbolic(key, min_size)


def symbolic_cache_info():
    """Return cache statistics (hits, misses, maxsize, currsize)."""
    return _cached_grid_to_symbolic.cache_info()


def symbolic_cache_clear():
    """Clear the symbolic analysis cache."""
    _cached_grid_to_symbolic.cache_clear()


def _grid_to_symbolic_impl(grid: Grid, min_size: int = 2) -> dict:
    """Convert grid to symbolic state with objects and relations (uncached)."""
    h, w = len(grid), len(grid[0])

    # Find connected components
    visited = [[False] * w for _ in range(h)]
    objects = []
    obj_id = 0

    for r in range(h):
        for c in range(w):
            if visited[r][c]:
                continue
            color = grid[r][c]
            # BFS
            cells = []
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                if cr < 0 or cr >= h or cc < 0 or cc >= w:
                    continue
                if visited[cr][cc] or grid[cr][cc] != color:
                    continue
                visited[cr][cc] = True
                cells.append((cr, cc))
                stack.extend([(cr+1, cc), (cr-1, cc), (cr, cc+1), (cr, cc-1)])

            if len(cells) < min_size:
                continue

            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            area = len(cells)
            bbox_area = height * width

            # Classify shape
            if area == bbox_area:
                if height == 1:
                    shape = "horizontal_line"
                elif width == 1:
                    shape = "vertical_line"
                elif height == width:
                    shape = "square"
                else:
                    shape = "rectangle"
            elif area < bbox_area * 0.5:
                shape = "sparse_cluster"
            else:
                shape = "irregular"

            # Skip very large background regions
            if area > 1000:
                shape = "background"

            obj_id += 1
            objects.append({
                "id": obj_id,
                "color": COLOR_NAMES.get(color, f"color_{color}"),
                "color_id": color,
                "shape": shape,
                "size": area,
                "position": {"row": min_r, "col": min_c},
                "bbox": {"min_row": min_r, "min_col": min_c, "max_row": max_r, "max_col": max_c},
                "center": {"row": sum(rows) // len(rows), "col": sum(cols) // len(cols)},
                "_cells": frozenset(cells),
            })

    # --- Suppress background-color fragments ---
    # If any connected component of a color exceeds the background threshold,
    # smaller fragments of that same color are almost certainly floor/corridor
    # pieces split by the player or walls — not meaningful foreground objects.
    bg_colors = {obj["color_id"] for obj in objects if obj["shape"] == "background"}
    for obj in objects:
        if obj["shape"] != "background" and obj["color_id"] in bg_colors:
            obj["shape"] = "background"

    # --- Enhance foreground objects (Steps 2-6) ---
    for obj in objects:
        if obj["shape"] == "background":
            continue

        norm = _normalize_cells(obj["_cells"])

        # Step 2: Sub-pattern (irregular/sparse only)
        if obj["shape"] in ("irregular", "sparse_cluster"):
            sp = _classify_subpattern(norm, obj["shape"], obj["size"])
            if sp is not None:
                obj["subpattern"] = sp

        # Step 3: Orientation (irregular/sparse, skip symmetric subpatterns)
        if obj["shape"] in ("irregular", "sparse_cluster"):
            if obj.get("subpattern") not in ("cross", "O_shape"):
                direction, axis_angle = _detect_orientation(norm)
                if direction is not None:
                    obj["orientation"] = direction
                    obj["dominant_axis_angle"] = axis_angle

        # Step 4: Symmetry (irregular/sparse, size <= 100)
        if obj["shape"] in ("irregular", "sparse_cluster") and obj["size"] <= 100:
            obj["symmetry"] = _detect_symmetry(norm)

        # Step 5: Holes (size >= 9, not background)
        if obj["size"] >= 9:
            bb = obj["bbox"]
            holes = _detect_holes(obj["_cells"], grid, bb["min_row"], bb["min_col"], bb["max_row"], bb["max_col"])
            if holes is not None:
                obj["holes"] = holes

        # Step 6: Rotation ID
        obj["rotation_id"] = _canonical_rotation_id(norm)

    # Sort: small interesting objects first, backgrounds last
    objects.sort(key=lambda o: (o["shape"] == "background", o["size"]))

    # Compute relations between non-background objects
    fg_objects = [o for o in objects if o["shape"] != "background"]
    relations = []
    for i, a in enumerate(fg_objects):
        for b in fg_objects[i+1:]:
            ar, ac = a["center"]["row"], a["center"]["col"]
            br, bc = b["center"]["row"], b["center"]["col"]
            if ar < br - 5:
                relations.append({"type": "above", "a": a["id"], "b": b["id"]})
            elif ar > br + 5:
                relations.append({"type": "below", "a": a["id"], "b": b["id"]})
            if ac < bc - 5:
                relations.append({"type": "left_of", "a": a["id"], "b": b["id"]})
            elif ac > bc + 5:
                relations.append({"type": "right_of", "a": a["id"], "b": b["id"]})

            # Adjacent check
            a_r1, a_c1, a_r2, a_c2 = a["bbox"]["min_row"], a["bbox"]["min_col"], a["bbox"]["max_row"], a["bbox"]["max_col"]
            b_r1, b_c1, b_r2, b_c2 = b["bbox"]["min_row"], b["bbox"]["min_col"], b["bbox"]["max_row"], b["bbox"]["max_col"]
            if (abs(a_r2 - b_r1) <= 1 or abs(b_r2 - a_r1) <= 1) and not (a_c2 < b_c1 or b_c2 < a_c1):
                relations.append({"type": "adjacent", "a": a["id"], "b": b["id"]})
            if (abs(a_c2 - b_c1) <= 1 or abs(b_c2 - a_c1) <= 1) and not (a_r2 < b_r1 or b_r2 < a_r1):
                relations.append({"type": "adjacent", "a": a["id"], "b": b["id"]})

            # Step 8: Containment check (A's bbox strictly contains B's)
            if a_r1 < b_r1 and a_c1 < b_c1 and a_r2 > b_r2 and a_c2 > b_c2:
                relations.append({"type": "contains", "a": a["id"], "b": b["id"]})
            elif b_r1 < a_r1 and b_c1 < a_c1 and b_r2 > a_r2 and b_c2 > a_c2:
                relations.append({"type": "contains", "a": b["id"], "b": a["id"]})

    # Step 7: Multi-color composites
    composites = _find_composites(fg_objects)

    # Strip temporary _cells from all objects before returning
    for obj in objects:
        obj.pop("_cells", None)

    # Count backgrounds
    bg_objects = [o for o in objects if o["shape"] == "background"]

    result = {
        "grid_size": [h, w],
        "num_objects": len(fg_objects),
        "num_backgrounds": len(bg_objects),
        "objects": fg_objects,
        "backgrounds": [{"color": o["color"], "size": o["size"]} for o in bg_objects],
        "relations": relations[:20],
    }
    if composites:
        result["composites"] = composites

    return result


# ---------------------------------------------------------------------------
# Step 9: diff_symbolic (enhanced)
# ---------------------------------------------------------------------------

def diff_symbolic(prev: dict, curr: dict) -> list[dict]:
    """Compare two symbolic states. Report what changed — no interpretation.

    Matches objects across frames by color + similar size, then reports
    any differences in position, size, or existence.
    """
    if not prev or not curr:
        return []

    prev_objs = prev.get("objects", [])
    curr_objs = curr.get("objects", [])

    # Match by color + closest center
    changes = []
    matched_curr: set[int] = set()

    for po in prev_objs:
        best_match = None
        best_dist = 999
        for i, co in enumerate(curr_objs):
            if i in matched_curr:
                continue
            if co["color_id"] != po["color_id"]:
                continue
            if abs(co["size"] - po["size"]) > max(po["size"], co["size"]) * 0.5:
                continue
            dist = abs(co["center"]["row"] - po["center"]["row"]) + abs(co["center"]["col"] - po["center"]["col"])
            if dist < best_dist:
                best_dist = dist
                best_match = i

        if best_match is not None:
            matched_curr.add(best_match)
            co = curr_objs[best_match]
            diffs = {}
            if po["center"] != co["center"]:
                diffs["center"] = {"was": po["center"], "now": co["center"]}
            if po["size"] != co["size"]:
                diffs["size"] = {"was": po["size"], "now": co["size"]}
            if po["bbox"] != co["bbox"]:
                diffs["bbox"] = {"was": po["bbox"], "now": co["bbox"]}
            if po["shape"] != co["shape"]:
                diffs["shape"] = {"was": po["shape"], "now": co["shape"]}
            # New field comparisons (Step 9)
            for field in ("orientation", "subpattern", "rotation_id"):
                pv = po.get(field)
                cv = co.get(field)
                if pv != cv:
                    diffs[field] = {"was": pv, "now": cv}
            if diffs:
                changes.append({
                    "color": po["color"],
                    "type": "changed",
                    **diffs,
                })
        else:
            changes.append({
                "color": po["color"],
                "type": "disappeared",
                "was_at": po["center"],
                "was_size": po["size"],
            })

    for i, co in enumerate(curr_objs):
        if i not in matched_curr:
            changes.append({
                "color": co["color"],
                "type": "appeared",
                "at": co["center"],
                "size": co["size"],
            })

    # Background size changes
    prev_bg = {b["color"]: b["size"] for b in prev.get("backgrounds", [])}
    curr_bg = {b["color"]: b["size"] for b in curr.get("backgrounds", [])}
    for color in set(prev_bg) | set(curr_bg):
        ps = prev_bg.get(color, 0)
        cs = curr_bg.get(color, 0)
        if ps != cs:
            changes.append({
                "color": color,
                "type": "background_size_changed",
                "size": {"was": ps, "now": cs},
            })

    return changes
