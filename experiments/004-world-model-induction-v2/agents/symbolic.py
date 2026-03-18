"""Convert a 64x64 int grid into a symbolic state representation.

Finds connected components, classifies shapes, determines spatial relations.
Output is a dict the model can reason over without pixel-squinting.
"""

COLOR_NAMES = {
    0: "white", 1: "light_gray", 2: "gray", 3: "dark_gray",
    4: "near_black", 5: "black", 6: "magenta", 7: "pink",
    8: "red", 9: "blue", 10: "light_blue", 11: "yellow",
    12: "orange", 13: "maroon", 14: "green", 15: "purple",
}

Grid = list[list[int]]


def grid_to_symbolic(grid: Grid, min_size: int = 2) -> dict:
    """Convert grid to symbolic state with objects and relations."""
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
                "position": [min_r, min_c],
                "bbox": [min_r, min_c, max_r, max_c],
                "center": [sum(rows) // len(rows), sum(cols) // len(cols)],
            })

    # Sort: small interesting objects first, backgrounds last
    objects.sort(key=lambda o: (o["shape"] == "background", o["size"]))

    # Compute relations between non-background objects
    fg_objects = [o for o in objects if o["shape"] != "background"][:15]
    relations = []
    for i, a in enumerate(fg_objects):
        for b in fg_objects[i+1:]:
            ar, ac = a["center"]
            br, bc = b["center"]
            if ar < br - 5:
                relations.append({"type": "above", "a": a["id"], "b": b["id"]})
            elif ar > br + 5:
                relations.append({"type": "below", "a": a["id"], "b": b["id"]})
            if ac < bc - 5:
                relations.append({"type": "left_of", "a": a["id"], "b": b["id"]})
            elif ac > bc + 5:
                relations.append({"type": "right_of", "a": a["id"], "b": b["id"]})

            # Adjacent check
            a_r1, a_c1, a_r2, a_c2 = a["bbox"]
            b_r1, b_c1, b_r2, b_c2 = b["bbox"]
            if (abs(a_r2 - b_r1) <= 1 or abs(b_r2 - a_r1) <= 1) and not (a_c2 < b_c1 or b_c2 < a_c1):
                relations.append({"type": "adjacent", "a": a["id"], "b": b["id"]})
            if (abs(a_c2 - b_c1) <= 1 or abs(b_c2 - a_c1) <= 1) and not (a_r2 < b_r1 or b_r2 < a_r1):
                relations.append({"type": "adjacent", "a": a["id"], "b": b["id"]})

    # Count backgrounds
    bg_objects = [o for o in objects if o["shape"] == "background"]

    return {
        "grid_size": [h, w],
        "num_objects": len(fg_objects),
        "num_backgrounds": len(bg_objects),
        "objects": fg_objects[:15],
        "backgrounds": [{"color": o["color"], "size": o["size"]} for o in bg_objects],
        "relations": relations[:20],
    }


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
            dist = abs(co["center"][0] - po["center"][0]) + abs(co["center"][1] - po["center"][1])
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
