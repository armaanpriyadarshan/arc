"""
Game 02: Wiring

Mechanics: Click-to-place wires + signal propagation + color mixing + toggle switches

The grid shows power sources (colored squares on the left edge) and light bulbs
(on the right edge) that need to be lit. The player clicks empty cells (ACTION6)
to place wire segments. Clicking an existing wire removes it. Clicking a switch
toggles which power sources are active. Wires propagate the color of whatever
power source they connect to. If two different-colored signals meet at a wire,
they mix (red + blue = purple, red + yellow = orange, blue + yellow = green).
Each bulb requires a specific color — shown as the colored center of the bulb
at all times. The player must wire the grid AND set switches correctly so all
bulbs receive the right color.

Actions:
    ACTION6 = Click to place/remove wire, or toggle a switch

Display:
    Left column:   power sources (colored squares, dimmed when off)
    Right column:  light bulbs (aqua border = unlit, green border = correct,
                   center always shows required color)
    Row 62-63:     status bar showing how many bulbs are correctly lit

Win condition: All bulbs lit with their required colors.

Color key:
    0  = empty / background
    1  = wall / grid lines
    2  = wire (unlit)
    3  = switch OFF frame
    4  = switch ON frame
    5  = HUD background
    6  = blue (signal / source / bulb)
    7  = yellow (signal / source / bulb)
    8  = red (signal / source / bulb)
    9  = purple (mixed: red + blue)
    10 = green (mixed: blue + yellow)
    11 = orange (mixed: red + yellow)
    12 = dimmed source indicator
    13 = bulb border (unlit / aqua)
    14 = bulb border (correct / green)
    15 = HUD label / border
"""

import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    Sprite,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID = 64
BG = 0
GRID_LINE = 1
WIRE_UNLIT = 2
SWITCH_OFF = 3
SWITCH_ON = 4
HUD_BG = 5
BLUE = 6
YELLOW = 7
RED = 8
PURPLE = 9
GREEN = 10
ORANGE = 11
DIMMED = 12
BULB_UNLIT = 13
BULB_CORRECT = 14
HUD_BORDER = 15

# Color mixing rules
COLOR_MIX: Dict[frozenset, int] = {
    frozenset([RED, BLUE]): PURPLE,
    frozenset([RED, YELLOW]): ORANGE,
    frozenset([BLUE, YELLOW]): GREEN,
}

BASE_COLORS = [RED, BLUE, YELLOW]

# ---------------------------------------------------------------------------
# Level config: describes sources, bulbs, switches, and playfield size
# ---------------------------------------------------------------------------
# Each level: (cell_size, sources, bulb_targets, switch_positions)
# - cell_size: pixels per logical cell (the playfield is divided into cells)
# - sources: list of (row, color, initially_on)
# - bulb_targets: list of (row, required_color)
# - switches: list of (cell_row, cell_col, source_index) — which source each switch controls


def _make_level_configs():
    """Return configs for 5 levels of increasing difficulty.

    Key constraint: signal propagation is flood-fill, so each bulb must be
    served by a completely isolated wire network. Two bulbs that need
    different mixes CANNOT share a source — use duplicate sources instead.
    """
    return [
        # Level 0: 1 source, 1 bulb — just connect a wire path (tutorial)
        {
            "cell_size": 6,
            "cols": 8,
            "rows": 8,
            "sources": [(1, RED, True)],
            "bulbs": [(1, RED)],
            "switches": [],
        },
        # Level 1: 2 sources, 2 bulbs — two separate paths, no mixing
        {
            "cell_size": 6,
            "cols": 8,
            "rows": 8,
            "sources": [(1, RED, True), (5, BLUE, True)],
            "bulbs": [(1, RED), (5, BLUE)],
            "switches": [],
        },
        # Level 2: 2 sources, 1 bulb — introduce color mixing (purple = red+blue)
        {
            "cell_size": 6,
            "cols": 8,
            "rows": 8,
            "sources": [(1, RED, True), (5, BLUE, True)],
            "bulbs": [(3, PURPLE)],
            "switches": [],
        },
        # Level 3: mixing + switch — BLUE is off, must toggle to make purple
        {
            "cell_size": 5,
            "cols": 10,
            "rows": 10,
            "sources": [(2, RED, True), (6, BLUE, False)],
            "bulbs": [(4, PURPLE)],
            "switches": [(6, 5, 1)],  # switch controls source 1 (BLUE)
        },
        # Level 4: two separate mixing circuits + switch
        # Each mix uses its own dedicated sources (no sharing).
        # Circuit 1: RED + BLUE -> PURPLE (both on, isolated network)
        # Circuit 2: RED2 + YELLOW -> ORANGE (RED2 is off, needs switch)
        {
            "cell_size": 4,
            "cols": 12,
            "rows": 12,
            "sources": [
                (1, RED, True),      # 0: for purple mix
                (3, BLUE, True),     # 1: for purple mix
                (7, RED, False),     # 2: for orange mix (OFF — needs switch)
                (9, YELLOW, True),   # 3: for orange mix
            ],
            "bulbs": [(2, PURPLE), (8, ORANGE)],
            "switches": [(7, 6, 2)],  # switch controls source 2 (RED, off)
        },
    ]


class WiringGame(ARCBaseGame):
    """Game 02: Wiring — place wires, mix signal colors, toggle switches."""

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = random.Random(seed)
        self._level_configs = _make_level_configs()

        # Per-level state
        self._cell_size = 6
        self._cols = 8
        self._rows = 8
        self._wire_grid: Optional[np.ndarray] = None  # 1 = wire present
        self._signal_grid: Optional[np.ndarray] = None  # propagated color per cell
        self._source_states: List[bool] = []  # on/off per source
        self._sources: List[Tuple[int, int, bool]] = []  # (row, color, on)
        self._bulbs: List[Tuple[int, int]] = []  # (row, required_color)
        self._switches: List[Tuple[int, int]] = []  # (cell_row, cell_col)
        self._switch_source_map: List[int] = []  # which source index each switch controls

        levels = [Level(sprites=[], grid_size=(GRID, GRID)) for _ in self._level_configs]
        camera = Camera(0, 0, GRID, GRID, BG, HUD_BG)
        super().__init__(
            game_id="game_02_wiring",
            levels=levels,
            camera=camera,
            available_actions=[6],
            win_score=len(self._level_configs),
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Level setup
    # ------------------------------------------------------------------

    def on_set_level(self, level):
        level_idx = self._levels.index(level)
        cfg = self._level_configs[level_idx]

        self._cell_size = cfg["cell_size"]
        self._cols = cfg["cols"]
        self._rows = cfg["rows"]
        self._sources = [(r, c, on) for r, c, on in cfg["sources"]]
        self._source_states = [on for _, _, on in cfg["sources"]]
        self._bulbs = list(cfg["bulbs"])
        self._switches = [(r, c) for r, c, *_ in cfg["switches"]]
        self._switch_source_map = [s[2] for s in cfg["switches"]]

        self._wire_grid = np.zeros((self._rows, self._cols), dtype=int)
        self._signal_grid = np.zeros((self._rows, self._cols), dtype=int)

        # Mark source columns and bulb columns as non-placeable
        # Sources are at col 0, bulbs at col (cols-1)

        self._render_full()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _pixel_to_cell(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        """Convert pixel coords to cell (row, col). Returns None if out of playfield."""
        # Playfield starts at pixel (0, 0) and extends cell_size * cols x cell_size * rows
        cs = self._cell_size
        col = px // cs
        row = py // cs
        if 0 <= row < self._rows and 0 <= col < self._cols:
            return (row, col)
        return None

    def _cell_to_pixel(self, row: int, col: int) -> Tuple[int, int]:
        """Top-left pixel of a cell."""
        cs = self._cell_size
        return (col * cs, row * cs)

    # ------------------------------------------------------------------
    # Signal propagation (BFS from active sources)
    # ------------------------------------------------------------------

    def _propagate_signals(self):
        """BFS from each active source through wire cells. Track colors per cell."""
        rows, cols = self._rows, self._cols
        # Each cell can carry a set of base signal colors
        color_sets: List[List[Set[int]]] = [[set() for _ in range(cols)] for _ in range(rows)]
        self._signal_grid = np.zeros((rows, cols), dtype=int)

        # Sources are in column 0
        queue: List[Tuple[int, int]] = []
        for i, (src_row, src_color, _) in enumerate(self._sources):
            if not self._source_states[i]:
                continue
            # Source injects its color into the adjacent wire cell (col 1)
            # But we also need to propagate through col 0 itself
            if src_row < rows:
                color_sets[src_row][0].add(src_color)
                # If there's a wire at col 1, propagate into it
                if cols > 1 and self._wire_grid[src_row, 1]:
                    if src_color not in color_sets[src_row][1]:
                        color_sets[src_row][1].add(src_color)
                        queue.append((src_row, 1))

        # BFS through wires
        visited_states: Set[Tuple[int, int, int]] = set()
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Can propagate into wire cells or bulb column
                    is_wire = self._wire_grid[nr, nc] == 1
                    is_bulb_col = nc == cols - 1
                    if is_wire or is_bulb_col:
                        for color in color_sets[r][c]:
                            state_key = (nr, nc, color)
                            if state_key not in visited_states:
                                visited_states.add(state_key)
                                color_sets[nr][nc].add(color)
                                if is_wire:
                                    queue.append((nr, nc))

        # Resolve mixed colors
        for r in range(rows):
            for c in range(cols):
                cs = color_sets[r][c]
                if not cs:
                    self._signal_grid[r, c] = 0
                elif len(cs) == 1:
                    self._signal_grid[r, c] = next(iter(cs))
                else:
                    # Try to mix
                    self._signal_grid[r, c] = self._mix_colors(cs)

    def _mix_colors(self, colors: Set[int]) -> int:
        """Mix a set of base colors. Returns the resulting palette index."""
        # Reduce pairwise
        colors_list = sorted(colors)
        if len(colors_list) == 2:
            key = frozenset(colors_list)
            return COLOR_MIX.get(key, colors_list[0])
        elif len(colors_list) == 3:
            # Mix first two, then mix result with third
            first = frozenset([colors_list[0], colors_list[1]])
            mid = COLOR_MIX.get(first, colors_list[0])
            second = frozenset([mid, colors_list[2]])
            return COLOR_MIX.get(second, mid)
        return colors_list[0]

    # ------------------------------------------------------------------
    # Check win
    # ------------------------------------------------------------------

    def _check_bulbs(self) -> bool:
        """Return True if every bulb receives its required color."""
        cols = self._cols
        for bulb_row, required_color in self._bulbs:
            # Bulb is at last column
            received = self._signal_grid[bulb_row, cols - 1]
            if received != required_color:
                return False
        return True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_full(self):
        """Clear all sprites and re-render the entire board."""
        level = self.current_level
        # Remove all existing sprites
        for s in list(level._sprites):
            level.remove_sprite(s)

        cs = self._cell_size
        sprite_idx = 0

        def _add(pixels, x, y, name=None, tags=None):
            nonlocal sprite_idx
            n = name or f"s_{sprite_idx}"
            sprite_idx += 1
            s = Sprite(pixels=pixels, name=n, visible=True, collidable=False,
                       tags=tags or [])
            s.set_position(x, y)
            level.add_sprite(s)

        # Draw grid lines (thin borders between cells)
        for r in range(self._rows + 1):
            py = r * cs
            if py < 60:
                line = [[GRID_LINE] * (self._cols * cs)]
                _add(line, 0, py, name=f"hline_{r}")
        for c in range(self._cols + 1):
            px = c * cs
            if px < GRID:
                line = [[GRID_LINE]] * min(self._rows * cs, 60)
                _add(line, px, 0, name=f"vline_{c}")

        # Draw sources (column 0)
        for i, (src_row, src_color, _) in enumerate(self._sources):
            color = src_color if self._source_states[i] else DIMMED
            inner = cs - 2
            if inner < 1:
                inner = 1
            px_block = [[color] * inner for _ in range(inner)]
            x, y = self._cell_to_pixel(src_row, 0)
            _add(px_block, x + 1, y + 1, name=f"source_{i}", tags=["source"])

        # Draw bulbs (last column) — center always shows required color,
        # border indicates lit (green) vs unlit (aqua)
        for i, (bulb_row, req_color) in enumerate(self._bulbs):
            received = self._signal_grid[bulb_row, self._cols - 1]
            lit = received == req_color and received != 0
            border_color = BULB_CORRECT if lit else BULB_UNLIT
            inner = cs - 2
            if inner < 1:
                inner = 1
            # Border
            border_px = [[border_color] * cs for _ in range(cs)]
            x, y = self._cell_to_pixel(bulb_row, self._cols - 1)
            _add(border_px, x, y, name=f"bulb_border_{i}")
            # Inner fill: ALWAYS show the required color so player knows what's needed
            fill_px = [[req_color] * inner for _ in range(inner)]
            _add(fill_px, x + 1, y + 1, name=f"bulb_fill_{i}", tags=["bulb"])

        # Draw switches — border shows on/off state, inner shows source color
        for i, (sw_row, sw_col) in enumerate(self._switches):
            is_on = self._get_switch_visual(i)
            frame_color = SWITCH_ON if is_on else SWITCH_OFF
            inner = cs - 2
            if inner < 1:
                inner = 1
            frame_px = [[frame_color] * cs for _ in range(cs)]
            x, y = self._cell_to_pixel(sw_row, sw_col)
            _add(frame_px, x, y, name=f"switch_{i}", tags=["switch"])
            # Inner marker: show the color of the source this switch controls
            src_idx = self._switch_source_map[i]
            src_color = self._sources[src_idx][1] if src_idx < len(self._sources) else HUD_BORDER
            mark_color = src_color if is_on else DIMMED
            mark_px = [[mark_color] * inner for _ in range(inner)]
            _add(mark_px, x + 1, y + 1, name=f"switch_mark_{i}")

        # Draw wires
        for r in range(self._rows):
            for c in range(self._cols):
                if self._wire_grid[r, c] == 1:
                    signal = self._signal_grid[r, c]
                    color = signal if signal != 0 else WIRE_UNLIT
                    inner = cs - 2
                    if inner < 1:
                        inner = 1
                    wire_px = [[color] * inner for _ in range(inner)]
                    x, y = self._cell_to_pixel(r, c)
                    _add(wire_px, x + 1, y + 1, name=f"wire_{r}_{c}", tags=["wire"])

        # HUD: status bar on rows 62-63
        hud_bg = [[HUD_BG] * GRID for _ in range(4)]
        _add(hud_bg, 0, 60, name="hud_bg")

        # Show how many bulbs are correctly lit
        total = len(self._bulbs)
        correct = 0
        for bulb_row, req_color in self._bulbs:
            received = self._signal_grid[bulb_row, self._cols - 1]
            if received == req_color and received != 0:
                correct += 1

        for i in range(total):
            dot_color = BULB_CORRECT if i < correct else BULB_UNLIT
            dot_px = [[dot_color] * 3 for _ in range(2)]
            _add(dot_px, 6 + i * 5, 62, name=f"status_dot_{i}")

        # Color mixing reference grid (right side of HUD)
        # Each row: [color_A 2x2] [color_B 2x2] [result 2x2]
        recipes = [(RED, BLUE, PURPLE), (RED, YELLOW, ORANGE), (BLUE, YELLOW, GREEN)]
        ref_x = 40  # right side of HUD
        for i, (a, b, result) in enumerate(recipes):
            y = 60 + i
            _add([[a, a, BG, b, b, BG, result, result]], ref_x, y,
                 name=f"mix_ref_{i}", tags=["hud"])

    def _get_switch_visual(self, switch_idx: int) -> bool:
        """Whether switch visually shows ON based on its linked source state."""
        src_idx = self._switch_source_map[switch_idx]
        if src_idx < len(self._source_states):
            return self._source_states[src_idx]
        return False

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def step(self):
        action = self.action.id

        if action == GameAction.ACTION6:
            x = self.action.data.get("x", 0)
            y = self.action.data.get("y", 0)
            coords = self.camera.display_to_grid(x, y)
            if coords:
                gx, gy = coords
                cell = self._pixel_to_cell(gx, gy)
                if cell:
                    row, col = cell

                    # Check if clicked cell is a switch — toggle it
                    switch_idx = self._get_switch_at(row, col)
                    if switch_idx is not None:
                        # Toggle the source this switch controls
                        src_idx = self._switch_source_map[switch_idx]
                        if src_idx < len(self._source_states):
                            self._source_states[src_idx] = not self._source_states[src_idx]
                        self._propagate_signals()
                        self._render_full()
                        if self._check_bulbs():
                            self.next_level()

                    # Otherwise, place/remove wire (not on source or bulb columns)
                    elif 0 < col < self._cols - 1:
                        if self._wire_grid[row, col] == 1:
                            self._wire_grid[row, col] = 0
                        else:
                            self._wire_grid[row, col] = 1
                        self._propagate_signals()
                        self._render_full()
                        if self._check_bulbs():
                            self.next_level()

        self.complete_action()

    def _get_switch_at(self, row: int, col: int) -> Optional[int]:
        """Return switch index if a switch is at (row, col), else None."""
        for i, (sr, sc) in enumerate(self._switches):
            if sr == row and sc == col:
                return i
        return None
